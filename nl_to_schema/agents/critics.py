"""Critic-Agenten und MAD-Moderator."""
from __future__ import annotations

import logging
from typing import Optional

from agents._llm import call_structured
from ddl_generator import generate_ddl, validate_ddl_structural
from models.critic import CriticFinding, CriticReport
from models.schema import LogicalSchema
from workflows.base import (
    CRITIC_C1_SYSTEM_PROMPT,
    CRITIC_C2_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    MODERATOR_SYSTEM_PROMPT,
    OA_VALIDATOR_SYSTEM_PROMPT,
    WorkflowState,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strukturelle Findings (deterministisch via sqlparse + Schema-Inspektion)
# ---------------------------------------------------------------------------

def _structural_findings(schema: LogicalSchema) -> list[CriticFinding]:
    try:
        ddl = generate_ddl(schema)
        ddl_error: Optional[str] = None
    except Exception as exc:
        ddl = ""
        ddl_error = str(exc)

    result = validate_ddl_structural(ddl, schema)

    syn_ok = result["syntaktisch_korrekt"] and ddl_error is None
    syn = CriticFinding(
        criterion="syntaktische_korrektheit",
        erfuellt=syn_ok,
        beschreibung=(
            "DDL ist syntaktisch valide."
            if syn_ok
            else f"DDL nicht erzeugbar/parsebar: {ddl_error or 'sqlparse-Fehler'}"
        ),
        korrekturanweisung=(
            ""
            if syn_ok
            else "Stelle sicher, dass jede Tabelle Spalten hat und keine "
            "Zirkelreferenzen vorliegen."
        ),
    )

    pk_ok = result["pk_vollstaendigkeit"] >= 1.0
    pk = CriticFinding(
        criterion="pk_vollstaendigkeit",
        erfuellt=pk_ok,
        beschreibung=(
            f"PK-Quote: {result['pk_vollstaendigkeit']:.2f}"
        ),
        korrekturanweisung=(
            ""
            if pk_ok
            else "Vergib jeder Tabelle ohne Primärschlüssel eine geeignete "
            "PK-Spalte (Surrogat-id INTEGER)."
        ),
    )

    fk_ok = result["fk_integritaet"] >= 1.0
    fk = CriticFinding(
        criterion="fk_integritaet",
        erfuellt=fk_ok,
        beschreibung=(
            f"FK-Integritätsquote: {result['fk_integritaet']:.2f}"
        ),
        korrekturanweisung=(
            ""
            if fk_ok
            else "Korrigiere FK-Referenzen, deren Zieltabelle/-spalte nicht im "
            "Schema existiert."
        ),
    )
    return [syn, pk, fk]


# ---------------------------------------------------------------------------
# LLM-basiertes Subset (inhaltliche Kriterien)
# ---------------------------------------------------------------------------

def _content_critic_user_message(state: WorkflowState) -> str:
    if state.logical_schema is None:
        raise ValueError("CriticAgent: kein LogicalSchema vorhanden.")
    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
        "",
        "LogicalSchema (zu bewerten):",
        state.logical_schema.model_dump_json(indent=2),
    ]
    if state.requirements_report is not None:
        parts += [
            "",
            "RequirementsReport (Kontext):",
            state.requirements_report.model_dump_json(indent=2),
        ]
    if state.er_modell is not None:
        parts += [
            "",
            "ERModell (Kontext):",
            state.er_modell.model_dump_json(indent=2),
        ]
    parts += [
        "",
        "Bewerte das Schema gemäß den Kriterien deines Systemprompts.",
    ]
    return "\n".join(parts)


def _hard_criteria_ok(findings: list[CriticFinding]) -> bool:
    hard = {
        "syntaktische_korrektheit",
        "pk_vollstaendigkeit",
        "fk_integritaet",
        "entitaets_vollstaendigkeit",
    }
    by_criterion = {f.criterion: f for f in findings}
    return all(
        by_criterion.get(c) is not None and by_criterion[c].erfuellt for c in hard
    )


def _merge_findings(
    structural: list[CriticFinding], llm_report: CriticReport
) -> list[CriticFinding]:
    """Strukturelle Findings überschreiben die LLM-Findings für die 3 harten Strukturkriterien."""
    structural_criteria = {f.criterion for f in structural}
    merged: list[CriticFinding] = list(structural)
    for f in llm_report.findings:
        if f.criterion not in structural_criteria:
            merged.append(f)
    return merged


def _run_content_critic(
    state: WorkflowState,
    *,
    agent_name: str,
    system_prompt: str,
) -> CriticReport:
    user_message = _content_critic_user_message(state)
    return call_structured(
        workflow_name=state.workflow_name,
        iteration=state.iteration,
        agent_name=agent_name,
        system_prompt=system_prompt,
        user_message=user_message,
        output_model=CriticReport,
    )


# ---------------------------------------------------------------------------
# Critic-Funktionen
# ---------------------------------------------------------------------------

def iw_critic(state: WorkflowState) -> WorkflowState:
    if state.logical_schema is None:
        return state.model_copy(update={"error": "iw_critic: kein logical_schema"})
    structural = _structural_findings(state.logical_schema)
    try:
        llm_report = _run_content_critic(
            state, agent_name="iw_critic", system_prompt=CRITIC_SYSTEM_PROMPT
        )
    except Exception as exc:
        logger.exception("[%s] iw_critic LLM-Aufruf fehlgeschlagen.", state.workflow_name)
        report = CriticReport(qualitaet_ausreichend=False, findings=structural)
        return state.model_copy(
            update={"critic_report": report, "error": f"iw_critic_failed: {exc}"}
        )

    findings = _merge_findings(structural, llm_report)
    report = CriticReport(
        qualitaet_ausreichend=_hard_criteria_ok(findings),
        findings=findings,
    )
    return state.model_copy(update={"critic_report": report})


def oa_validator(state: WorkflowState) -> WorkflowState:
    if state.logical_schema is None:
        return state.model_copy(update={"error": "oa_validator: kein logical_schema"})
    structural = _structural_findings(state.logical_schema)
    try:
        llm_report = _run_content_critic(
            state,
            agent_name="oa_validator",
            system_prompt=OA_VALIDATOR_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.exception("[%s] oa_validator LLM-Aufruf fehlgeschlagen.", state.workflow_name)
        report = CriticReport(
            qualitaet_ausreichend=False, findings=structural, origin_step="lsd"
        )
        return state.model_copy(
            update={"critic_report": report, "error": f"oa_validator_failed: {exc}"}
        )

    findings = _merge_findings(structural, llm_report)
    hard_ok = _hard_criteria_ok(findings)

    if hard_ok:
        origin: Optional[str] = None
    else:
        # Frühester betroffener Schritt: ra > cmd > lsd.
        origin = llm_report.origin_step
        if origin is None:
            # Heuristik anhand der harten Findings:
            failed = {f.criterion for f in findings if not f.erfuellt}
            if "entitaets_vollstaendigkeit" in failed:
                origin = "ra"
            elif (
                "beziehungs_korrektheit" in failed
                or "attribut_vollstaendigkeit" in failed
            ):
                origin = "cmd"
            else:
                origin = "lsd"

    report = CriticReport(
        qualitaet_ausreichend=hard_ok,
        findings=findings,
        origin_step=origin,
    )
    return state.model_copy(update={"critic_report": report})


def mad_critic_c1(state: WorkflowState) -> WorkflowState:
    if state.logical_schema is None:
        return state.model_copy(update={"error": "mad_critic_c1: kein logical_schema"})
    structural = _structural_findings(state.logical_schema)
    try:
        llm_report = _run_content_critic(
            state, agent_name="mad_critic_c1", system_prompt=CRITIC_C1_SYSTEM_PROMPT
        )
    except Exception as exc:
        logger.exception("[%s] mad_critic_c1 fehlgeschlagen.", state.workflow_name)
        report = CriticReport(qualitaet_ausreichend=False, findings=structural)
        new_verlauf = list(state.diskussionsverlauf) + [
            f"[iter={state.iteration}] C1: Fehler {exc}"
        ]
        return state.model_copy(
            update={
                "critic_report_c1": report,
                "diskussionsverlauf": new_verlauf,
                "error": f"mad_critic_c1_failed: {exc}",
            }
        )

    findings = _merge_findings(structural, llm_report)
    report = CriticReport(
        qualitaet_ausreichend=_hard_criteria_ok(findings),
        findings=findings,
    )
    n_unmet = sum(1 for f in findings if not f.erfuellt)
    new_verlauf = list(state.diskussionsverlauf) + [
        f"[iter={state.iteration}] C1: {n_unmet} unerfüllte Findings."
    ]
    return state.model_copy(
        update={
            "critic_report_c1": report,
            "diskussionsverlauf": new_verlauf,
        }
    )


def mad_critic_c2(state: WorkflowState) -> WorkflowState:
    if state.logical_schema is None:
        return state.model_copy(update={"error": "mad_critic_c2: kein logical_schema"})
    structural = _structural_findings(state.logical_schema)
    try:
        llm_report = _run_content_critic(
            state, agent_name="mad_critic_c2", system_prompt=CRITIC_C2_SYSTEM_PROMPT
        )
    except Exception as exc:
        logger.exception("[%s] mad_critic_c2 fehlgeschlagen.", state.workflow_name)
        report = CriticReport(qualitaet_ausreichend=False, findings=structural)
        new_verlauf = list(state.diskussionsverlauf) + [
            f"[iter={state.iteration}] C2: Fehler {exc}"
        ]
        return state.model_copy(
            update={
                "critic_report_c2": report,
                "diskussionsverlauf": new_verlauf,
                "error": f"mad_critic_c2_failed: {exc}",
            }
        )

    findings = _merge_findings(structural, llm_report)
    report = CriticReport(
        qualitaet_ausreichend=_hard_criteria_ok(findings),
        findings=findings,
    )
    n_unmet = sum(1 for f in findings if not f.erfuellt)
    new_verlauf = list(state.diskussionsverlauf) + [
        f"[iter={state.iteration}] C2: {n_unmet} unerfüllte Findings."
    ]
    return state.model_copy(
        update={
            "critic_report_c2": report,
            "diskussionsverlauf": new_verlauf,
        }
    )


def mad_moderator(state: WorkflowState) -> WorkflowState:
    """Synthetisiert das finale LogicalSchema aus dem Diskussionsverlauf."""
    if state.logical_schema is None:
        return state.model_copy(update={"error": "mad_moderator: kein logical_schema"})

    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
        "",
        "Aktuelles LogicalSchema:",
        state.logical_schema.model_dump_json(indent=2),
    ]
    if state.critic_report_c1 is not None:
        parts += [
            "",
            "CriticReport C1 (Normalisierung/Schlüssel):",
            state.critic_report_c1.model_dump_json(indent=2),
        ]
    if state.critic_report_c2 is not None:
        parts += [
            "",
            "CriticReport C2 (Vollständigkeit):",
            state.critic_report_c2.model_dump_json(indent=2),
        ]
    if state.diskussionsverlauf:
        parts += [
            "",
            "Diskussionsverlauf:",
            "\n".join(state.diskussionsverlauf),
        ]
    parts += [
        "",
        "Erzeuge das finale, synthetisierte LogicalSchema.",
    ]
    user_message = "\n".join(parts)

    try:
        final_schema = call_structured(
            workflow_name=state.workflow_name,
            iteration=state.iteration,
            agent_name="mad_moderator",
            system_prompt=MODERATOR_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=LogicalSchema,
        )
    except Exception as exc:
        logger.exception("[%s] mad_moderator fehlgeschlagen.", state.workflow_name)
        return state.model_copy(update={"error": f"mad_moderator_failed: {exc}"})

    # Final-Critic anhand strukturell + Aggregation der Critics berechnen.
    structural = _structural_findings(final_schema)
    findings = list(structural)
    if state.critic_report_c1 is not None:
        for f in state.critic_report_c1.findings:
            if f.criterion not in {x.criterion for x in findings}:
                findings.append(f)
    if state.critic_report_c2 is not None:
        for f in state.critic_report_c2.findings:
            if f.criterion not in {x.criterion for x in findings}:
                findings.append(f)
    final_report = CriticReport(
        qualitaet_ausreichend=_hard_criteria_ok(findings),
        findings=findings,
    )

    new_verlauf = list(state.diskussionsverlauf) + [
        f"[iter={state.iteration}] Moderator: finales Schema mit "
        f"{len(final_schema.tables)} Tabelle(n)."
    ]
    return state.model_copy(
        update={
            "logical_schema": final_schema,
            "final_critic_report": final_report,
            "diskussionsverlauf": new_verlauf,
        }
    )
