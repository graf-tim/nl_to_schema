"""Analyse-Agenten: Requirements Analyst, Conceptual Model Designer, Logical Schema Designer."""
from __future__ import annotations

import logging

from agents._llm import call_structured
from models.intermediate import RequirementsReport, ERModell
from models.schema import LogicalSchema
from workflows.base import (
    CMD_SYSTEM_PROMPT,
    LSD_SYSTEM_PROMPT,
    RA_SYSTEM_PROMPT,
    WorkflowState,
)


logger = logging.getLogger(__name__)


def requirements_analyst(state: WorkflowState) -> WorkflowState:
    parts = ["Anforderungstext:", "---", state.anforderungstext, "---"]

    if state.iteration > 0 and state.requirements_report is not None:
        parts += [
            "",
            "Dein vorheriger RequirementsReport (zu überarbeiten):",
            state.requirements_report.model_dump_json(indent=2),
        ]

    if state.iteration > 0 and state.critic_report is not None:
        ra_findings = [
            f for f in state.critic_report.findings
            if f.criterion == "entitaets_vollstaendigkeit" and not f.erfuellt
        ]
        if ra_findings:
            parts += [
                "",
                "Validator-Findings, die auf deinen RequirementsReport zurückgehen:",
                *[
                    f"- [{f.criterion}] {f.beschreibung} → {f.korrekturanweisung}"
                    for f in ra_findings
                ],
                "",
                "Adressiere diese Findings explizit im überarbeiteten RequirementsReport.",
            ]

    parts += ["", "Erzeuge einen vollständigen RequirementsReport."]
    user_message = "\n".join(parts)

    try:
        report = call_structured(
            workflow_name=state.workflow_name,
            iteration=state.iteration,
            agent_name="requirements_analyst",
            system_prompt=RA_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=RequirementsReport,
        )
        return state.model_copy(update={"requirements_report": report})
    except Exception as exc:
        logger.exception("[%s] requirements_analyst fehlgeschlagen.", state.workflow_name)
        return state.model_copy(update={"error": f"requirements_analyst_failed: {exc}"})


def conceptual_model_designer(state: WorkflowState) -> WorkflowState:
    if state.requirements_report is None:
        return state.model_copy(
            update={"error": "conceptual_model_designer: kein requirements_report vorhanden"}
        )

    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
        "",
        "RequirementsReport:",
        state.requirements_report.model_dump_json(indent=2),
    ]

    if state.iteration > 0 and state.er_modell is not None:
        parts += [
            "",
            "Dein vorheriges ERModell (zu überarbeiten):",
            state.er_modell.model_dump_json(indent=2),
        ]

    if state.iteration > 0 and state.critic_report is not None:
        cmd_findings = [
            f for f in state.critic_report.findings
            if f.criterion in {"attribut_vollstaendigkeit", "beziehungs_korrektheit"}
            and not f.erfuellt
        ]
        if cmd_findings:
            parts += [
                "",
                "Validator-Findings, die auf dein ERModell zurückgehen:",
                *[
                    f"- [{f.criterion}] {f.beschreibung} → {f.korrekturanweisung}"
                    for f in cmd_findings
                ],
                "",
                "Adressiere diese Findings explizit im überarbeiteten ERModell.",
            ]

    parts += ["", "Erzeuge ein vollständiges ERModell."]
    user_message = "\n".join(parts)

    try:
        er = call_structured(
            workflow_name=state.workflow_name,
            iteration=state.iteration,
            agent_name="conceptual_model_designer",
            system_prompt=CMD_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=ERModell,
        )
        return state.model_copy(update={"er_modell": er})
    except Exception as exc:
        logger.exception(
            "[%s] conceptual_model_designer fehlgeschlagen.", state.workflow_name
        )
        return state.model_copy(
            update={"error": f"conceptual_model_designer_failed: {exc}"}
        )


def logical_schema_designer(state: WorkflowState) -> WorkflowState:
    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
    ]
    if state.er_modell is not None:
        parts += [
            "",
            "ERModell:",
            state.er_modell.model_dump_json(indent=2),
        ]
    if state.critic_report is not None and state.critic_report.findings:
        parts += [
            "",
            "Vorheriger CriticReport (insbesondere erfuellt=False sind harte Fehler):",
            state.critic_report.model_dump_json(indent=2),
            "",
            "Adressiere jedes Finding mit erfuellt=False explizit.",
        ]
    parts += [
        "",
        "Erzeuge das vollständige LogicalSchema.",
    ]
    user_message = "\n".join(parts)
    try:
        schema = call_structured(
            workflow_name=state.workflow_name,
            iteration=state.iteration,
            agent_name="logical_schema_designer",
            system_prompt=LSD_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=LogicalSchema,
        )
        return state.model_copy(update={"logical_schema": schema})
    except Exception as exc:
        logger.exception(
            "[%s] logical_schema_designer fehlgeschlagen.", state.workflow_name
        )
        return state.model_copy(
            update={"error": f"logical_schema_designer_failed: {exc}"}
        )
