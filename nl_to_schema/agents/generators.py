"""Generierende Agenten: erzeugen ein LogicalSchema."""
from __future__ import annotations

import logging

from agents._llm import call_structured
from models.schema import LogicalSchema
from workflows.base import LSD_SYSTEM_PROMPT, WorkflowState


logger = logging.getLogger(__name__)


def _safe(state: WorkflowState, update: dict) -> WorkflowState:
    return state.model_copy(update=update)


def _llm_call(state: WorkflowState, agent_name: str, user_message: str) -> WorkflowState:
    try:
        schema = call_structured(
            workflow_name=state.workflow_name,
            iteration=state.iteration,
            agent_name=agent_name,
            system_prompt=LSD_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=LogicalSchema,
        )
        return _safe(state, {"logical_schema": schema})
    except Exception as exc:
        logger.exception("[%s] %s fehlgeschlagen.", state.workflow_name, agent_name)
        return _safe(state, {"error": f"{agent_name}_failed: {exc}"})


def sa_generator(state: WorkflowState) -> WorkflowState:
    """Single-Agent: erzeugt LogicalSchema direkt aus dem Anforderungstext."""
    user_message = (
        "Anforderungstext:\n"
        f"---\n{state.anforderungstext}\n---\n\n"
        "Erzeuge ein vollständiges, normalisiertes LogicalSchema."
    )
    return _llm_call(state, "sa_generator", user_message)


def iw_generator(state: WorkflowState) -> WorkflowState:
    """Iterativer Generator: nutzt ab Iteration 1 das aktuelle Schema und den CriticReport."""
    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
    ]
    if state.iteration > 0 and state.logical_schema is not None:
        parts += [
            "",
            "Aktuelles LogicalSchema (zu überarbeiten):",
            state.logical_schema.model_dump_json(indent=2),
        ]
    if state.iteration > 0 and state.critic_report is not None:
        parts += [
            "",
            "CriticReport mit Findings (insbesondere erfuellt=False sind harte Fehler):",
            state.critic_report.model_dump_json(indent=2),
            "",
            "Adressiere jedes Finding mit erfuellt=False explizit anhand der korrekturanweisung.",
        ]
    parts += [
        "",
        "Erzeuge das vollständige neue LogicalSchema.",
    ]
    user_message = "\n".join(parts)

    new_state = _llm_call(state, "iw_generator", user_message)
    return _safe(new_state, {"iteration": state.iteration + 1})


def mad_architekt(state: WorkflowState) -> WorkflowState:
    """MAD-Architekt: nutzt ab Iteration 1 beide CriticReports."""
    parts = [
        "Anforderungstext:",
        "---",
        state.anforderungstext,
        "---",
    ]
    if state.iteration > 0 and state.logical_schema is not None:
        parts += [
            "",
            "Aktuelles LogicalSchema (zu überarbeiten):",
            state.logical_schema.model_dump_json(indent=2),
        ]
    if state.iteration > 0 and state.critic_report_c1 is not None:
        parts += [
            "",
            "CriticReport C1 (Normalisierung/Schlüssel):",
            state.critic_report_c1.model_dump_json(indent=2),
        ]
    if state.iteration > 0 and state.critic_report_c2 is not None:
        parts += [
            "",
            "CriticReport C2 (Vollständigkeit):",
            state.critic_report_c2.model_dump_json(indent=2),
        ]
    if state.iteration > 0:
        parts += [
            "",
            "Adressiere JEDES Finding mit erfuellt=False aus beiden Reports explizit.",
        ]
    parts += [
        "",
        "Erzeuge das vollständige neue LogicalSchema.",
    ]
    user_message = "\n".join(parts)

    new_state = _llm_call(state, "mad_architekt", user_message)
    new_verlauf = list(state.diskussionsverlauf)
    if new_state.logical_schema is not None:
        new_verlauf.append(
            f"[iter={state.iteration}] Architekt: "
            f"Schema mit {len(new_state.logical_schema.tables)} Tabelle(n)."
        )
    return _safe(
        new_state,
        {
            "iteration": state.iteration + 1,
            "diskussionsverlauf": new_verlauf,
        },
    )
