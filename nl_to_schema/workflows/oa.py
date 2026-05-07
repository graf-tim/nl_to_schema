"""Orchestrator Agent (OA): RA -> CMD -> LSD -> Validator -> Rücksprung an origin_step."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.analysts import (
    conceptual_model_designer,
    logical_schema_designer,
    requirements_analyst,
)
from agents.critics import oa_validator
from workflows.base import WorkflowState, finalize


def _bump_iteration(state: WorkflowState) -> WorkflowState:
    return state.model_copy(update={"iteration": state.iteration + 1})


def route_oa(state: WorkflowState) -> str:
    if state.error is not None:
        return "finalize"
    if state.critic_report is None:
        return "finalize"
    if state.critic_report.qualitaet_ausreichend or state.iteration >= state.max_iter:
        return "finalize"
    origin = state.critic_report.origin_step or "lsd"
    return origin


def _validator_with_iter_bump(state: WorkflowState) -> WorkflowState:
    new_state = oa_validator(state)
    return _bump_iteration(new_state)


def build_oa():
    g = StateGraph(WorkflowState)
    g.add_node("ra", requirements_analyst)
    g.add_node("cmd", conceptual_model_designer)
    g.add_node("lsd", logical_schema_designer)
    g.add_node("validator", _validator_with_iter_bump)
    g.add_node("finalize", finalize)

    g.set_entry_point("ra")
    g.add_edge("ra", "cmd")
    g.add_edge("cmd", "lsd")
    g.add_edge("lsd", "validator")
    g.add_conditional_edges(
        "validator",
        route_oa,
        {
            "ra": "ra",
            "cmd": "cmd",
            "lsd": "lsd",
            "finalize": "finalize",
        },
    )
    g.add_edge("finalize", END)
    return g.compile()
