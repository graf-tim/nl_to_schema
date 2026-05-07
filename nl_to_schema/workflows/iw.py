"""Iterativer Workflow (IW): generator -> critic -> [generator | finalize]."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.critics import iw_critic
from agents.generators import iw_generator
from workflows.base import WorkflowState, finalize


def route_iw(state: WorkflowState) -> str:
    if state.error is not None:
        return "finalize"
    if state.critic_report is None:
        return "finalize"
    if state.critic_report.qualitaet_ausreichend or state.iteration >= state.max_iter:
        return "finalize"
    return "generator"


def build_iw():
    g = StateGraph(WorkflowState)
    g.add_node("generator", iw_generator)
    g.add_node("critic", iw_critic)
    g.add_node("finalize", finalize)
    g.set_entry_point("generator")
    g.add_edge("generator", "critic")
    g.add_conditional_edges(
        "critic",
        route_iw,
        {"generator": "generator", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()
