"""Multi-Agent Discussion (MAD): Architekt <-> {C1, C2} -> Moderator -> finalize.

Hinweis: Die beiden Critics werden konzeptionell parallel ausgeführt (sehen
beide das gleiche Schema, ohne Information voneinander). Aufgrund der
Pydantic-State-Architektur ohne Reducer für die einzelnen Felder
(diskussionsverlauf, critic_report_c1, critic_report_c2) werden sie hier
sequenziell innerhalb eines einzigen Knotens aufgerufen. Das logische Ergebnis
ist identisch zu echter Parallelausführung.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.critics import mad_critic_c1, mad_critic_c2, mad_moderator
from agents.generators import mad_architekt
from workflows.base import WorkflowState, finalize


def critics_node(state: WorkflowState) -> WorkflowState:
    after_c1 = mad_critic_c1(state)
    after_c2 = mad_critic_c2(after_c1)
    return after_c2


def route_mad(state: WorkflowState) -> str:
    if state.error is not None:
        return "moderator"
    if state.iteration >= state.max_iter:
        return "moderator"
    return "critics"


def build_mad():
    g = StateGraph(WorkflowState)
    g.add_node("architekt", mad_architekt)
    g.add_node("critics", critics_node)
    g.add_node("moderator", mad_moderator)
    g.add_node("finalize", finalize)

    g.set_entry_point("architekt")
    g.add_conditional_edges(
        "architekt",
        route_mad,
        {"critics": "critics", "moderator": "moderator"},
    )
    g.add_edge("critics", "architekt")
    g.add_edge("moderator", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
