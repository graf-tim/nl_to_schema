"""Single Agent (SA): ein einziger Generator -> finalize."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.generators import sa_generator
from workflows.base import WorkflowState, finalize


def build_sa():
    g = StateGraph(WorkflowState)
    g.add_node("generator", sa_generator)
    g.add_node("finalize", finalize)
    g.set_entry_point("generator")
    g.add_edge("generator", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
