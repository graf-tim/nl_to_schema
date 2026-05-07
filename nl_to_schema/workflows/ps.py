"""Pipeline Sequenz (PS): RA -> CMD -> LSD -> finalize."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.analysts import (
    conceptual_model_designer,
    logical_schema_designer,
    requirements_analyst,
)
from workflows.base import WorkflowState, finalize


def build_ps():
    g = StateGraph(WorkflowState)
    g.add_node("ra", requirements_analyst)
    g.add_node("cmd", conceptual_model_designer)
    g.add_node("lsd", logical_schema_designer)
    g.add_node("finalize", finalize)
    g.set_entry_point("ra")
    g.add_edge("ra", "cmd")
    g.add_edge("cmd", "lsd")
    g.add_edge("lsd", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
