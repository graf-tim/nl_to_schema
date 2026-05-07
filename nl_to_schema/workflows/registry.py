"""Zentrale Registry der 5 Workflow-Architekturen."""
from __future__ import annotations

from workflows.iw import build_iw
from workflows.mad import build_mad
from workflows.oa import build_oa
from workflows.ps import build_ps
from workflows.sa import build_sa


WORKFLOWS = {
    "sa": build_sa,
    "ps": build_ps,
    "iw": build_iw,
    "oa": build_oa,
    "mad": build_mad,
}


def build(name: str):
    if name not in WORKFLOWS:
        raise ValueError(
            f"Unbekannter Workflow '{name}'. Verfügbar: {list(WORKFLOWS.keys())}"
        )
    return WORKFLOWS[name]()
