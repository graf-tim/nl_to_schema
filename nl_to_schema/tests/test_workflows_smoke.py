"""Smoke-Tests für alle 5 Workflows.

Die LLM-Aufrufe werden gemockt, sodass die Tests offline laufen. Sie prüfen
ausschließlich, dass jeder Graph durchläuft und am Ende ein LogicalSchema
mit mindestens einer Tabelle plus ein DDL-String entstehen.
"""
from __future__ import annotations

import pytest

from models.intermediate import (
    Beziehung,
    Entitaet,
    ERAttribut,
    ERBeziehung,
    EREntitaet,
    ERModell,
    RequirementsReport,
)
from models.schema import Column, Type, ForeignKey, LogicalSchema, Table
from models.critic import CriticFinding, CriticReport
from workflows.base import WorkflowState


MIN_REQUIREMENT = (
    "Eine Bibliothek verwaltet Bücher und Autoren. Jedes Buch hat einen Titel "
    "und gehört zu genau einem Autor. Autoren haben einen Namen."
)


def _ok_schema() -> LogicalSchema:
    return LogicalSchema(
        tables=[
            Table(
                name="autor",
                columns=[
                    Column(name="id", type=Type.INTEGER, nullable=False, primary_key=True),
                    Column(name="name", type=Type.VARCHAR, nullable=False),
                ],
            ),
            Table(
                name="buch",
                columns=[
                    Column(name="id", type=Type.INTEGER, nullable=False, primary_key=True),
                    Column(name="titel", type=Type.VARCHAR, nullable=False),
                    Column(name="autor_id", type=Type.INTEGER, nullable=False),
                ],
                foreign_keys=[
                    ForeignKey(from_column="autor_id", references_table="autor", references_column="id"),
                ],
            ),
        ]
    )


def _ok_requirements() -> RequirementsReport:
    return RequirementsReport(
        entitaeten=[
            Entitaet(name="Autor", attribute=["name"], beschreibung="Autor eines Buchs"),
            Entitaet(name="Buch", attribute=["titel"], beschreibung="Bibliotheksbuch"),
        ],
        beziehungen=[
            Beziehung(von="Autor", zu="Buch", kardinalitaet="1:N", beschreibung="schreibt"),
        ],
    )


def _ok_er() -> ERModell:
    return ERModell(
        entitaeten=[
            EREntitaet(
                name="Autor",
                attribute=[
                    ERAttribut(name="id", datentyp="INTEGER", primaerschluessel=True),
                    ERAttribut(name="name", datentyp="VARCHAR"),
                ],
            ),
            EREntitaet(
                name="Buch",
                attribute=[
                    ERAttribut(name="id", datentyp="INTEGER", primaerschluessel=True),
                    ERAttribut(name="titel", datentyp="VARCHAR"),
                ],
            ),
        ],
        beziehungen=[ERBeziehung(von="Autor", zu="Buch", kardinalitaet="1:N")],
    )


def _ok_critic() -> CriticReport:
    return CriticReport(
        qualitaet_ausreichend=True,
        findings=[
            CriticFinding(
                criterion="entitaets_vollstaendigkeit",
                erfuellt=True,
                beschreibung="alle Entitäten vorhanden",
                korrekturanweisung="",
            ),
            CriticFinding(
                criterion="attribut_vollstaendigkeit",
                erfuellt=True,
                beschreibung="alle Attribute vorhanden",
                korrekturanweisung="",
            ),
            CriticFinding(
                criterion="beziehungs_korrektheit",
                erfuellt=True,
                beschreibung="ok",
                korrekturanweisung="",
            ),
            CriticFinding(
                criterion="normalisierung",
                erfuellt=True,
                beschreibung="3NF",
                korrekturanweisung="",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Ersetzt den strukturierten LLM-Aufruf durch deterministische Stubs."""
    from agents import _llm

    schema = _ok_schema()
    rr = _ok_requirements()
    er = _ok_er()
    critic = _ok_critic()

    def _fake(*, output_model, **kwargs):
        if output_model is LogicalSchema:
            return schema
        if output_model is RequirementsReport:
            return rr
        if output_model is ERModell:
            return er
        if output_model is CriticReport:
            return critic
        raise AssertionError(f"unerwartetes output_model: {output_model}")

    monkeypatch.setattr(_llm, "call_structured", _fake)

    # Auch in den importierten Modulen ersetzen
    from agents import generators, analysts, critics

    monkeypatch.setattr(generators, "call_structured", _fake)
    monkeypatch.setattr(analysts, "call_structured", _fake)
    monkeypatch.setattr(critics, "call_structured", _fake)


def _run(name: str) -> WorkflowState:
    from workflows.registry import build

    graph = build(name)
    initial = WorkflowState(anforderungstext=MIN_REQUIREMENT, workflow_name=name)
    raw = graph.invoke(initial)
    if isinstance(raw, WorkflowState):
        return raw
    return WorkflowState.model_validate(raw)


@pytest.mark.parametrize("name", ["sa", "ps", "iw", "oa", "mad"])
def test_workflow_smoke(name):
    state = _run(name)
    assert state.error is None, f"Workflow {name} hatte Fehler: {state.error}"
    assert state.logical_schema is not None
    assert len(state.logical_schema.tables) >= 1
    assert state.final_ddl
    assert "CREATE TABLE" in state.final_ddl
