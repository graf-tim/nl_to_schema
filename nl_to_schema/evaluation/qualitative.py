"""Qualitative Evaluation per LLM-as-Judge."""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from agents._llm import call_structured
from models.schema import LogicalSchema


logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """Du bist ein erfahrener Datenbank-Architekt und beurteilst die
qualitative Güte eines logischen Relationenschemas anhand einer festen Rubrik.

Bewerte die folgenden drei Dimensionen, jede auf einer Skala von 1 (sehr schlecht)
bis 5 (sehr gut):

1) bezeichnungsqualitaet: Sind Tabellen- und Spaltennamen semantisch treffend,
   sprechend und domain-passend? (Keine generischen Namen wie "data1".)

2) datentypangemessenheit: Sind die gewählten Datentypen fachlich korrekt
   (z.B. DATE statt VARCHAR für Datumsspalten, DECIMAL für Geld, BOOLEAN für Flags)?

3) konventionskonsistenz: Sind Namenskonventionen einheitlich
   (snake_case vs. CamelCase, Singular vs. Plural, Konsistenz bei id-Spalten,
   einheitliche Pluralisierung bei Brückentabellen)?

Vorgehen:
<analyse>
1. Gehe alle Tabellen einzeln durch und sammle Beobachtungen pro Dimension.
2. Vergib pro Dimension einen Integer-Score 1..5 mit kurzer Begründung.
</analyse>

Gib das Ergebnis ausschließlich im geforderten strukturierten Format zurück."""


class QualitativeJudgement(BaseModel):
    bezeichnungsqualitaet: int = Field(ge=1, le=5)
    bezeichnungsqualitaet_begruendung: str
    datentypangemessenheit: int = Field(ge=1, le=5)
    datentypangemessenheit_begruendung: str
    konventionskonsistenz: int = Field(ge=1, le=5)
    konventionskonsistenz_begruendung: str


def qualitative_score(schema: LogicalSchema, anforderungstext: str) -> dict:
    """Liefert ein Dict mit normierten Scores in [0,1] und Detail-Begründungen."""
    user_message = (
        "Anforderungstext:\n"
        f"---\n{anforderungstext}\n---\n\n"
        "LogicalSchema (zu bewerten):\n"
        f"{schema.model_dump_json(indent=2)}\n\n"
        "Bewerte das Schema nach der Rubrik."
    )
    try:
        judgement = call_structured(
            workflow_name="evaluation",
            iteration=0,
            agent_name="llm_as_judge",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=QualitativeJudgement,
        )
    except Exception as exc:
        logger.exception("LLM-as-Judge fehlgeschlagen: %s", exc)
        return {
            "bezeichnungsqualitaet": 0.0,
            "datentypangemessenheit": 0.0,
            "konventionskonsistenz": 0.0,
            "qualitative_score": 0.0,
            "error": f"judge_failed: {exc}",
        }

    norm = lambda s: (s - 1) / 4.0
    b = norm(judgement.bezeichnungsqualitaet)
    d = norm(judgement.datentypangemessenheit)
    k = norm(judgement.konventionskonsistenz)
    return {
        "bezeichnungsqualitaet": b,
        "bezeichnungsqualitaet_begruendung": judgement.bezeichnungsqualitaet_begruendung,
        "datentypangemessenheit": d,
        "datentypangemessenheit_begruendung": judgement.datentypangemessenheit_begruendung,
        "konventionskonsistenz": k,
        "konventionskonsistenz_begruendung": judgement.konventionskonsistenz_begruendung,
        "qualitative_score": (b + d + k) / 3.0,
    }
