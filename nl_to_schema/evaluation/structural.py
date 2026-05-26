"""Strukturelle Evaluation."""
from __future__ import annotations

from ddl_generator import validate_ddl_structural
from models.schema import LogicalSchema


def structural_evaluation(ddl: str, schema: LogicalSchema) -> dict:
    """Liefert den Strukturscore plus absolute Zählwerte für die Auswertung."""
    raw = validate_ddl_structural(ddl, schema)

    syntax_gate = 1 if raw["syntaktisch_korrekt"] else 0
    fk_total = raw["fk_referenzen_gesamt"]
    fk_invalid = raw["fk_referenzen_ungueltig"]
    fk_gate = 1 if (fk_total > 0 and fk_invalid == 0) else 0
    pk_rate = raw["pk_vollstaendigkeit"]

    score = syntax_gate * fk_gate * pk_rate

    return {
        "score": score,
        "syntax_gate": syntax_gate,
        "fk_gate": fk_gate,
        "pk_rate": pk_rate,
        "pk_tabellen_gesamt": raw["pk_tabellen_gesamt"],
        "pk_tabellen_mit_pk": raw["pk_tabellen_mit_pk"],
        "fk_referenzen_gesamt": fk_total,
        "fk_referenzen_ungueltig": fk_invalid,
    }
