"""Strukturelle Evaluation."""
from __future__ import annotations

from ddl_generator import validate_ddl_structural
from models.schema import LogicalSchema


def structural_score(ddl: str, schema: LogicalSchema) -> float:
    """Strukturscore in [0,1]. Syntaxfehler => 0.0 (Gate-Kriterium)."""
    result = validate_ddl_structural(ddl, schema)
    if not result["syntaktisch_korrekt"]:
        return 0.0
    return (result["pk_vollstaendigkeit"] + result["fk_integritaet"]) / 2.0


def structural_details(ddl: str, schema: LogicalSchema) -> dict:
    """Vollständiges Detail-Dict inklusive Score."""
    result = validate_ddl_structural(ddl, schema)
    result["score"] = structural_score(ddl, schema)
    return result
