from pydantic import BaseModel
from typing import Optional, Literal


class CriticFinding(BaseModel):
    criterion: Literal[
        "syntaktische_korrektheit",
        "pk_vollstaendigkeit",
        "fk_integritaet",
        "entitaets_vollstaendigkeit",
        "attribut_vollstaendigkeit",
        "beziehungs_korrektheit",
        "normalisierung",
    ]
    erfuellt: bool
    beschreibung: str
    # Bei erfuellt=True wird das Feld vom LLM gerne weggelassen (statt als ""
    # geschickt) — Default sorgt dafür, dass die Validierung trotzdem durchgeht.
    korrekturanweisung: str = ""


class CriticReport(BaseModel):
    qualitaet_ausreichend: bool
    findings: list[CriticFinding]
    origin_step: Optional[Literal["ra", "cmd", "lsd"]] = None
