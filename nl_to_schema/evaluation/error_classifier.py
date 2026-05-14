"""LLM-basierte Fehlerklassifikation für semantische Abweichungen.

Dieses Modul ist KEIN Evaluator mit Score-Gewicht — es klassifiziert
diagnostisch, warum ein Referenzelement nicht im generierten Schema
gefunden wurde. Die Klassifikation läuft nur für eine manuell ausgewählte
Stichprobe (typisch ~20 Testfälle) und nicht für alle Evaluations-Runs.

Drei Fehlerklassen:
  - SEMANTISCHE_UNVOLLSTAENDIGKEIT: Entität oder Attribut fehlt vollständig,
    Attribute auch nicht verstreut in anderen Tabellen auffindbar.
  - NORMALISIERUNGSPROBLEM: Attribute der Referenztabelle tauchen verstreut
    in anderen generierten Tabellen auf (falsche Zusammenfassung/Aufteilung).
  - STRUKTURELLES_PROBLEM: Tabellen sind vorhanden, aber die FK-Beziehung
    fehlt oder ist falsch deklariert.
"""
from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from agents._llm import call_structured
from models.schema import LogicalSchema
from workflows.base import get_error_classifier_llm


logger = logging.getLogger(__name__)


class Fehlerklasse(str, Enum):
    SEMANTISCHE_UNVOLLSTAENDIGKEIT = "SEMANTISCHE_UNVOLLSTAENDIGKEIT"
    NORMALISIERUNGSPROBLEM = "NORMALISIERUNGSPROBLEM"
    STRUKTURELLES_PROBLEM = "STRUKTURELLES_PROBLEM"


class Fehlerklassifikation(BaseModel):
    fehlerklasse: Fehlerklasse = Field(
        description="Eine der drei Fehlerklassen, die die Abweichung am besten beschreibt."
    )
    begruendung: str = Field(
        description="Kurze Begründung (ein Satz) für die gewählte Klasse."
    )


CLASSIFIER_SYSTEM_PROMPT = """Du bist ein erfahrener Datenbankarchitekt. Deine Aufgabe ist
es zu klassifizieren, warum ein Element des Referenzschemas im generierten
Schema fehlt oder nicht korrekt gematcht wurde.

Du erhältst:
- Den ursprünglichen Anforderungstext
- Das vollständige Referenzschema (Soll)
- Das vollständige generierte Schema (Ist)
- Die konkrete Abweichung: ein Element (Tabelle, Attribut oder Beziehung),
  das im generierten Schema fehlt.

Wähle GENAU EINE der drei folgenden Klassen:

1) SEMANTISCHE_UNVOLLSTAENDIGKEIT
   Das Element fehlt vollständig. Bei Tabellen: keine semantisch äquivalente
   Tabelle im generierten Schema. Bei Attributen: weder in der erwarteten
   Tabelle noch in irgendeiner anderen Tabelle vorhanden.
   Beispiel: Referenz hat "patients" mit "insurance_number"; im generierten
   Schema fehlen weder eine Patient-Tabelle noch eine versicherungsnumern-
   Spalte irgendwo.

2) NORMALISIERUNGSPROBLEM
   Das Element existiert konzeptionell im generierten Schema, ist aber
   anders verteilt: Attribute einer Referenztabelle sind über mehrere
   generierte Tabellen verstreut (zu stark normalisiert), oder mehrere
   Referenztabellen wurden zu einer einzigen zusammengeführt (zu wenig
   normalisiert).
   Beispiel: Referenz hat separate "addresses"-Tabelle, im generierten
   Schema sind die Adress-Felder direkt in "customers" eingebettet.

3) STRUKTURELLES_PROBLEM
   Die beteiligten Tabellen sind korrekt vorhanden, aber eine FK-Beziehung
   zwischen ihnen fehlt oder ist falsch (referenziert die falsche Spalte
   oder existiert nicht).
   Beispiel: "orders" und "customers" sind beide da, aber "orders" hat keinen
   FK auf "customers" — obwohl der Anforderungstext eine solche Beziehung
   explizit nennt.

Vorgehen:
<analyse>
1. Lies die Abweichung sorgfältig: handelt es sich um eine Tabelle, ein
   Attribut oder eine Beziehung?
2. Prüfe das generierte Schema:
   - Tabelle: existiert eine semantisch äquivalente Tabelle (auch unter
     anderem Namen)?
   - Attribut: existiert das Attribut in der erwarteten Tabelle? In einer
     anderen Tabelle? Gar nicht?
   - Beziehung: existieren die beiden beteiligten Tabellen? Existiert
     ein FK zwischen ihnen?
3. Wähle die Klasse, die das Muster am besten beschreibt.
</analyse>

Gib das Ergebnis ausschließlich im geforderten strukturierten Format zurück."""


def _format_abweichung(abweichung: dict) -> str:
    typ = abweichung.get("typ", "?")
    element = abweichung.get("element", "?")
    return f"Typ: {typ}\nElement: {element}"


def classify_abweichung(
    *,
    anforderungstext: str,
    reference: LogicalSchema,
    generated: LogicalSchema,
    abweichung: dict,
    workflow_name: str = "error_classifier",
    iteration: int = 0,
) -> dict:
    """Klassifiziert eine einzelne Abweichung via LLM.

    Rückgabe:
        {"fehlerklasse": "...", "begruendung": "..."}
        oder bei Fehler:
        {"fehlerklasse": None, "begruendung": None, "error": "..."}
    """
    user_message = (
        "Anforderungstext:\n"
        f"---\n{anforderungstext}\n---\n\n"
        "Referenzschema (Soll):\n"
        f"{reference.model_dump_json(indent=2)}\n\n"
        "Generiertes Schema (Ist):\n"
        f"{generated.model_dump_json(indent=2)}\n\n"
        "Abweichung (Referenzelement, das nicht im generierten Schema gefunden wurde):\n"
        f"{_format_abweichung(abweichung)}\n\n"
        "Klassifiziere diese Abweichung gemäß der Definition deines Systemprompts."
    )
    try:
        result = call_structured(
            workflow_name=workflow_name,
            iteration=iteration,
            agent_name="error_classifier",
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_message=user_message,
            output_model=Fehlerklassifikation,
            llm_factory=get_error_classifier_llm,
        )
    except Exception as exc:
        logger.exception("Fehlerklassifikation fehlgeschlagen: %s", exc)
        return {
            "fehlerklasse": None,
            "begruendung": None,
            "error": f"classifier_failed: {exc}",
        }

    return {
        "fehlerklasse": result.fehlerklasse.value,
        "begruendung": result.begruendung,
    }
