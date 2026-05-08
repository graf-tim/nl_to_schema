"""Alle Prompts für die Testfall-Generierung.

Sprache: Deutsch. Werte werden über str.format(...) eingesetzt.
"""
from __future__ import annotations

KONVENTIONEN = """
KONVENTIONEN
  
TYPEN:
- INTEGER: für ganze Zahlen (z.B. Alter, Anzahl)
- VARCHAR: für kurze Texte (z.B. Name, E-Mail)
- TEXT: für längere Texte (z.B. Beschreibung, Kommentar)
- DATE: für Datumsangaben (z.B. Geburtsdatum, Ausleihdatum)
- BOOLEAN: für Wahrheitswerte (z.B. ist_aktiv)
- DECIMAL: für Dezimalzahlen (z.B. Preis, Bewertung)
- TIMESTAMP: für Zeitstempel (z.B. Erstellungszeit)

CONSTRAINTS:
- Pflichtfelder:    nullable: false
- Optionale Felder: nullable: true
- Primärschlüssel:  primary_key: true

NAMENSKONVENTIONEN:

TABELLEN (Logisches Schema):
- Plural, snake_case, Kleinschreibung, plural, englisch
- Umlaute ersetzen: ä→ae, ö→oe, ü→ue, ß→ss
- Korrekt: "persons", "books", "authors", "borrowings"

PRIMÄRSCHLÜSSEL:
- Format: id INTEGER PRIMARY KEY
- Immer die ERSTE Spalte der Tabelle
- Korrekt: "id"

FREMDSCHLÜSSEL:
- Format: {referenz_singular}_id INTEGER NOT NULL
- Korrekt: "person_id", "book_id"

JUNCTION TABLES (N:M Beziehungen):
- Format: {tabelle_a}_{tabelle_b} (alphabetisch sortiert)
- Korrekt: "author_books", "course_students"

ATTRIBUTE:
- snake_case, Kleinschreibung
- Kurze Namen in englisch:
  "E-Mail-Adresse" → "email"
  "Vorname"     → "firstname"
  "Nachname"      → "lastname"
  "Geburtsdatum"     → "birthdate"
  "Erscheinungsjahr" → "publicationyear"
  "Ausleihdatum"  → "checkoutdate"
  "Rückgabedatum"    → "returndate"
"""

SYSTEM_PROMPT = """Du bist ein erfahrener Datenbankarchitekt und Anforderungsanalyst mit fundiertem
Wissen in relationaler Datenbankmodellierung und Normalisierungstheorie.

Deine Aufgabe ist es, synthetische Testfälle für die Evaluation von
KI-Systemen zur automatischen SQL-Schema-Generierung zu erstellen.

Jeder Testfall besteht aus:
1. Einer natürlichsprachlichen Anforderungsbeschreibung (deutsch)
2. Einem vollständigen logischen Referenzschema in 3NF
3. Einer Begründung der Designentscheidungen

Qualitätsanforderungen an das Referenzschema:
- Muss exakt der Dritten Normalform (3NF) entsprechen
- Jede Tabelle hat mindestens einen Primärschlüssel
- Alle Fremdschlüsselreferenzen sind konsistent und existieren im Schema
- Datentypen sind fachlich angemessen
- Namenskonventionen sind eingehalten
- M:N-Beziehungen werden durch eine Zwischentabelle aufgelöst

Qualitätsanforderungen an den Anforderungstext:
- Alltagssprache, keine technischen Datenbankbegriffe
- Keine explizite Nennung von Tabellen, Primärschlüsseln oder Fremdschlüsseln
- Keine Hinweise auf Normalisierung
- Enthält implizite Informationen, die Domänenwissen erfordern

""" + KONVENTIONEN + """

Antworte ausschliesslich mit einem validen JSON-Objekt.
Keine Markdown-Backticks, keine Präambel, kein abschliessender Text."""


USER_PROMPT_STUFE_1 = """Erstelle Testfall {id} der Komplexitätsstufe 1 für die Domäne: {domäne}

Spezifikation Stufe 1:
- Anzahl Tabellen im finalen Schema: 3 bis 5
- Beziehungstypen: ausschliesslich 1:N (keine M:N)
- Semantische Ambiguität: keine — alle Anforderungen eindeutig formuliert
- Domänenkomplexität: alltagsnah, kein Fachwissen erforderlich
- Anforderungstextlänge: 80 bis 150 Wörter

Der Anforderungstext soll klar und eindeutig sein. Keine offenen Formulierungen,
keine impliziten Abhängigkeiten, die mehrere Interpretationen zulassen.
"""


USER_PROMPT_STUFE_2 = """Erstelle Testfall {id} der Komplexitätsstufe 2 für die Domäne: {domäne}

Spezifikation Stufe 2:
- Anzahl Tabellen im finalen Schema: 5 bis 8
- Beziehungstypen: mindestens eine 1:N- und mindestens eine M:N-Beziehung
- Semantische Ambiguität: 1 bis 2 Stellen mit offenen Formulierungen,
  die ohne Domänenwissen nicht eindeutig aufgelöst werden können
- Domänenkomplexität: grundlegendes Fachwissen erforderlich
- Anforderungstextlänge: 150 bis 250 Wörter

Beispiel für eine offene Formulierung: "Ein Mitarbeiter kann mehrere Projekte
betreuen" — ist das eine direkte Zuweisung oder über eine Rolle vermittelt?

Markiere im Feld "ambiguitaeten" der Begründung explizit, welche Stellen
im Anforderungstext bewusst ambig gestaltet wurden und wie du sie aufgelöst hast.
"""


USER_PROMPT_STUFE_3 = """Erstelle Testfall {id} der Komplexitätsstufe 3 für die Domäne: {domäne}

Spezifikation Stufe 3:
- Anzahl Tabellen im finalen Schema: 8 oder mehr
- Beziehungstypen: mehrere 1:N- und M:N-Beziehungen.
- Semantische Ambiguität: substanziell — mindestens 3 Stellen mit offenen
  oder widersprüchlichen Formulierungen
- Domaenenkomplexität: spezialisiertes Fachwissen erforderlich;
  der Text enthält Fachbegriffe der jeweiligen Domäne
- Anforderungstextlänge: 250 bis 400 Wörter

Die Ambiguitäten sollen realistisch sein — so wie ein Fachexperte ohne
Datenbankwissen eine Anforderung formulieren würde. Typische Ambiguitäten:
unklare Kardinalitäten, implizite Rollenzuweisungen, fehlende Pflichtfeldangaben,
nicht explizit genannte Zwischenentitäten.

Markiere im Feld "ambiguitaeten" alle ambigen Stellen und erkläre die
getroffene Designentscheidung.
"""


USER_PROMPTS_BY_STUFE: dict[int, str] = {
    1: USER_PROMPT_STUFE_1,
    2: USER_PROMPT_STUFE_2,
    3: USER_PROMPT_STUFE_3,
}


def render_user_prompt(*, stufe: int, id: str, domaene: str) -> str:
    template = USER_PROMPTS_BY_STUFE.get(stufe)
    if template is None:
        raise ValueError(f"Unbekannte Stufe: {stufe}")
    return template.format(id=id, domäne=domaene)
