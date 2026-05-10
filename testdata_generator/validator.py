"""Validator: Stichprobenauswahl, Konsistenzprüfung und Statistiken.

Aufruf:
    python validator.py --input output/testcases.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models import LogicalSchema, Testfall


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

DEFAULT_PER_STUFE = 5


def random_sample_per_stufe(
    testfaelle: list[Testfall],
    per_stufe: int = DEFAULT_PER_STUFE,
) -> list[Testfall]:
    """Wählt pro Stufe zufällig `per_stufe` Testfälle (oder alle, falls weniger).

    Sortiert die Auswahl pro Stufe nach ID, damit der Validierungsbericht eine
    nachvollziehbare Reihenfolge hat (auch wenn die Auswahl selbst zufällig ist).
    """
    by_stufe: dict[int, list[Testfall]] = {}
    for tc in testfaelle:
        by_stufe.setdefault(tc.stufe, []).append(tc)

    sample: list[Testfall] = []
    for stufe in sorted(by_stufe.keys()):
        items = by_stufe[stufe]
        n = min(per_stufe, len(items))
        chosen = random.sample(items, n)
        chosen.sort(key=lambda tc: tc.id)
        sample.extend(chosen)
    return sample


# ---------------------------------------------------------------------------
# Konsistenzprüfung
# ---------------------------------------------------------------------------

def validiere_schema_konsistenz(testfall: Testfall) -> list[str]:
    """Prüft strukturelle Konsistenz des Referenzschemas.

    Rückgabe: Liste von Fehlermeldungen (leer = ok).
    """
    fehler: list[str] = []
    schema: LogicalSchema = testfall.referenzschema
    table_columns: dict[str, set[str]] = {}

    for table in schema.tables:
        if table.name in table_columns:
            fehler.append(f"Doppelter Tabellenname: '{table.name}'")
        else:
            table_columns[table.name] = {c.name for c in table.columns}

        if not any(c.primary_key for c in table.columns):
            fehler.append(f"Tabelle '{table.name}' hat keinen Primärschlüssel.")

    for table in schema.tables:
        own_columns = table_columns.get(table.name, set())
        for fk in table.foreign_keys:
            if fk.from_column not in own_columns:
                fehler.append(
                    f"FK in '{table.name}': Quellspalte '{fk.from_column}' "
                    f"existiert nicht in der eigenen Tabelle."
                )
            if fk.references_table not in table_columns:
                fehler.append(
                    f"FK in '{table.name}': Zieltabelle '{fk.references_table}' existiert nicht."
                )
            elif fk.references_column not in table_columns[fk.references_table]:
                fehler.append(
                    f"FK in '{table.name}': Zielspalte '{fk.references_column}' "
                    f"existiert nicht in '{fk.references_table}'."
                )
    return fehler


# ---------------------------------------------------------------------------
# Statistiken
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def berechne_statistiken(testfaelle: list[Testfall]) -> dict[str, Any]:
    pro_stufe: dict[int, list[Testfall]] = {1: [], 2: [], 3: []}
    for tc in testfaelle:
        pro_stufe.setdefault(tc.stufe, []).append(tc)

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    stats_pro_stufe: dict[int, dict[str, float]] = {}
    for stufe, items in pro_stufe.items():
        woerter = [_word_count(tc.anforderungstext) for tc in items]
        tabellen = [len(tc.referenzschema.tables) for tc in items]
        spalten = [
            sum(len(t.columns) for t in tc.referenzschema.tables) for tc in items
        ]
        stats_pro_stufe[stufe] = {
            "anzahl": len(items),
            "anforderungstext_woerter_avg": round(_avg([float(w) for w in woerter]), 1),
            "tabellen_avg": round(_avg([float(t) for t in tabellen]), 2),
            "spalten_avg": round(_avg([float(s) for s in spalten]), 2),
        }

    n = len(testfaelle)
    return {
        "gesamt": n,
        "pro_stufe": stats_pro_stufe,
    }


# ---------------------------------------------------------------------------
# Auto-Prüfung: alle Testfälle als CSV
# ---------------------------------------------------------------------------

AUTO_PRUEFUNG_FIELDS = [
    "testfall_id",
    "stufe",
    "domaene",
    "anzahl_tabellen",
    "anzahl_spalten",
    "anzahl_foreign_keys",
    "automatische_pruefung",  # "OK" oder "FEHLER"
    "anzahl_fehler",
    "fehler",  # ' | '-getrennte Liste der konkreten Fehlermeldungen
]


def auto_pruefung_zeile(testfall: Testfall) -> dict[str, Any]:
    fehler = validiere_schema_konsistenz(testfall)
    n_tabellen = len(testfall.referenzschema.tables)
    n_spalten = sum(len(t.columns) for t in testfall.referenzschema.tables)
    n_fks = sum(len(t.foreign_keys) for t in testfall.referenzschema.tables)
    return {
        "testfall_id": testfall.id,
        "stufe": testfall.stufe,
        "domaene": testfall.domaene,
        "anzahl_tabellen": n_tabellen,
        "anzahl_spalten": n_spalten,
        "anzahl_foreign_keys": n_fks,
        "automatische_pruefung": "OK" if not fehler else "FEHLER",
        "anzahl_fehler": len(fehler),
        "fehler": " | ".join(fehler),
    }


def schreibe_auto_pruefung_csv(testfaelle: list[Testfall], path: Path) -> dict[str, int]:
    """Führt die automatische Prüfung über alle Testfälle aus und schreibt ein CSV.

    Rückgabe: dict mit Zusammenfassung {gesamt, ok, fehler}.
    """
    rows = [auto_pruefung_zeile(tc) for tc in testfaelle]
    rows.sort(key=lambda r: r["testfall_id"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_PRUEFUNG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_fehler = sum(1 for r in rows if r["automatische_pruefung"] == "FEHLER")
    return {
        "gesamt": len(rows),
        "ok": len(rows) - n_fehler,
        "fehler": n_fehler,
    }


# ---------------------------------------------------------------------------
# Stichprobe als CSV (für manuelle Prüfung)
# ---------------------------------------------------------------------------

STICHPROBE_FIELDS = [
    "testfall_id",
    "stufe",
    "domaene",
    "anforderungstext",
    "schema",
    # Manuelle Prüfung — leer zum Ausfüllen
    "inhaltliche_vollstaendigkeit",
    "datentypen_angemessen",
    "normalisierung_3nf",
    "komplexitaetsstufe_konform",
    "anmerkungen",
]


def schreibe_stichprobe_csv(stichprobe: list[Testfall], path: Path) -> None:
    """Schreibt die Stichprobe als CSV mit leeren Spalten für die manuelle Prüfung."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STICHPROBE_FIELDS)
        writer.writeheader()
        for tc in stichprobe:
            schema_lines = [
                _format_table_inline(t).strip() for t in tc.referenzschema.tables
            ]
            writer.writerow(
                {
                    "testfall_id": tc.id,
                    "stufe": tc.stufe,
                    "domaene": tc.domaene,
                    "anforderungstext": tc.anforderungstext,
                    "schema": "\n".join(schema_lines),
                    "inhaltliche_vollstaendigkeit": "",
                    "datentypen_angemessen": "",
                    "normalisierung_3nf": "",
                    "komplexitaetsstufe_konform": "",
                    "anmerkungen": "",
                }
            )


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------

def _format_table_inline(table) -> str:
    parts: list[str] = []
    for col in table.columns:
        flags: list[str] = []
        if col.primary_key:
            flags.append("PK")
        if not col.nullable:
            flags.append("NOT NULL")
        flag_str = " " + " ".join(flags) if flags else ""
        parts.append(f"{col.name} {col.type.value}{flag_str}")
    fk_strs = [
        f"FK→{fk.references_table}.{fk.references_column}({fk.from_column})"
        for fk in table.foreign_keys
    ]
    if fk_strs:
        parts.extend(fk_strs)
    return f"  {table.name} ({', '.join(parts)})"


def _format_block(testfall: Testfall, fehler: list[str]) -> str:
    lines: list[str] = []
    bar = "═" * 46
    lines.append(bar)
    lines.append(f"{testfall.id} | Stufe {testfall.stufe} | {testfall.domaene}")
    lines.append(bar)
    lines.append("ANFORDERUNGSTEXT:")
    lines.append(testfall.anforderungstext)
    lines.append("")
    lines.append("SCHEMA:")
    for t in testfall.referenzschema.tables:
        lines.append(_format_table_inline(t))
    lines.append("")
    if fehler:
        lines.append("AUTOMATISCHE PRÜFUNG: FEHLER")
        for fmsg in fehler:
            lines.append(f"  - {fmsg}")
    else:
        lines.append("AUTOMATISCHE PRÜFUNG: OK")
    lines.append("")
    lines.append("MANUELLE PRÜFUNG:")
    lines.append("  [ ] Referenzschema inhaltlich korrekt")
    lines.append("  [ ] Übereinstimmung Anforderung ↔ Schema")
    lines.append("  [ ] 3NF eingehalten")
    lines.append("  [ ] Komplexitätsstufe korrekt zugeordnet")
    lines.append("  Anmerkungen: ___________________________")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_testcases(path: Path) -> list[Testfall]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Erwartet eine JSON-Liste in {path}, erhalten {type(data)}.")
    out: list[Testfall] = []
    for item in data:
        try:
            out.append(Testfall.model_validate(item))
        except ValidationError as exc:
            tc_id = item.get("id", "?") if isinstance(item, dict) else "?"
            print(f"WARN: konnte Testfall {tc_id} nicht validieren: {exc}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "testcases.json",
        help="Pfad zur konsolidierten testcases.json",
    )
    parser.add_argument(
        "--per-stufe",
        type=int,
        default=DEFAULT_PER_STUFE,
        help=f"Stichprobengröße pro Stufe (Default: {DEFAULT_PER_STUFE}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Eingabedatei nicht gefunden: {args.input}", file=sys.stderr)
        return 2

    testfaelle = _load_testcases(args.input)
    if not testfaelle:
        print("Keine Testfälle in der Eingabedatei.", file=sys.stderr)
        return 2

    # Stichprobe: zufällig pro Stufe
    stichprobe = random_sample_per_stufe(testfaelle, per_stufe=args.per_stufe)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stichprobe_path = OUTPUT_DIR / "stichprobe.json"
    with stichprobe_path.open("w", encoding="utf-8") as f:
        json.dump(
            [tc.model_dump(mode="json") for tc in stichprobe],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Stichprobe zusätzlich als CSV mit leeren Prüfspalten für die manuelle Bewertung
    stichprobe_csv_path = OUTPUT_DIR / "stichprobe.csv"
    schreibe_stichprobe_csv(stichprobe, stichprobe_csv_path)

    # Validierungsbericht
    bericht_lines: list[str] = []
    bericht_lines.append(
        f"Zufallsstichprobe: {args.per_stufe} Testfälle pro Stufe"
    )
    bericht_lines.append(f"Ausgewählte IDs: {', '.join(tc.id for tc in stichprobe)}")
    bericht_lines.append("")
    for tc in stichprobe:
        fehler = validiere_schema_konsistenz(tc)
        bericht_lines.append(_format_block(tc, fehler))

    bericht_path = OUTPUT_DIR / "validierungsbericht.txt"
    with bericht_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(bericht_lines))

    # Auto-Prüfung über ALLE Testfälle als CSV
    auto_csv_path = OUTPUT_DIR / "auto_pruefung.csv"
    auto_summary = schreibe_auto_pruefung_csv(testfaelle, auto_csv_path)

    # Statistiken
    stats = berechne_statistiken(testfaelle)

    # Konsolen-Output
    print(f"Stichprobe (JSON):      {stichprobe_path} ({len(stichprobe)} Testfälle)")
    print(f"Stichprobe (CSV):       {stichprobe_csv_path}")
    print(f"Validierungsbericht:    {bericht_path}")
    print(
        f"Auto-Prüfung CSV:       {auto_csv_path} "
        f"(gesamt={auto_summary['gesamt']}, "
        f"ok={auto_summary['ok']}, "
        f"fehler={auto_summary['fehler']})"
    )
    print()
    print("Statistiken:")
    print(f"  Gesamt: {stats['gesamt']}")
    for stufe in (1, 2, 3):
        s = stats["pro_stufe"].get(stufe)
        if s is None:
            continue
        print(
            f"  Stufe {stufe}: n={s['anzahl']}, "
            f"Wörter avg={s['anforderungstext_woerter_avg']}, "
            f"Tabellen avg={s['tabellen_avg']}, "
            f"Spalten avg={s['spalten_avg']}"
        )

    # Exit-Code 1 wenn irgendein Testfall in der Auto-Prüfung gefehlt hat
    if auto_summary["fehler"]:
        print(
            f"\nAuto-Prüfung: {auto_summary['fehler']}/{auto_summary['gesamt']} "
            f"Testfälle haben strukturelle Fehler. Details in {auto_csv_path.name}."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
