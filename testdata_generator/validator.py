"""Validator: Stichprobenauswahl, Konsistenzprüfung und Statistiken.

Aufruf:
    python validator.py --input output/testcases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models import LogicalSchema, Testfall


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

STICHPROBE_IDS: list[str] = [
    # Stufe 1
    "TC005", "TC012", "TC019", "TC026", "TC033",
    # Stufe 2
    "TC038", "TC044", "TC050", "TC056", "TC062",
    # Stufe 3
    "TC070", "TC076", "TC082", "TC088", "TC094",
]


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


def _has_m_n_bridge(testfall: Testfall) -> bool:
    """Heuristik: Tabelle mit zusammengesetztem PK aus 2+ FK-Spalten = Brückentabelle."""
    for table in testfall.referenzschema.tables:
        pk_cols = {c.name for c in table.columns if c.primary_key}
        if len(pk_cols) < 2:
            continue
        fk_cols = {fk.from_column for fk in table.foreign_keys}
        if len(pk_cols & fk_cols) >= 2:
            return True
    return False


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
    mit_bruecke = sum(1 for tc in testfaelle if _has_m_n_bridge(tc))
    return {
        "gesamt": n,
        "pro_stufe": stats_pro_stufe,
        "anteil_mit_m_n_bruecke": round(mit_bruecke / n, 3) if n else 0.0,
    }


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
        parts.append(f"{col.name} {col.data_type.value}{flag_str}")
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
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Eingabedatei nicht gefunden: {args.input}", file=sys.stderr)
        return 2

    testfaelle = _load_testcases(args.input)
    by_id = {tc.id: tc for tc in testfaelle}

    # Stichprobe ziehen
    stichprobe: list[Testfall] = []
    fehlende: list[str] = []
    for sid in STICHPROBE_IDS:
        if sid in by_id:
            stichprobe.append(by_id[sid])
        else:
            fehlende.append(sid)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stichprobe_path = OUTPUT_DIR / "stichprobe.json"
    with stichprobe_path.open("w", encoding="utf-8") as f:
        json.dump(
            [tc.model_dump(mode="json") for tc in stichprobe],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Validierungsbericht
    bericht_lines: list[str] = []
    if fehlende:
        bericht_lines.append(
            f"WARNUNG: folgende Stichproben-IDs fehlen im Datensatz: {', '.join(fehlende)}"
        )
        bericht_lines.append("")
    for tc in stichprobe:
        fehler = validiere_schema_konsistenz(tc)
        bericht_lines.append(_format_block(tc, fehler))

    bericht_path = OUTPUT_DIR / "validierungsbericht.txt"
    with bericht_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(bericht_lines))

    # Statistiken
    stats = berechne_statistiken(testfaelle)

    # Konsolen-Output
    print(f"Stichprobe gespeichert: {stichprobe_path} ({len(stichprobe)} Testfälle)")
    print(f"Validierungsbericht:    {bericht_path}")
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
    print(f"  Anteil mit M:N-Brückentabelle: {stats['anteil_mit_m_n_bruecke']}")

    # Anzahl Stichproben mit Konsistenzfehler
    fehlerhaft = sum(1 for tc in stichprobe if validiere_schema_konsistenz(tc))
    if fehlerhaft:
        print(f"  Stichproben-Fehler: {fehlerhaft}/{len(stichprobe)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
