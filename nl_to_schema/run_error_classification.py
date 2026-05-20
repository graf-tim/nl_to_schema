"""Vorbereitung einer manuellen Fehlerklassifikation.

Aus den persistierten Evaluations-Outputs (`results/<wf>/<id>.json`) wird eine
stratifizierte Auswahl der Testfälle mit den **schlechtesten durchschnittlichen
Gesamt-Scores** über alle Workflows gezogen — das sind die Testfälle, bei
denen die Workflows am stärksten gestrauchelt sind und die deshalb am
aufschlussreichsten für eine manuelle Fehler-Diagnose sind.

Pro gewähltem Testfall werden zwei CSV-Dateien geschrieben:

  - `manual_review_selection.csv`  — eine Zeile pro Testfall, mit Kontext
    (Domäne, Anforderungstext, durchschnittlicher Gesamt-Score).
  - `manual_review.csv`             — eine Zeile pro `(testfall, workflow)`
    mit allen Scores plus einer zusammengefassten Abweichungs-Spalte, plus
    leeren Spalten `fehlerklasse` und `begruendung` zum manuellen Ausfüllen.

Aufruf (Auto-Selektion, Default 3/3/4 = 10 Testfälle):
    python run_error_classification.py \\
        --testcases ../testdata_generator/output/testcases.json \\
        --results-dir results/

Mit expliziter Liste:
    python run_error_classification.py \\
        --testfall-ids TC042 TC067 TC089 \\
        --testcases ../testdata_generator/output/testcases.json \\
        --results-dir results/

Optional:
    --workflows sa ps iw oa mad   # Default: alle vorhandenen Unterordner
    --per-stufe 3 3 4              # Stichprobengrößen Stufe 1, 2, 3
    --force                        # bestehende manual_review.csv überschreiben
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _load_testcases_index(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {tc["id"]: tc for tc in data}


def _load_result(results_dir: Path, workflow: str, testfall_id: str) -> dict | None:
    path = results_dir / workflow / f"{testfall_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _discover_workflows(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    return sorted(
        p.name for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _collect_scores_per_testfall(
    results_dir: Path,
    testcases_index: dict[str, dict],
    workflows: list[str],
) -> dict[str, list[float]]:
    """Sammelt pro Testfall die `score_gesamt`-Werte aus allen vorhandenen Workflows."""
    per_tc: dict[str, list[float]] = {}
    for wf in workflows:
        wf_dir = results_dir / wf
        if not wf_dir.is_dir():
            continue
        for path in sorted(wf_dir.glob("TC*.json")):
            tc_id = path.stem
            if tc_id not in testcases_index:
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            score = data.get("score_gesamt")
            if score is None:
                continue
            per_tc.setdefault(tc_id, []).append(float(score))
    return per_tc


def _auto_select_testfaelle(
    results_dir: Path,
    testcases_index: dict[str, dict],
    workflows: list[str],
    per_stufe: dict[int, int],
) -> tuple[list[str], dict[str, float]]:
    """Stratifizierte Auswahl nach durchschnittlichem Gesamt-Score (aufsteigend).

    Pro Testfall wird der Mittelwert über `score_gesamt` aller vorhandenen
    Workflows berechnet. Pro Stufe werden die N Testfälle mit den
    **niedrigsten** Durchschnittswerten ausgewählt — also die schwierigsten.

    Rückgabe: (gewählte IDs in stabiler Reihenfolge, dict[id → avg_score]).
    """
    scores = _collect_scores_per_testfall(results_dir, testcases_index, workflows)
    avg_per_tc: dict[str, float] = {
        tc_id: (sum(s) / len(s)) if s else 0.0 for tc_id, s in scores.items()
    }

    by_stufe: dict[int, list[tuple[str, float]]] = {}
    for tc_id, avg in avg_per_tc.items():
        stufe = testcases_index[tc_id].get("stufe")
        if stufe is None:
            continue
        by_stufe.setdefault(int(stufe), []).append((tc_id, avg))

    selected: list[str] = []
    for stufe in sorted(by_stufe.keys()):
        n = per_stufe.get(stufe, 0)
        if n <= 0:
            continue
        # Niedrigster avg zuerst (schlechtester Score); Tie-Breaker: ID alphabetisch
        ranked = sorted(by_stufe[stufe], key=lambda x: (x[1], x[0]))
        for tc_id, _ in ranked[:n]:
            selected.append(tc_id)

    return selected, avg_per_tc


SELECTION_FIELDS = [
    "testfall_id",
    "stufe",
    "domaene",
    "avg_score_gesamt",
    "anforderungstext",
]

REVIEW_FIELDS = [
    "testfall_id",
    "stufe",
    "domaene",
    "workflow",
    # Scores
    "score_strukturell",
    "score_semantisch",
    "score_gesamt",
    # Struktur-Diagnose
    "syntax_gate",
    "fk_gate",
    "pk_rate",
    "fk_referenzen_ungueltig",
    # Semantik-Diagnose
    "entitaet_f1",
    "attribut_f1",
    "beziehung_recall",
    # Abweichungen
    "abweichungen_total",
    "abweichungen",
    # leer für manuelle Eingabe
    "fehlerklasse",
    "begruendung",
]


def _format_abweichungen(abweichungen: list[dict]) -> str:
    """Verdichtet die Abweichungs-Liste zu einer pipe-getrennten Kompakt-Spalte."""
    if not abweichungen:
        return ""
    parts: list[str] = []
    for a in abweichungen:
        typ = a.get("typ", "?")
        element = a.get("element", "?")
        parts.append(f"{typ}:{element}")
    return " | ".join(parts)


def _write_selection_overview(
    out_path: Path,
    testfall_ids: list[str],
    testcases_index: dict[str, dict],
    avg_per_tc: dict[str, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        for tc_id in testfall_ids:
            tc = testcases_index.get(tc_id) or {}
            writer.writerow(
                {
                    "testfall_id": tc_id,
                    "stufe": tc.get("stufe"),
                    "domaene": tc.get("domaene"),
                    "avg_score_gesamt": round(avg_per_tc.get(tc_id, 0.0), 3),
                    "anforderungstext": tc.get("anforderungstext", ""),
                }
            )


def _write_manual_review_json(
    out_path: Path,
    testfall_ids: list[str],
    testcases_index: dict[str, dict],
    avg_per_tc: dict[str, float],
    results_dir: Path,
    workflows: list[str],
) -> int:
    """Schreibt die vollständigen Resultate als JSON.

    Pro Testfall: Anforderungstext, Referenzschema und pro Workflow die
    vollständigen Eval-Daten plus das generierte Schema und DDL.

    Rückgabe: Anzahl Testfälle im JSON.
    """
    payload: list[dict[str, Any]] = []
    for tc_id in testfall_ids:
        tc = testcases_index.get(tc_id) or {}
        workflows_data: dict[str, dict[str, Any]] = {}
        for wf in workflows:
            result = _load_result(results_dir, wf, tc_id)
            if result is None:
                continue
            state = result.get("state") or {}
            workflows_data[wf] = {
                "score_strukturell": result.get("score_strukturell"),
                "score_semantisch": result.get("score_semantisch"),
                "score_gesamt": result.get("score_gesamt"),
                "detail_strukturell": result.get("detail_strukturell"),
                "detail_semantisch": result.get("detail_semantisch"),
                "abweichungen": result.get("abweichungen"),
                "generated_schema": state.get("logical_schema"),
                "final_ddl": state.get("final_ddl"),
                "workflow_error": state.get("error"),
            }
        payload.append(
            {
                "testfall_id": tc_id,
                "stufe": tc.get("stufe"),
                "domaene": tc.get("domaene"),
                "anforderungstext": tc.get("anforderungstext"),
                "referenzschema": tc.get("referenzschema"),
                "avg_score_gesamt": round(avg_per_tc.get(tc_id, 0.0), 3),
                "workflows": workflows_data,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload)


def _write_manual_review(
    out_path: Path,
    testfall_ids: list[str],
    testcases_index: dict[str, dict],
    results_dir: Path,
    workflows: list[str],
) -> int:
    """Schreibt die Detail-CSV: eine Zeile pro (testfall, workflow).

    Rückgabe: Anzahl Zeilen.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for tc_id in testfall_ids:
            tc = testcases_index.get(tc_id) or {}
            for wf in workflows:
                result = _load_result(results_dir, wf, tc_id)
                if result is None:
                    continue
                detail_s = result.get("detail_strukturell") or {}
                detail_sem = result.get("detail_semantisch") or {}
                entitaet = detail_sem.get("entitaet") or {}
                attribut = detail_sem.get("attribut") or {}
                beziehung = detail_sem.get("beziehung") or {}
                abweichungen = result.get("abweichungen") or []

                writer.writerow(
                    {
                        "testfall_id": tc_id,
                        "stufe": tc.get("stufe"),
                        "domaene": tc.get("domaene"),
                        "workflow": wf,
                        "score_strukturell": result.get("score_strukturell"),
                        "score_semantisch": result.get("score_semantisch"),
                        "score_gesamt": result.get("score_gesamt"),
                        "syntax_gate": detail_s.get("syntax_gate"),
                        "fk_gate": detail_s.get("fk_gate"),
                        "pk_rate": detail_s.get("pk_rate"),
                        "fk_referenzen_ungueltig": detail_s.get(
                            "fk_referenzen_ungueltig"
                        ),
                        "entitaet_f1": entitaet.get("f1"),
                        "attribut_f1": attribut.get("f1"),
                        "beziehung_recall": beziehung.get("recall"),
                        "abweichungen_total": len(abweichungen),
                        "abweichungen": _format_abweichungen(abweichungen),
                        "fehlerklasse": "",
                        "begruendung": "",
                    }
                )
                n_rows += 1
    return n_rows


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for candidate in (
        Path.cwd() / ".env",
        script_dir / ".env",
        script_dir.parent / ".env",
    ):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)
    _setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testfall-ids",
        nargs="+",
        default=None,
        help=(
            "Optionale, explizite Liste der Testfall-IDs. "
            "Wenn weggelassen: automatische stratifizierte Auswahl pro Stufe "
            "anhand des durchschnittlichen Gesamt-Scores über alle "
            "vorhandenen Workflows (schlechteste zuerst)."
        ),
    )
    parser.add_argument(
        "--testcases",
        type=Path,
        default=Path("testcases.json"),
        help="Pfad zur testcases.json (für Anforderungstext + Stufe + Domäne).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Verzeichnis mit den Evaluations-Outputs (results/<workflow>/<id>.json).",
    )
    parser.add_argument(
        "--workflows",
        nargs="*",
        default=None,
        help="Workflows die berücksichtigt werden. Default: alle in results-dir gefundenen.",
    )
    parser.add_argument(
        "--per-stufe",
        nargs=3,
        type=int,
        default=[3, 3, 4],
        metavar=("STUFE_1", "STUFE_2", "STUFE_3"),
        help=(
            "Stichprobengrößen für Stufe 1, 2, 3 im Auto-Select-Modus. "
            "Default: 3 3 4 (= 10 Testfälle gesamt). "
            "Wird ignoriert, wenn --testfall-ids gesetzt ist."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bestehende manual_review.csv und manual_review_selection.csv "
            "überschreiben. Default: Abbruch, wenn die Dateien existieren — "
            "schützt manuell eingetragene Klassifikationen."
        ),
    )
    args = parser.parse_args()

    testcases = _load_testcases_index(args.testcases)
    workflows = args.workflows or _discover_workflows(args.results_dir)
    if not workflows:
        logging.error("Keine Workflows in %s gefunden.", args.results_dir)
        return

    # Auswahl bestimmen
    if args.testfall_ids:
        testfall_ids: list[str] = list(args.testfall_ids)
        # avg nachträglich nur für Logging berechnen
        scores = _collect_scores_per_testfall(args.results_dir, testcases, workflows)
        avg_per_tc = {
            tc_id: (sum(s) / len(s)) if s else 0.0 for tc_id, s in scores.items()
        }
        logging.info(
            "Manuelle Auswahl: %d Testfälle, %d Workflows.",
            len(testfall_ids),
            len(workflows),
        )
    else:
        per_stufe = {
            1: args.per_stufe[0],
            2: args.per_stufe[1],
            3: args.per_stufe[2],
        }
        testfall_ids, avg_per_tc = _auto_select_testfaelle(
            results_dir=args.results_dir,
            testcases_index=testcases,
            workflows=workflows,
            per_stufe=per_stufe,
        )
        if not testfall_ids:
            logging.error(
                "Auto-Selektion fand keine Testfälle in %s "
                "(haben die Eval-Files das neue Schema mit `score_gesamt`?).",
                args.results_dir,
            )
            return
        logging.info(
            "Auto-Selektion (stratifiziert: Stufe 1=%d, 2=%d, 3=%d) — "
            "schlechteste durchschnittliche Gesamt-Scores zuerst:",
            per_stufe[1],
            per_stufe[2],
            per_stufe[3],
        )
        for tc_id in testfall_ids:
            stufe = testcases.get(tc_id, {}).get("stufe")
            domaene = testcases.get(tc_id, {}).get("domaene", "")
            logging.info(
                "  Stufe %s | %s | %s | avg_score_gesamt=%.3f",
                stufe,
                tc_id,
                domaene,
                avg_per_tc.get(tc_id, 0.0),
            )

    # Output-Pfade
    selection_path = args.results_dir / "manual_review_selection.csv"
    review_path = args.results_dir / "manual_review.csv"
    review_json_path = args.results_dir / "manual_review.json"

    if not args.force and (
        selection_path.exists() or review_path.exists() or review_json_path.exists()
    ):
        logging.error(
            "Output existiert bereits (%s, %s oder %s). "
            "Nutze --force zum Überschreiben — Vorsicht: manuell eingetragene "
            "Klassifikationen gehen verloren!",
            selection_path,
            review_path,
            review_json_path,
        )
        return

    # CSVs + JSON schreiben
    _write_selection_overview(selection_path, testfall_ids, testcases, avg_per_tc)
    n_review_rows = _write_manual_review(
        review_path,
        testfall_ids,
        testcases,
        args.results_dir,
        workflows,
    )
    n_json_testfaelle = _write_manual_review_json(
        review_json_path,
        testfall_ids,
        testcases,
        avg_per_tc,
        args.results_dir,
        workflows,
    )

    logging.info("")
    logging.info("Selection-Übersicht: %s (%d Testfälle)", selection_path, len(testfall_ids))
    logging.info(
        "Manual-Review-CSV:   %s (%d Zeilen = Testfälle × Workflows)",
        review_path,
        n_review_rows,
    )
    logging.info(
        "Manual-Review-JSON:  %s (%d Testfälle mit Anforderungstext, "
        "Referenzschema und vollständigen Workflow-Resultaten)",
        review_json_path,
        n_json_testfaelle,
    )
    logging.info(
        "Spalten `fehlerklasse` und `begruendung` sind leer — bitte manuell "
        "befüllen mit einer der drei Klassen: "
        "SEMANTISCHE_UNVOLLSTAENDIGKEIT, NORMALISIERUNGSPROBLEM, STRUKTURELLES_PROBLEM."
    )


if __name__ == "__main__":
    main()
