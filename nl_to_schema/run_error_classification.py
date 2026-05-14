"""Fehlerklassifikation für eine manuell ausgewählte Stichprobe.

Lädt persistierte Evaluations-Outputs aus `results/<wf>/<id>.json`, ruft für
jede `abweichungen[i]` mit `fehlerklasse=None` das Klassifikations-LLM
(Gemini 2.5 Pro per default), und schreibt die ergänzten Daten zurück.

Aufruf:
    python run_error_classification.py \\
        --testfall-ids TC042 TC067 TC089 \\
        --testcases ../testdata_generator/output/testcases.json \\
        --results-dir results/

Optional:
    --workflows sa ps iw oa mad   # Default: alle vorhandenen Unterordner
    --force                        # auch bereits klassifizierte Abweichungen neu klassifizieren
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

from agents._telemetry import LEDGER
from evaluation.error_classifier import classify_abweichung
from models.schema import LogicalSchema


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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_result(results_dir: Path, workflow: str, testfall_id: str, data: dict) -> Path:
    path = results_dir / workflow / f"{testfall_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _discover_workflows(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    return sorted(
        p.name for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


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
        required=True,
        help="Testfall-IDs für die klassifiziert werden soll (z.B. TC042 TC067).",
    )
    parser.add_argument(
        "--testcases",
        type=Path,
        default=Path("testcases.json"),
        help="Pfad zur testcases.json (für Anforderungstext + Referenzschema).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Verzeichnis mit den Evaluations-Outputs.",
    )
    parser.add_argument(
        "--workflows",
        nargs="*",
        default=None,
        help="Workflows die klassifiziert werden. Default: alle in results-dir gefundenen.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Auch bereits klassifizierte Abweichungen erneut klassifizieren. "
            "Default: skip wenn fehlerklasse != null."
        ),
    )
    args = parser.parse_args()

    testcases = _load_testcases_index(args.testcases)
    workflows = args.workflows or _discover_workflows(args.results_dir)
    if not workflows:
        logging.error("Keine Workflows in %s gefunden.", args.results_dir)
        return

    logging.info(
        "Klassifiziere %d Testfälle × %d Workflows = bis zu %d Files.",
        len(args.testfall_ids),
        len(workflows),
        len(args.testfall_ids) * len(workflows),
    )

    summary_rows: list[dict[str, Any]] = []
    for tc_id in args.testfall_ids:
        tc = testcases.get(tc_id)
        if tc is None:
            logging.warning("Testfall %s nicht in testcases.json — skip.", tc_id)
            continue
        anforderungstext = tc["anforderungstext"]
        reference = LogicalSchema.model_validate(tc["referenzschema"])

        for wf in workflows:
            result = _load_result(args.results_dir, wf, tc_id)
            if result is None:
                logging.warning("Kein Eval-Output für %s/%s — skip.", wf, tc_id)
                continue

            abweichungen = result.get("abweichungen") or []
            if not abweichungen:
                logging.info("[%s][%s] keine Abweichungen — skip.", wf, tc_id)
                continue

            state_data = result.get("state") or {}
            generated_schema = state_data.get("logical_schema")
            if not generated_schema:
                logging.warning(
                    "[%s][%s] state.logical_schema fehlt — skip Klassifikation.",
                    wf,
                    tc_id,
                )
                continue
            generated = LogicalSchema.model_validate(generated_schema)

            LEDGER.reset()
            n_new = 0
            n_skip = 0
            for abw in abweichungen:
                if abw.get("fehlerklasse") is not None and not args.force:
                    n_skip += 1
                    continue
                klass = classify_abweichung(
                    anforderungstext=anforderungstext,
                    reference=reference,
                    generated=generated,
                    abweichung=abw,
                    workflow_name=wf,
                    iteration=0,
                )
                abw["fehlerklasse"] = klass.get("fehlerklasse")
                abw["begruendung"] = klass.get("begruendung")
                if klass.get("error"):
                    abw["fehler"] = klass["error"]
                n_new += 1
                summary_rows.append(
                    {
                        "testfall_id": tc_id,
                        "workflow": wf,
                        "element": abw.get("element"),
                        "typ": abw.get("typ"),
                        "fehlerklasse": abw.get("fehlerklasse") or "",
                        "begruendung": abw.get("begruendung") or "",
                    }
                )

            telemetry = LEDGER.summary()
            # Telemetrie der Klassifikation separat aufzeichnen, ohne die
            # Workflow-Telemetrie zu überschreiben.
            classifier_telemetry = result.get("classifier_telemetry") or {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "calls": [],
            }
            classifier_telemetry["total_calls"] += telemetry.get("total_calls", 0)
            classifier_telemetry["total_input_tokens"] += telemetry.get(
                "total_input_tokens", 0
            )
            classifier_telemetry["total_output_tokens"] += telemetry.get(
                "total_output_tokens", 0
            )
            classifier_telemetry["total_tokens"] += telemetry.get("total_tokens", 0)
            classifier_telemetry["total_cost_usd"] = round(
                (classifier_telemetry["total_cost_usd"] or 0.0)
                + (telemetry.get("total_cost_usd", 0.0) or 0.0),
                6,
            )
            classifier_telemetry["calls"].extend(telemetry.get("calls", []))
            result["classifier_telemetry"] = classifier_telemetry

            _save_result(args.results_dir, wf, tc_id, result)
            logging.info(
                "[%s][%s] klassifiziert=%d übersprungen=%d cost=$%.5f",
                wf,
                tc_id,
                n_new,
                n_skip,
                telemetry.get("total_cost_usd", 0.0),
            )

    if summary_rows:
        csv_path = args.results_dir / "fehlerklassifikation_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logging.info(
            "Zusammenfassung geschrieben: %s (%d Klassifikationen)",
            csv_path,
            len(summary_rows),
        )


if __name__ == "__main__":
    main()
