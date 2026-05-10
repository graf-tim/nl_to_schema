"""Hauptskript: führt alle 5 Workflows auf allen Testfällen aus und evaluiert sie.

Aufruf:
    python run_evaluation.py --testcases testcases.json --output results/
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
from evaluation.qualitative import qualitative_score
from evaluation.semantic import semantic_score
from evaluation.structural import structural_details
from models.schema import LogicalSchema
from workflows.base import WorkflowState
from workflows.registry import WORKFLOWS, build


WEIGHTS = {"strukturell": 0.30, "semantisch": 0.40, "qualitativ": 0.30}


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _load_testcases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_workflow(name: str, anforderungstext: str) -> WorkflowState:
    graph = build(name)
    initial = WorkflowState(
        anforderungstext=anforderungstext,
        workflow_name=name,
    )
    raw = graph.invoke(initial)
    if isinstance(raw, WorkflowState):
        return raw
    return WorkflowState.model_validate(raw)


def _evaluate(
    state: WorkflowState,
    anforderungstext: str,
    reference: LogicalSchema,
    *,
    skip_qualitative: bool = False,
) -> dict[str, Any]:
    """Berechnet die drei Score-Dimensionen plus gewichteten Gesamtscore.

    Mit `skip_qualitative=True` wird der LLM-as-Judge-Aufruf weggelassen.
    `weighted_total` wird dann mit re-normalisierten Gewichten aus den beiden
    übrigen Dimensionen gebildet, damit der Score weiterhin in [0,1] und
    intern vergleichbar bleibt.
    """
    if state.logical_schema is None:
        return {
            "structural": {"score": 0.0, "syntaktisch_korrekt": False},
            "semantic": {"semantic_score": 0.0},
            "qualitative": {"qualitative_score": None, "skipped": skip_qualitative},
            "weighted_total": 0.0,
            "note": "Workflow lieferte kein logical_schema.",
        }
    structural = structural_details(state.final_ddl or "", state.logical_schema)
    semantic = semantic_score(state.logical_schema, reference)

    if skip_qualitative:
        qualitative = {"qualitative_score": None, "skipped": True}
        # Re-normalisiert auf die zwei aktiven Dimensionen.
        denom = WEIGHTS["strukturell"] + WEIGHTS["semantisch"]
        total = (
            (WEIGHTS["strukturell"] / denom) * structural["score"]
            + (WEIGHTS["semantisch"] / denom) * semantic["semantic_score"]
        )
    else:
        qualitative = qualitative_score(state.logical_schema, anforderungstext)
        total = (
            WEIGHTS["strukturell"] * structural["score"]
            + WEIGHTS["semantisch"] * semantic["semantic_score"]
            + WEIGHTS["qualitativ"] * qualitative["qualitative_score"]
        )
    return {
        "structural": structural,
        "semantic": semantic,
        "qualitative": qualitative,
        "weighted_total": total,
    }


def _serialize_state(state: WorkflowState) -> dict:
    return state.model_dump(mode="json")


def _result_path(out_dir: Path, workflow: str, testfall_id: str) -> Path:
    return out_dir / workflow / f"{testfall_id}.json"


def _save_result(
    out_dir: Path,
    workflow: str,
    testfall_id: str,
    state: WorkflowState,
    evaluation: dict,
    telemetry: dict,
) -> None:
    path = _result_path(out_dir, workflow, testfall_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "testfall_id": testfall_id,
        "workflow": workflow,
        "state": _serialize_state(state),
        "evaluation": evaluation,
        "telemetry": telemetry,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_existing_result(
    out_dir: Path, workflow: str, testfall_id: str
) -> dict | None:
    """Liest ein bereits gespeichertes Ergebnis, oder None falls nicht/ defekt."""
    path = _result_path(out_dir, workflow, testfall_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(
            "Bestehende Datei konnte nicht gelesen werden, wird neu gerechnet: %s (%s)",
            path,
            exc,
        )
        return None


def _summary_row(
    *,
    tc: dict,
    workflow: str,
    evaluation: dict,
    telemetry: dict,
    error: str,
) -> dict[str, Any]:
    """Baut eine Zeile für summary.csv aus evaluation+telemetry-Dicts."""
    structural = evaluation.get("structural") or {}
    semantic = evaluation.get("semantic") or {}
    qualitative = evaluation.get("qualitative") or {}
    return {
        "testfall_id": tc.get("id"),
        "stufe": tc.get("stufe"),
        "workflow": workflow,
        "structural_score": structural.get("score"),
        "syntaktisch_korrekt": structural.get("syntaktisch_korrekt"),
        "semantic_score": semantic.get("semantic_score"),
        "qualitative_score": qualitative.get("qualitative_score"),
        "weighted_total": evaluation.get("weighted_total"),
        "input_tokens": telemetry.get("total_input_tokens", 0),
        "output_tokens": telemetry.get("total_output_tokens", 0),
        "total_tokens": telemetry.get("total_tokens", 0),
        "cost_usd": round(telemetry.get("total_cost_usd", 0.0) or 0.0, 6),
        "llm_calls": telemetry.get("total_calls", 0),
        "error": error or "",
    }


def _select_testcases(
    testcases: list[dict],
    *,
    ids: list[str] | None,
    limit_per_stufe: int | None,
) -> list[dict]:
    """Wählt Testfälle aus.

    1. Wenn `ids` gesetzt: nur diese IDs (in Eingangsreihenfolge der Datei).
    2. Sonst, wenn `limit_per_stufe` gesetzt: pro Stufe die ersten N Einträge
       in der Reihenfolge, in der sie in `testcases.json` stehen
       (typischerweise alphabetisch nach ID, da der Generator sortiert speichert).
    3. Sonst: alle Testfälle.
    """
    if ids:
        wanted = set(ids)
        return [t for t in testcases if t["id"] in wanted]
    if limit_per_stufe is not None:
        per_stufe_count: dict[int, int] = {}
        out: list[dict] = []
        for tc in testcases:
            stufe = tc.get("stufe")
            n = per_stufe_count.get(stufe, 0)
            if n < limit_per_stufe:
                out.append(tc)
                per_stufe_count[stufe] = n + 1
        return out
    return list(testcases)


def main() -> None:
    # .env-Suche analog zum Testdata-Generator: lädt nl_to_schema/.env auch dann,
    # wenn das Skript aus einem anderen cwd gestartet wird. override=True, damit
    # leere Shell-Vars (z.B. GOOGLE_API_KEY="") überschrieben werden.
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
    parser.add_argument("--testcases", type=Path, default=Path("testcases.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--workflows",
        nargs="*",
        default=list(WORKFLOWS.keys()),
        help="Liste der auszuführenden Workflows (Default: alle).",
    )
    parser.add_argument(
        "--testfall-ids",
        nargs="*",
        default=None,
        help="Optional: nur diese Testfall-IDs ausführen.",
    )
    parser.add_argument(
        "--limit-per-stufe",
        type=int,
        default=None,
        help=(
            "Pro Stufe nur die ersten N Testfälle ausführen "
            "(z.B. --limit-per-stufe 15)."
        ),
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help=(
            "LLM-as-Judge (qualitative Evaluation) überspringen. "
            "weighted_total wird dann aus structural+semantic re-normalisiert."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bestehende results/<workflow>/<id>.json überschreiben. "
            "Default: vorhandene Ergebnisse werden übersprungen und ihre "
            "Werte in die summary.csv übernommen (Idempotenz)."
        ),
    )
    args = parser.parse_args()

    all_testcases = _load_testcases(args.testcases)
    testcases = _select_testcases(
        all_testcases,
        ids=args.testfall_ids,
        limit_per_stufe=args.limit_per_stufe,
    )
    if not testcases:
        logging.error("Keine Testfälle ausgewählt — Abbruch.")
        return
    logging.info(
        "Ausgewählt: %d von %d Testfällen", len(testcases), len(all_testcases)
    )
    if args.skip_qualitative:
        logging.warning(
            "LLM-as-Judge wird übersprungen (--skip-qualitative). "
            "weighted_total = (3/7)·structural + (4/7)·semantic."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for tc in testcases:
        tc_id = tc["id"]
        anforderungstext = tc["anforderungstext"]
        reference = LogicalSchema.model_validate(tc["referenzschema"])
        for wf in args.workflows:
            # Idempotenz: bestehendes Ergebnis nicht erneut rechnen.
            if not args.force:
                existing = _load_existing_result(args.output, wf, tc_id)
                if existing is not None:
                    evaluation = existing.get("evaluation") or {}
                    telemetry = existing.get("telemetry") or {}
                    state_data = existing.get("state") or {}
                    error = state_data.get("error") or ""
                    logging.info(
                        "[%s][%s] SKIP — bereits in %s",
                        wf,
                        tc_id,
                        _result_path(args.output, wf, tc_id),
                    )
                    summary_rows.append(
                        _summary_row(
                            tc=tc,
                            workflow=wf,
                            evaluation=evaluation,
                            telemetry=telemetry,
                            error=error,
                        )
                    )
                    continue

            logging.info("Starte Workflow=%s Testfall=%s", wf, tc_id)
            LEDGER.reset()
            try:
                state = _run_workflow(wf, anforderungstext)
            except Exception as exc:
                logging.exception("Workflow %s/%s abgebrochen.", wf, tc_id)
                state = WorkflowState(
                    anforderungstext=anforderungstext,
                    workflow_name=wf,
                    error=f"workflow_crashed: {exc}",
                )
            evaluation = _evaluate(
                state,
                anforderungstext,
                reference,
                skip_qualitative=args.skip_qualitative,
            )
            telemetry = LEDGER.summary()
            logging.info(
                "[%s][%s] tokens in=%d out=%d cost=$%.5f calls=%d",
                wf,
                tc_id,
                telemetry["total_input_tokens"],
                telemetry["total_output_tokens"],
                telemetry["total_cost_usd"],
                telemetry["total_calls"],
            )
            _save_result(args.output, wf, tc_id, state, evaluation, telemetry)
            summary_rows.append(
                _summary_row(
                    tc=tc,
                    workflow=wf,
                    evaluation=evaluation,
                    telemetry=telemetry,
                    error=state.error or "",
                )
            )

    csv_path = args.output / "summary.csv"
    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logging.info("Zusammenfassung geschrieben: %s", csv_path)


if __name__ == "__main__":
    main()
