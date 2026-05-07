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
    state: WorkflowState, anforderungstext: str, reference: LogicalSchema
) -> dict[str, Any]:
    if state.logical_schema is None:
        return {
            "structural": {"score": 0.0, "syntaktisch_korrekt": False},
            "semantic": {"semantic_score": 0.0},
            "qualitative": {"qualitative_score": 0.0},
            "weighted_total": 0.0,
            "note": "Workflow lieferte kein logical_schema.",
        }
    structural = structural_details(state.final_ddl or "", state.logical_schema)
    semantic = semantic_score(state.logical_schema, reference)
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


def _save_result(
    out_dir: Path,
    workflow: str,
    testfall_id: str,
    state: WorkflowState,
    evaluation: dict,
    telemetry: dict,
) -> None:
    sub = out_dir / workflow
    sub.mkdir(parents=True, exist_ok=True)
    payload = {
        "testfall_id": testfall_id,
        "workflow": workflow,
        "state": _serialize_state(state),
        "evaluation": evaluation,
        "telemetry": telemetry,
    }
    with (sub / f"{testfall_id}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    load_dotenv()
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
    args = parser.parse_args()

    testcases = _load_testcases(args.testcases)
    if args.testfall_ids:
        testcases = [t for t in testcases if t["id"] in args.testfall_ids]

    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for tc in testcases:
        tc_id = tc["id"]
        anforderungstext = tc["anforderungstext"]
        reference = LogicalSchema.model_validate(tc["referenzschema"])
        for wf in args.workflows:
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
            evaluation = _evaluate(state, anforderungstext, reference)
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
                {
                    "testfall_id": tc_id,
                    "stufe": tc.get("stufe"),
                    "workflow": wf,
                    "structural_score": evaluation["structural"].get("score"),
                    "syntaktisch_korrekt": evaluation["structural"].get(
                        "syntaktisch_korrekt"
                    ),
                    "semantic_score": evaluation["semantic"].get("semantic_score"),
                    "qualitative_score": evaluation["qualitative"].get(
                        "qualitative_score"
                    ),
                    "weighted_total": evaluation["weighted_total"],
                    "input_tokens": telemetry["total_input_tokens"],
                    "output_tokens": telemetry["total_output_tokens"],
                    "total_tokens": telemetry["total_tokens"],
                    "cost_usd": round(telemetry["total_cost_usd"], 6),
                    "llm_calls": telemetry["total_calls"],
                    "error": state.error or "",
                }
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
