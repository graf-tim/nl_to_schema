"""Hauptskript: führt alle 5 Workflows auf allen Testfällen aus und evaluiert sie.

Aufruf:
    python run_evaluation.py --testcases testcases.json --output results/

Bewertet zwei Dimensionen:
  - strukturell (40 %): Score = syntax_gate · fk_gate · pk_rate
  - semantisch  (60 %): 0.4·entitaet_f1 + 0.4·attribut_f1 + 0.2·beziehung_recall

Pro (testfall, workflow) wird zusätzlich eine Liste "abweichungen" persistiert:
nicht-gematchte Referenz-Elemente (Tabellen/Attribute/Beziehungen) mit
`fehlerklasse: null`.
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
from evaluation.semantic import semantic_score
from evaluation.structural import structural_evaluation
from models.schema import LogicalSchema
from workflows.base import WorkflowState
from workflows.registry import WORKFLOWS, build


WEIGHTS = {"strukturell": 0.40, "semantisch": 0.60}


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


def _empty_structural() -> dict:
    return {
        "score": 0.0,
        "syntax_gate": 0,
        "fk_gate": 0,
        "pk_rate": 0.0,
        "pk_tabellen_gesamt": 0,
        "pk_tabellen_mit_pk": 0,
        "fk_referenzen_gesamt": 0,
        "fk_referenzen_ungueltig": 0,
    }


def _empty_semantic() -> dict:
    leer = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "matched": 0,
        "nur_in_referenz": 0,
        "nur_in_generiert": 0,
    }
    return {
        "score": 0.0,
        "entitaet": dict(leer),
        "attribut": dict(leer),
        "beziehung": {
            "recall": 0.0,
            "matched": 0,
            "nur_in_referenz": 0,
            "nur_in_generiert": 0,
        },
        "abweichungen": [],
    }


def _evaluate(
    state: WorkflowState,
    anforderungstext: str,
    reference: LogicalSchema,
) -> dict[str, Any]:
    """Berechnet die zwei Score-Dimensionen plus gewichteten Gesamtscore.

    Output-Format folgt der Spec aus dem Plan:
      score_strukturell, score_semantisch, score_gesamt,
      detail_strukturell, detail_semantisch, abweichungen
    """
    if state.logical_schema is None:
        empty_struct = _empty_structural()
        empty_sem = _empty_semantic()
        return {
            "score_strukturell": 0.0,
            "score_semantisch": 0.0,
            "score_gesamt": 0.0,
            "detail_strukturell": {k: v for k, v in empty_struct.items() if k != "score"},
            "detail_semantisch": {k: v for k, v in empty_sem.items() if k not in ("score", "abweichungen")},
            "abweichungen": [],
            "note": "Workflow lieferte kein logical_schema.",
        }

    structural = structural_evaluation(state.final_ddl or "", state.logical_schema)
    semantic = semantic_score(state.logical_schema, reference)

    score_strukturell = structural["score"]
    score_semantisch = semantic["score"]
    score_gesamt = (
        WEIGHTS["strukturell"] * score_strukturell
        + WEIGHTS["semantisch"] * score_semantisch
    )

    detail_strukturell = {k: v for k, v in structural.items() if k != "score"}
    detail_semantisch = {
        "entitaet": semantic["entitaet"],
        "attribut": semantic["attribut"],
        "beziehung": semantic["beziehung"],
    }

    return {
        "score_strukturell": score_strukturell,
        "score_semantisch": score_semantisch,
        "score_gesamt": score_gesamt,
        "detail_strukturell": detail_strukturell,
        "detail_semantisch": detail_semantisch,
        "abweichungen": semantic["abweichungen"],
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
        **evaluation,
        "state": _serialize_state(state),
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
    detail_s = evaluation.get("detail_strukturell") or {}
    detail_sem = evaluation.get("detail_semantisch") or {}
    entitaet = detail_sem.get("entitaet") or {}
    attribut = detail_sem.get("attribut") or {}
    beziehung = detail_sem.get("beziehung") or {}
    return {
        "testfall_id": tc.get("id"),
        "stufe": tc.get("stufe"),
        "workflow": workflow,
        "score_strukturell": evaluation.get("score_strukturell"),
        "score_semantisch": evaluation.get("score_semantisch"),
        "score_gesamt": evaluation.get("score_gesamt"),
        "syntax_gate": detail_s.get("syntax_gate"),
        "fk_gate": detail_s.get("fk_gate"),
        "pk_rate": detail_s.get("pk_rate"),
        "pk_tabellen_gesamt": detail_s.get("pk_tabellen_gesamt"),
        "pk_tabellen_mit_pk": detail_s.get("pk_tabellen_mit_pk"),
        "fk_referenzen_gesamt": detail_s.get("fk_referenzen_gesamt"),
        "fk_referenzen_ungueltig": detail_s.get("fk_referenzen_ungueltig"),
        "entitaet_precision": entitaet.get("precision"),
        "entitaet_recall": entitaet.get("recall"),
        "entitaet_f1": entitaet.get("f1"),
        "entitaet_matched": entitaet.get("matched"),
        "entitaet_nur_in_referenz": entitaet.get("nur_in_referenz"),
        "entitaet_nur_in_generiert": entitaet.get("nur_in_generiert"),
        "attribut_precision": attribut.get("precision"),
        "attribut_recall": attribut.get("recall"),
        "attribut_f1": attribut.get("f1"),
        "attribut_matched": attribut.get("matched"),
        "attribut_nur_in_referenz": attribut.get("nur_in_referenz"),
        "attribut_nur_in_generiert": attribut.get("nur_in_generiert"),
        "beziehung_recall": beziehung.get("recall"),
        "beziehung_matched": beziehung.get("matched"),
        "beziehung_nur_in_referenz": beziehung.get("nur_in_referenz"),
        "beziehung_nur_in_generiert": beziehung.get("nur_in_generiert"),
        "abweichungen_total": len(evaluation.get("abweichungen") or []),
        "input_tokens": telemetry.get("total_input_tokens", 0),
        "output_tokens": telemetry.get("total_output_tokens", 0),
        "total_tokens": telemetry.get("total_tokens", 0),
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
    # leere Shell-Vars überschrieben werden.
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
                if (
                    existing is not None
                    and "score_gesamt" in existing  # neues Schema, sonst neu rechnen
                ):
                    evaluation = {
                        k: existing.get(k)
                        for k in (
                            "score_strukturell",
                            "score_semantisch",
                            "score_gesamt",
                            "detail_strukturell",
                            "detail_semantisch",
                            "abweichungen",
                        )
                    }
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
            evaluation = _evaluate(state, anforderungstext, reference)
            telemetry = LEDGER.summary()
            logging.info(
                "[%s][%s] score_gesamt=%.3f tokens in=%d out=%d cost=$%.5f calls=%d",
                wf,
                tc_id,
                evaluation.get("score_gesamt") or 0.0,
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
