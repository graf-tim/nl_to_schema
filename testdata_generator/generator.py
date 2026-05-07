"""Hauptgenerierungsskript: erzeugt Testfälle via OpenAI (LangChain).

Strukturierter Output wird durch LangChain's `with_structured_output(Testfall,
include_raw=True)` erzwungen — analog zu den Workflows in nl_to_schema/.

Aufruf:
    python generator.py                       # alle 100
    python generator.py --limit-per-stufe 5   # nur die ersten 5 pro Stufe (Smoke-Test)
    python generator.py --ids TC001 TC034     # gezielt einzelne IDs

Voraussetzungen:
- OPENAI_API_KEY in .env (oder Umgebungsvariable)
- Pakete aus requirements.txt installiert

Modellwahl: GPT-5.4 Pro (Default). Über die Env-Variable `TESTDATA_GEN_MODEL`
kann ein anderer OpenAI-Modellname gesetzt werden, ohne den Code zu ändern.

Bereits vorhandene output/raw/<id>.json werden ohne API-Aufruf übersprungen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from domains import DOMAIN_ASSIGNMENTS
from models import Testfall
from prompts import SYSTEM_PROMPT, render_user_prompt


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.4"
TEMPERATURE = 1.0
MAX_TOKENS = 8000
PROMPT_VERSION = "v1.2-openai"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 60

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "output" / "raw"
TESTCASES_FILE = ROOT / "output" / "testcases.json"
LOG_FILE = ROOT / "generation_log.json"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_testfall(testfall: Testfall) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{testfall.id}.json"
    payload = testfall.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _load_existing_testfall(tc_id: str) -> Optional[Testfall]:
    path = RAW_DIR / f"{tc_id}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Testfall.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _model_name() -> str:
    """Aktueller Modellname. Per Env-Variable überschreibbar."""
    return os.getenv("TESTDATA_GEN_MODEL", DEFAULT_MODEL)


def _get_structured_llm():
    """Erzeugt einen ChatOpenAI mit erzwungenem strukturiertem Output."""
    from langchain_openai import ChatOpenAI

    base = ChatOpenAI(
        model=_model_name(),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,  # eigene Retry-Logik im Generator
    )
    return base.with_structured_output(Testfall, include_raw=True)


def _extract_usage(raw_message) -> tuple[int, int]:
    usage = getattr(raw_message, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    return (0, 0)


# ---------------------------------------------------------------------------
# Generierungslogik
# ---------------------------------------------------------------------------

def _generate_testfall(
    structured_llm,
    *,
    tc_id: str,
    stufe: int,
    domaene: str,
) -> tuple[Optional[Testfall], int, Optional[str], int, int]:
    """Versucht bis MAX_RETRIES, einen validen Testfall zu erzeugen.

    Rückgabe: (testfall_or_none, anzahl_versuche, fehler_or_none, in_tokens, out_tokens)
    """
    user_prompt = render_user_prompt(stufe=stufe, id=tc_id, domaene=domaene)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(
            f"    Versuch {attempt}/{MAX_RETRIES} (Timeout {REQUEST_TIMEOUT_SECONDS}s) ...",
            flush=True,
        )
        try:
            response = structured_llm.invoke(messages)
        except Exception as exc:
            last_error = f"api_error: {exc}"
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(
                    f"    API-Fehler in Versuch {attempt}/{MAX_RETRIES}: {exc}. "
                    f"Warte {wait}s.",
                    flush=True,
                )
                time.sleep(wait)
                continue
            break

        parsed = response.get("parsed") if isinstance(response, dict) else None
        raw = response.get("raw") if isinstance(response, dict) else None
        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None

        if parsing_error is not None and parsed is None:
            last_error = f"parse_error: {parsing_error}"
            print(
                f"    Parse-Fehler in Versuch {attempt}/{MAX_RETRIES}: {parsing_error}",
                flush=True,
            )
            continue

        if not isinstance(parsed, Testfall):
            last_error = f"parse_error: erwartet Testfall, erhalten {type(parsed)}"
            print(f"    {last_error}", flush=True)
            continue

        in_tokens, out_tokens = _extract_usage(raw) if raw is not None else (0, 0)

        # Sicherstellen, dass id/stufe/domaene dem Soll entsprechen.
        if parsed.id != tc_id or parsed.stufe != stufe or parsed.domaene != domaene:
            parsed = parsed.model_copy(
                update={"id": tc_id, "stufe": stufe, "domaene": domaene}
            )
        parsed = parsed.model_copy(
            update={
                "generiert_mit": _model_name(),
                "generiert_am": _now_iso(),
                "prompt_version": PROMPT_VERSION,
            }
        )
        return parsed, attempt, None, in_tokens, out_tokens

    return None, MAX_RETRIES, last_error, 0, 0


# ---------------------------------------------------------------------------
# Konsolidierung & Logging
# ---------------------------------------------------------------------------

def _consolidate_testcases() -> int:
    items: list[dict] = []
    for path in sorted(RAW_DIR.glob("TC*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                items.append(json.load(f))
        except json.JSONDecodeError:
            continue
    items.sort(key=lambda x: x.get("id", ""))
    TESTCASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TESTCASES_FILE.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return len(items)


def _write_log(eintraege: list[dict]) -> None:
    erfolgreich = sum(1 for e in eintraege if e["status"] == "ok")
    fehlgeschlagen = sum(1 for e in eintraege if e["status"] == "failed")
    pro_stufe: dict[str, int] = {"1": 0, "2": 0, "3": 0}
    for e in eintraege:
        if e["status"] == "ok":
            pro_stufe[str(e["stufe"])] += 1

    total_in = sum(e.get("input_tokens", 0) for e in eintraege)
    total_out = sum(e.get("output_tokens", 0) for e in eintraege)

    log = {
        "generiert_am": _now_iso(),
        "modell": _model_name(),
        "temperatur": TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "eintraege": eintraege,
        "zusammenfassung": {
            "gesamt": len(eintraege),
            "erfolgreich": erfolgreich,
            "fehlgeschlagen": fehlgeschlagen,
            "pro_stufe": pro_stufe,
            "input_tokens_total": total_in,
            "output_tokens_total": total_out,
        },
    }
    with LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Auswahl
# ---------------------------------------------------------------------------

def _select_assignments(
    *, limit_per_stufe: Optional[int], ids: Optional[list[str]]
) -> list[dict]:
    if ids:
        wanted = set(ids)
        return [a for a in DOMAIN_ASSIGNMENTS if a["id"] in wanted]
    if limit_per_stufe is not None:
        per_stufe_count: dict[int, int] = {}
        out: list[dict] = []
        for a in DOMAIN_ASSIGNMENTS:
            n = per_stufe_count.get(a["stufe"], 0)
            if n < limit_per_stufe:
                out.append(a)
                per_stufe_count[a["stufe"]] = n + 1
        return out
    return list(DOMAIN_ASSIGNMENTS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-per-stufe",
        type=int,
        default=None,
        help="Nur die ersten N Testfälle pro Stufe generieren (für Smoke-Tests).",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional: nur diese IDs generieren (z.B. TC001 TC034 TC067).",
    )
    args = parser.parse_args()

    # .env-Suche an mehreren plausiblen Orten. override=True, damit Werte aus
    # der .env auch leere Shell-Vars wie ANTHROPIC_API_KEY="" überschreiben.
    for candidate in (
        ROOT / ".env",
        ROOT.parent / ".env",
        ROOT.parent / "nl_to_schema" / ".env",
    ):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)

    try:
        structured_llm = _get_structured_llm()
    except Exception as exc:
        print(f"Konnte ChatAnthropic nicht initialisieren: {exc}", file=sys.stderr)
        return 2

    selected = _select_assignments(
        limit_per_stufe=args.limit_per_stufe, ids=args.ids
    )
    if not selected:
        print("Keine Testfälle zur Generierung ausgewählt.", file=sys.stderr)
        return 2

    eintraege: list[dict] = []
    total = len(selected)

    for idx, assignment in enumerate(selected, start=1):
        tc_id = assignment["id"]
        stufe = assignment["stufe"]
        domaene = assignment["domaene"]
        prefix = f"[{idx:03d}/{total}] {tc_id} | Stufe {stufe} | {domaene}"

        existing = _load_existing_testfall(tc_id)
        if existing is not None:
            print(f"{prefix} ... SKIP (bereits vorhanden)", flush=True)
            eintraege.append(
                {
                    "id": tc_id,
                    "stufe": stufe,
                    "domaene": domaene,
                    "status": "ok",
                    "versuche": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "timestamp": _now_iso(),
                    "skipped": True,
                }
            )
            continue

        print(f"{prefix}", flush=True)
        testfall, attempts, error, in_tokens, out_tokens = _generate_testfall(
            structured_llm, tc_id=tc_id, stufe=stufe, domaene=domaene
        )

        if testfall is None:
            print(f"FAILED ({error})", flush=True)
            eintraege.append(
                {
                    "id": tc_id,
                    "stufe": stufe,
                    "domaene": domaene,
                    "status": "failed",
                    "versuche": attempts,
                    "fehler": error or "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "timestamp": _now_iso(),
                }
            )
            continue

        _save_testfall(testfall)
        print(
            f"OK (Versuch {attempts}/{MAX_RETRIES}, "
            f"in={in_tokens} out={out_tokens})",
            flush=True,
        )
        eintraege.append(
            {
                "id": tc_id,
                "stufe": stufe,
                "domaene": domaene,
                "status": "ok",
                "versuche": attempts,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "timestamp": _now_iso(),
            }
        )

    consolidated = _consolidate_testcases()
    _write_log(eintraege)

    erfolgreich = sum(1 for e in eintraege if e["status"] == "ok")
    pro_stufe = {1: 0, 2: 0, 3: 0}
    for e in eintraege:
        if e["status"] == "ok":
            pro_stufe[e["stufe"]] += 1
    total_in = sum(e.get("input_tokens", 0) for e in eintraege)
    total_out = sum(e.get("output_tokens", 0) for e in eintraege)

    print()
    print("Zusammenfassung:")
    print(f"  Gesamt:        {len(eintraege)}")
    print(f"  Erfolgreich:   {erfolgreich}")
    print(f"  Fehlgeschlagen:{len(eintraege) - erfolgreich}")
    print(
        "  Pro Stufe:     "
        f"Stufe 1 = {pro_stufe[1]}, Stufe 2 = {pro_stufe[2]}, Stufe 3 = {pro_stufe[3]}"
    )
    print(f"  Tokens:        in={total_in}, out={total_out}, total={total_in+total_out}")
    print(f"  Konsolidiert in: {TESTCASES_FILE} ({consolidated} Einträge)")
    print(f"  Log:             {LOG_FILE}")
    return 0 if erfolgreich == len(eintraege) else 1


if __name__ == "__main__":
    sys.exit(main())
