"""LLM-Aufruf-Helfer mit strukturiertem Output, Retry, Logging und Token-Tracking."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from agents._telemetry import LEDGER
from workflows.base import get_llm


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

LLMFactory = Callable[[], object]


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _extract_usage(raw_message) -> tuple[int, int]:
    """Holt (input_tokens, output_tokens) aus einer LangChain-AIMessage.

    Fällt auf 0/0 zurück, wenn keine usage_metadata verfügbar.
    """
    usage = getattr(raw_message, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    # Fallback: response_metadata.token_usage (älteres Format)
    meta = getattr(raw_message, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
    )


def call_structured(
    *,
    workflow_name: str,
    iteration: int,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    output_model: Type[T],
    max_retries: int = 2,
    llm_factory: Optional[LLMFactory] = None,
) -> T:
    """Ruft das LLM mit strukturiertem Output auf, mit Retry, Logging und Tokens.

    Funktioniert provider-agnostisch (ChatOpenAI, ChatAnthropic, ...). Nutzt
    `with_structured_output(..., include_raw=True)`, damit die rohe AIMessage
    mit `usage_metadata` erhalten bleibt und Token-Counts in den TokenLedger
    geschrieben werden können.

    Args:
        llm_factory: Optionaler LLM-Konstruktor. Default: workflow-LLM
            (Claude Opus 4.6). Für den qualitativen Judge wird hier
            workflows.base.get_judge_llm übergeben.
    """
    factory = llm_factory or get_llm
    base_llm = factory()
    structured = base_llm.with_structured_output(output_model, include_raw=True)
    model_name = (
        getattr(base_llm, "model", None)
        or getattr(base_llm, "model_name", None)
        or "unknown-model"
    )
    approx_in = _approx_tokens(system_prompt) + _approx_tokens(user_message)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = structured.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )
            # include_raw=True liefert {"raw": AIMessage, "parsed": Pydantic, "parsing_error": ...}
            if isinstance(response, dict):
                parsed = response.get("parsed")
                raw = response.get("raw")
                parsing_error = response.get("parsing_error")
                if parsing_error is not None and parsed is None:
                    raise parsing_error
            else:
                # Fallback (sollte mit include_raw=True nicht auftreten)
                parsed = response
                raw = None

            if not isinstance(parsed, output_model):
                raise ValueError(
                    f"Erwartet {output_model.__name__}, erhalten {type(parsed).__name__}"
                )

            in_tokens, out_tokens = (0, 0)
            if raw is not None:
                in_tokens, out_tokens = _extract_usage(raw)
            if in_tokens == 0 and out_tokens == 0:
                # echte Werte fehlen → nutze grobe Schätzung, damit der Run nicht 0 wird
                in_tokens = approx_in
                out_tokens = _approx_tokens(parsed.model_dump_json())

            LEDGER.record(
                workflow_name=workflow_name,
                iteration=iteration,
                agent_name=agent_name,
                model=model_name,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            logger.info(
                "[%s][iter=%d][%s] in_tokens=%d out_tokens=%d attempt=%d",
                workflow_name,
                iteration,
                agent_name,
                in_tokens,
                out_tokens,
                attempt,
            )
            return parsed
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[%s][iter=%d][%s] Versuch %d/%d fehlgeschlagen: %s",
                workflow_name,
                iteration,
                agent_name,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    assert last_exc is not None
    raise last_exc
