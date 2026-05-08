"""Token-Ledger: erfasst echte Token-Nutzung pro Workflow-Run.

Der Ledger ist ein Modul-Singleton. Vor jedem Run wird er via reset() geleert,
am Ende der Auswertung via summary() ausgelesen und ins Ergebnis geschrieben.

Preise sind Stand 2025/2026. Nicht gelistete Modelle führen zu cost_usd=0.0.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


# USD pro 1 Token (1M-Preis / 1_000_000). Quellen: OpenAI- und Anthropic-Pricing.
PRICING_PER_TOKEN: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini":      {"input":  0.15 / 1_000_000, "output":  0.60 / 1_000_000},
    "gpt-4o":           {"input":  2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4.1-mini":     {"input":  0.40 / 1_000_000, "output":  1.60 / 1_000_000},
    "gpt-4.1":          {"input":  2.00 / 1_000_000, "output":  8.00 / 1_000_000},
    # Anthropic
    "claude-opus-4-6":  {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-opus-4-5":  {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-sonnet-4-6":{"input":  3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-5":{"input":  3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5": {"input":  0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    # Google (Gemini): grobe Richtwerte; bitte ggf. an Live-Pricing anpassen.
    "gemini-3.1-pro":   {"input":  2.00 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-2.5-pro":   {"input":  1.25 / 1_000_000, "output":  5.00 / 1_000_000},
    "gemini-2.0-pro":   {"input":  1.25 / 1_000_000, "output":  5.00 / 1_000_000},
    "gemini-1.5-pro":   {"input":  1.25 / 1_000_000, "output":  5.00 / 1_000_000},
}


@dataclass
class TokenCall:
    workflow_name: str
    iteration: int
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class TokenLedger:
    calls: list[TokenCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()

    def record(
        self,
        *,
        workflow_name: str,
        iteration: int,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        prices = PRICING_PER_TOKEN.get(model)
        if prices is None:
            cost = 0.0
        else:
            cost = (
                input_tokens * prices["input"] + output_tokens * prices["output"]
            )
        with self._lock:
            self.calls.append(
                TokenCall(
                    workflow_name=workflow_name,
                    iteration=iteration,
                    agent_name=agent_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            )

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        total_in = sum(c.input_tokens for c in calls)
        total_out = sum(c.output_tokens for c in calls)
        total_cost = sum(c.cost_usd for c in calls)
        per_agent: dict[str, dict] = {}
        for c in calls:
            slot = per_agent.setdefault(
                c.agent_name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            slot["calls"] += 1
            slot["input_tokens"] += c.input_tokens
            slot["output_tokens"] += c.output_tokens
            slot["cost_usd"] += c.cost_usd
        return {
            "total_calls": len(calls),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": total_cost,
            "per_agent": per_agent,
            "calls": [
                {
                    "workflow_name": c.workflow_name,
                    "iteration": c.iteration,
                    "agent_name": c.agent_name,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                }
                for c in calls
            ],
        }


# Modul-globaler Ledger.
LEDGER = TokenLedger()
