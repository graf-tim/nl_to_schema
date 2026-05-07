"""TokenLedger: Aggregation und Pricing."""
from agents._telemetry import LEDGER, PRICING_PER_TOKEN


def test_record_aggregates_and_prices_correctly():
    LEDGER.reset()
    LEDGER.record(
        workflow_name="sa",
        iteration=0,
        agent_name="sa_generator",
        model="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500,
    )
    LEDGER.record(
        workflow_name="sa",
        iteration=0,
        agent_name="llm_as_judge",
        model="gpt-4o-mini",
        input_tokens=2000,
        output_tokens=300,
    )
    summary = LEDGER.summary()
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 3000
    assert summary["total_output_tokens"] == 800
    assert summary["total_tokens"] == 3800

    expected = (
        3000 * PRICING_PER_TOKEN["gpt-4o-mini"]["input"]
        + 800 * PRICING_PER_TOKEN["gpt-4o-mini"]["output"]
    )
    assert abs(summary["total_cost_usd"] - expected) < 1e-9
    assert set(summary["per_agent"].keys()) == {"sa_generator", "llm_as_judge"}
    assert summary["per_agent"]["sa_generator"]["calls"] == 1
    assert summary["per_agent"]["llm_as_judge"]["input_tokens"] == 2000


def test_unknown_model_costs_zero():
    LEDGER.reset()
    LEDGER.record(
        workflow_name="sa",
        iteration=0,
        agent_name="x",
        model="some-other-model",
        input_tokens=1000,
        output_tokens=1000,
    )
    summary = LEDGER.summary()
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_tokens"] == 2000


def test_reset_clears_calls():
    LEDGER.reset()
    LEDGER.record(
        workflow_name="x",
        iteration=0,
        agent_name="x",
        model="gpt-4o-mini",
        input_tokens=1,
        output_tokens=1,
    )
    LEDGER.reset()
    summary = LEDGER.summary()
    assert summary["total_calls"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_cost_usd"] == 0.0
    assert summary["calls"] == []
