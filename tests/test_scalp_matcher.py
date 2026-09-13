"""Synthetic interface evidence, never evidence of market profitability."""

import numpy as np
import pandas as pd
import pytest

from nqscalp.execution import TradeSpec, simulate_one
from nqscalp.matcher import Matcher, replay


def fixture():
    c = 100 + np.arange(61) * 0.25
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-03T14:00Z", periods=61, freq="min"),
            "open": c - 0.25,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 100,
        }
    )
    signature = {
        "id": "synthetic",
        "family": "fixture",
        "session_bucket": "all",
        "conditions": [{"feature": "r5", "op": ">=", "value": 1.0}],
    }
    pattern = {
        "pattern_id": "synthetic",
        "signature": signature,
        "action": {"direction": 1, "target_points": 4, "stop_points": 4, "horizon": 3},
        "support": {
            "available_at": "2023-12-31T00:00Z",
            "sample_size": 200,
            "mean_net_points": 1.0,
            "score": 0.5,
        },
        "confirmation": {"passed": False},
    }
    return bars, {"schema_version": 1, "patterns": [pattern], "active_pattern_id": "synthetic"}


def test_unconfirmed_never_emits_action_in_normal_mode():
    bars, library = fixture()
    assert Matcher(library).match(bars, known_close="2024-01-03T21:00Z")["decision"] == "no_trade"


def test_diagnostic_provides_action_and_prefix_stream_parity():
    bars, library = fixture()
    matcher = Matcher(library, diagnostic=True)
    batch = matcher.match(bars, known_close="2024-01-03T21:00Z")
    for row in bars.to_dict("records"):
        streamed = matcher.update(row, known_close="2024-01-03T21:00Z")
    assert batch == streamed
    assert batch["decision"] == "buy" and batch["pattern_id"] == "synthetic"
    assert batch["target_offset_points"] == 4 and batch["stop_offset_points"] == -4
    assert batch["entry_rule"] == "next_bar_open" and batch["historical_sample"] == 200
    assert batch["horizon_minutes"] == 3


def test_future_support_occupied_position_and_insufficient_horizon_abstain():
    bars, library = fixture()
    matcher = Matcher(library, diagnostic=True)
    assert matcher.match(bars, flat=False)["reason"] == "position_occupied"
    assert (
        matcher.match(bars, known_close="2024-01-03T15:02Z")["reason"]
        == "insufficient_calendar_horizon"
    )


def test_replay_uses_same_decisions_and_actual_occupancy_as_streaming():
    _, library = fixture()
    c = 100 + np.arange(120) * 0.25
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-03T14:00Z", periods=120, freq="min"),
            "open": c - 0.25,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 100,
        }
    )
    schedule = pd.DataFrame(
        {
            "market_open": [pd.Timestamp("2024-01-03T14:00Z")],
            "market_close": [pd.Timestamp("2024-01-03T17:00Z")],
        }
    )
    pattern = library["patterns"][0]
    with pytest.raises(ValueError, match="unconfirmed"):
        replay(bars, pattern, schedule)
    ledger = replay(bars, pattern, schedule, diagnostic=True)
    matcher = Matcher(library, diagnostic=True, schedule=schedule)
    observations = []
    i = 60
    while i < len(bars):
        decision = matcher.match(bars.iloc[i - 60 : i + 1])
        if decision["decision"] == "buy":
            trade = simulate_one(bars, i, TradeSpec(**pattern["action"]))
            observations.append(trade)
            i = max(i + 1, trade["termination_index"])
        else:
            i += 1
    assert ledger.detection_index.tolist() == [r["detection_index"] for r in observations]
    np.testing.assert_allclose(
        ledger.net_points, [r["net_points"] for r in observations], equal_nan=True
    )


def test_invalid_risk_scope_and_missing_calendar_never_produce_orders():
    bars, library = fixture()
    assert Matcher(library, diagnostic=True).match(bars)["reason"] == "calendar_unavailable"
    for changes, reason in [
        ({"horizon": 46}, "rejected_out_of_scope"),
        ({"stop_points": 30}, "risk_limit"),
        ({"entry_kind": "limit"}, "unsupported_entry"),
    ]:
        _, modified = fixture()
        modified["patterns"][0]["action"].update(changes)
        assert (
            Matcher(modified, diagnostic=True).match(bars, known_close="2024-01-03T21:00Z")[
                "reason"
            ]
            == reason
        )
    library["patterns"][0]["support"]["available_at"] = "2025-01-01T00:00Z"
    assert (
        Matcher(library, diagnostic=True).match(bars, known_close="2024-01-03T21:00Z")["reason"]
        == "historical_support_not_available"
    )
