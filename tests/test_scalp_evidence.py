"""Hand-computed accounting and dependence-aware inference regressions."""

import numpy as np
import pandas as pd
import pytest

from nqscalp.evidence import acceptance, day_evidence, holm, summarize


def ledger(values=(2.0, -1.0, 3.0)):
    return pd.DataFrame(
        [
            {
                "status": "closed",
                "net_points": v,
                "net_dollars": 20 * v,
                "holding_bars": i + 1,
                "mae_points": 2.0,
                "fees_dollars": 10.0,
                "entry_timestamp": pd.Timestamp("2024-01-02T15:00Z") + pd.Timedelta(days=i),
                "exit_timestamp": pd.Timestamp("2024-01-02T15:01Z") + pd.Timedelta(days=i),
            }
            for i, v in enumerate(values)
        ]
    )


def test_exact_accounting_cost_stress_and_excursions():
    s = summarize(ledger(), ["2024-01-02", "2024-01-03", "2024-01-04"])
    assert s["net_points"] == 4
    assert s["net_dollars"] == 80
    assert s["mean_net_points"] == pytest.approx(4 / 3)
    assert s["stress_net_points"] == 2.5
    assert s["severe_stress_net_points"] == 1
    assert s["closed_drawdown_dollars"] == 20
    assert s["intratrade_drawdown_dollars"] == 70
    assert s["final_balance_dollars"] == 50080
    assert s["resolution_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert 0 < s["win_rate_interval"][0] < s["win_rate_interval"][1] < 1
    assert s["low_confidence"]


def test_unresolved_is_unpriced_not_zero():
    t = ledger()
    t.loc[1, "status"] = "unresolved"
    t.loc[1, ["net_points", "net_dollars"]] = np.nan
    s = summarize(t, [])
    assert s["n_unresolved"] == 1 and s["n_closed"] == 2
    assert s["net_points"] == 5
    assert s["final_balance_dollars"] is None
    assert s["unpriced_outcomes"]


def test_accounting_and_overlap_are_detected():
    t = ledger()
    t.loc[0, "net_dollars"] = 41
    t.loc[0, "exit_timestamp"] = t.loc[1, "entry_timestamp"] + pd.Timedelta(minutes=1)
    s = summarize(t, [])
    assert s["accounting_violations"] == 1
    assert s["overlap_violations"] == 1


def test_equal_exit_and_next_entry_not_overlap():
    t = ledger()
    t.loc[0, "exit_timestamp"] = t.loc[1, "entry_timestamp"]
    assert summarize(t, [])["overlap_violations"] == 0


def test_missing_closed_pnl_is_violation():
    t = ledger()
    t.loc[0, "net_points"] = np.nan
    assert summarize(t, [])["accounting_violations"] > 0


def test_daily_calendar_zeros_and_timezone():
    t = ledger((1.0,))
    t.loc[0, "entry_timestamp"] = pd.Timestamp("2024-01-02T01:00Z")
    t.loc[0, "exit_timestamp"] = pd.Timestamp("2024-01-02T01:01Z")
    e = day_evidence(t, ["2024-01-01", "2024-01-02"], reps=99)
    assert e["calendar_dates"] == 2
    assert e["daily_trade_counts"] == [1, 0]
    assert e["daily_net_points"] == [1.0, 0.0]


def test_bootstrap_deterministic_and_constant_positive():
    t = ledger(tuple([2.0] * 80))
    days = list(pd.date_range("2024-01-02", periods=80).strftime("%Y-%m-%d"))
    a = day_evidence(t, days, reps=99, seed=7)
    assert a == day_evidence(t, days, reps=99, seed=7)
    assert a["lower95"] == 2 and a["upper95"] == 2
    assert a["p_value"] == 0.01
    assert set(a["blocks"]) == {"5", "20"}


def test_empty_evidence_and_calendar_omission():
    e = day_evidence(pd.DataFrame(), [], reps=19)
    assert e["lower95"] is None and e["p_value"] == 1
    with pytest.raises(ValueError, match="calendar"):
        day_evidence(ledger(), ["2024-01-02"], reps=19)


def test_holm_stepdown_and_invalid_values():
    assert holm({"a": 0.001, "b": 0.013, "c": 0.014}) == {"a": True, "b": False, "c": False}
    assert holm({}) == {}
    with pytest.raises(ValueError):
        holm({"a": float("nan")})


def test_acceptance_all_independent_gates():
    s = {
        "n_closed": 150,
        "n_unresolved": 0,
        "trade_dates": 40,
        "event_weeks": 12,
        "net_points": 75,
        "mean_net_points": 0.5,
        "stress_net_points": 1,
        "closed_drawdown_dollars": 5000,
        "intratrade_drawdown_dollars": 5000,
        "accounting_violations": 0,
        "overlap_violations": 0,
    }
    e = {"lower95": 0.01}
    windows = [{"n_closed": 20, "net_points": 1}] * 4
    assert acceptance(s, e, windows) == []
    assert acceptance(s, e, windows[:3], heldout=True) == []
    assert acceptance(s, e, windows[:3])
    assert acceptance({**s, "n_unresolved": 1}, e, windows)
    assert acceptance({**s, "stress_net_points": 0}, e, windows)
    assert acceptance(s, {"lower95": None}, windows)
    assert acceptance(s, e, [{"n_closed": 20, "net_points": -1}] + windows[:2], heldout=True)
