"""Synthetic timing, leakage and accounting tests; never opens market datasets."""
import numpy as np
import pandas as pd
import pytest
from tools.mnq_oracle_prepare import FEATURE_NAMES, causal_features
from tools.mnq_oracle_execution import alma_signals, replay
from tools.mnq_oracle_research import signals_from_probabilities, split_masks, classification_metrics


def candles(count=80):
    times = pd.date_range("2026-10-05T12:00:00Z", periods=count, freq="min")
    prices = 100 + np.round(np.sin(np.arange(count)/4)*8)/4
    return pd.DataFrame({"timestamp": times, "open": prices, "high": prices+1, "low": prices-1, "close": prices+.25, "volume": 100, "segment": 1})


def run(bars, signals, rule="time10", commission=1.5, ticks=1):
    return replay(bars, signals, np.full(len(bars), 4.), bars.timestamp.iloc[0], bars.timestamp.iloc[-1]+pd.Timedelta(minutes=1), rule, {"commission_per_side": commission, "slippage_ticks_per_side": ticks})


def test_features_prefix_invariant_and_no_target_columns():
    bars = candles()
    before = causal_features(bars)
    bars.loc[60:, ["open", "high", "low", "close"]] += 500
    after = causal_features(bars)
    pd.testing.assert_frame_equal(before.iloc[:60], after.iloc[:60])
    assert not any("teacher" in name or "exit" in name for name in FEATURE_NAMES)


def test_alma_causal_prefix_and_completed_bucket():
    bars = candles(96)
    full = alma_signals(bars)
    for length in (31, 43, 58, 85):
        np.testing.assert_array_equal(alma_signals(bars.iloc[:length]), full[:length])
    assert all((bars.timestamp.iloc[index].value // 60000000000) % 8 == 7 for index in np.flatnonzero(full))
    assert np.any(alma_signals(bars, lookahead=True) != full)


def test_next_open_fill_fees_slippage_and_no_samebar_entry():
    bars = candles(14)
    signals = np.zeros(len(bars), int)
    signals[0] = 1
    summary, trades, _ = run(bars, signals)
    assert trades[0]["entry_index"] == 1
    assert trades[0]["entry"] == bars.open.iloc[1]+.25
    assert trades[0]["held_bars"] == 10
    assert summary["commissions"] == 3
    assert summary["slippage"] == 1
    assert summary["net"] == summary["gross_price_move"]-4


@pytest.mark.parametrize("side", [-1, 1])
def test_ambiguous_bar_stop_before_target(side):
    bars = candles(4)
    bars.loc[1, ["high", "low"]] = [120, 80]
    summary, trades, _ = run(bars, np.array([side, 0, 0, 0]), "atr1_target1.5_time30")
    assert trades[0]["reason"] == "stop_first_ambiguous"
    assert trades[0]["net"] < 0
    assert summary["ambiguous_trades"] == 1


def test_gap_cancels_signal_and_preserves_unresolved_exposure():
    bars = candles(8)
    bars.loc[3:, "timestamp"] += pd.Timedelta(minutes=20)
    signals = np.zeros(8, int)
    signals[0] = 1
    summary, trades, events = run(bars, signals)
    assert trades[0]["reason"] == "unresolved_gap_next_observed_open"
    assert trades[0]["exit_raw"] == bars.open.iloc[3]
    assert summary["gap_unresolved_trades"] == 1
    signals[:] = 0
    signals[2] = 1
    summary, trades, events = run(bars, signals)
    assert not trades
    assert events[0]["kind"] == "signal_cancelled_gap"


def test_zero_volume_no_trade_and_short_accounting():
    bars = candles(13)
    signals = np.zeros(13, int)
    signals[0] = -1
    _, trades, _ = run(bars, signals)
    assert trades[0]["net"] == (trades[0]["entry_raw"]-trades[0]["exit_raw"])*2-4
    bars.loc[1, "volume"] = 0
    summary, trades, events = run(bars, signals)
    assert not trades and summary["cancelled_signals"] == 1


def test_cost_sensitivities_are_four_six_ten_dollars():
    bars = candles(14)
    signals = np.zeros(14, int)
    signals[0] = 1
    for commission, ticks, total in ((1.5, 1, 4), (2, 2, 6), (3, 4, 10)):
        summary, _, _ = run(bars, signals, commission=commission, ticks=ticks)
        assert summary["gross_price_move"]-summary["net"] == total


def test_no_future_end_liquidation_signal_and_flat_control():
    bars = candles(4)
    signals = np.array([0, 0, 0, 1])
    summary, trades, _ = run(bars, signals)
    assert not trades and summary["net"] == 0 and summary["max_drawdown_bar_close"] == 0


def test_purge_teacher_exit_used_only_for_removal():
    times = pd.date_range("2026-10-01T00:00:00Z", periods=400, freq="min")
    features = pd.DataFrame({"timestamp_utc": times, "feature_ready": True})
    teacher = pd.DataFrame({"entry_utc": [times[2]], "exit_utc": [times[210]]})
    plan = {"train_start": times[0].isoformat(), "validation_start": times[180].isoformat(), "final_start": times[330].isoformat(), "coverage_end_exclusive": (times[-1]+pd.Timedelta(minutes=1)).isoformat(), "purge_minutes": 60}
    masks = split_masks(features, teacher, plan)
    assert not masks["train"][2]
    assert not masks["train"][119:].any()
    assert not masks["development"][269:].any()
    assert masks["final"][330:].all()
    assert not np.any(masks["train"] & masks["development"])


def test_wait_dominance_and_metrics_not_accuracy_only():
    probabilities = np.array([[.4, .5, .1], [.6, .1, .3], [.1, .1, .8]])
    signals = signals_from_probabilities(probabilities, .5)
    np.testing.assert_array_equal(signals, [0, -1, 1])
    metrics = classification_metrics(np.array([0, -1, 1]), probabilities, signals)
    assert metrics["directional_event_recall"] == 1
    assert metrics["directions"]["1"]["average_precision"] == 1
    assert metrics["multiclass_brier"] > 0
