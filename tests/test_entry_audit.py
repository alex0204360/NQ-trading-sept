import pandas as pd
import pytest

from tools.entry_audit import analyze_trade_paths


def fixture():
    start = pd.Timestamp("2024-02-01T14:00:00Z")
    rows = []
    for i in range(26):
        p = 20000.0 + 0.25 * i
        rows.append({"timestamp": start + pd.Timedelta(minutes=i),
                     "open": p, "high": p + 2, "low": p - 2, "close": p + 0.25,
                     "volume": 100})
    bars = pd.DataFrame(rows)
    trade = {"detection_index": 20, "entry_index": 21, "exit_index": 22,
             "status": "closed", "reason": "stop", "direction": 1,
             "entry_price": bars.loc[21, "open"] + 0.25,
             "exit_price": bars.loc[22, "open"] - 2, "net_dollars": -65.,
             "fees_dollars": 10., "slippage_points": 0.5,
             "entry_timestamp": bars.loc[21, "timestamp"],
             "exit_timestamp": bars.loc[22, "timestamp"] + pd.Timedelta(minutes=1)}
    return bars, pd.DataFrame([trade])


def test_stop_excludes_exit_bar_high_low():
    bars, trades = fixture()
    # This dramatic exit-candle high occurred after an unknown stop time.
    bars.loc[22, "high"] = 20100.
    rows, info = analyze_trade_paths(bars, trades)
    assert rows.iloc[0].pre_exit_mfe_points_lower_bound == pytest.approx(1.75)
    assert rows.iloc[0].intrabar_exit_bar_excluded
    assert info["2025_holdout_opened"] is False
    assert info["closed_trades"] == 1


def test_timeout_includes_exit_bar():
    bars, trades = fixture()
    trades.loc[0, "reason"] = "timeout"
    bars.loc[22, "high"] = 20100.
    rows, _ = analyze_trade_paths(bars, trades)
    assert rows.iloc[0].pre_exit_mfe_points_lower_bound > 90
    assert not rows.iloc[0].intrabar_exit_bar_excluded


def test_signal_features_do_not_use_future_prices():
    bars, trades = fixture()
    a, _ = analyze_trade_paths(bars, trades)
    bars.loc[22, "high"] = 20100.
    b, _ = analyze_trade_paths(bars, trades)
    assert a.iloc[0].signal_atr14_points == b.iloc[0].signal_atr14_points
    assert a.iloc[0].compression_5_to_20 == b.iloc[0].compression_5_to_20
    assert a.iloc[0].breakout_prior20 == b.iloc[0].breakout_prior20


def test_unresolved_trade_blocks_inference():
    bars, trades = fixture()
    trades.loc[0, "status"] = "unresolved"
    with pytest.raises(ValueError, match="Unresolved"):
        analyze_trade_paths(bars, trades)


def test_holdout_not_accepted():
    bars, trades = fixture()
    bars.loc[25, "timestamp"] = pd.Timestamp("2025-01-02T14:00Z")
    with pytest.raises(ValueError, match="Holdout"):
        analyze_trade_paths(bars, trades)


def test_gaps_reported_not_deleted():
    bars, trades = fixture()
    bars.loc[22:, "timestamp"] += pd.Timedelta(minutes=2)
    trades.loc[0, "exit_timestamp"] += pd.Timedelta(minutes=2)
    rows, info = analyze_trade_paths(bars, trades)
    assert info["path_incomplete_rows"] == 1
    assert rows.iloc[0].net_dollars == -65.
    assert pd.isna(rows.iloc[0].pre_exit_mfe_points_lower_bound)
