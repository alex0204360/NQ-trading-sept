"""Read-only entry economics audit for pre-2025 NQ research trade ledgers.

Descriptive diagnostics, NOT a new backtest or a trainable signal. No 2025 access.
Only completed-bar information is used as a potential signal-time feature;
post-entry excursions are stored as retrospective outcome diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEST_START = pd.Timestamp("2025-01-01T05:00:00Z")
POINT_VALUE = 20.0
ENTRY_COLUMNS = {
    "detection_index", "entry_index", "exit_index", "status", "reason",
    "direction", "entry_price", "exit_price", "net_dollars", "fees_dollars",
    "slippage_points", "entry_timestamp", "exit_timestamp",
}
CLOSE_EXITS = {"timeout", "hold_limit", "interval_end"}


def _time_column(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, errors="raise")
    if not isinstance(result.dtype, pd.DatetimeTZDtype) or result.isna().any():
        raise ValueError(f"{name} must contain non-null, timezone-aware times")
    return result.dt.tz_convert("UTC")


def _prefix_features(bars: pd.DataFrame, times: pd.Series, signal: int, direction: int):
    """At the close of signal candle, no later candle may influence these values."""
    if signal < 20:
        return None, None, None
    window = bars.iloc[signal - 20: signal + 1]
    window_times = times.iloc[signal - 20: signal + 1]
    if not (window_times.diff().iloc[1:] == pd.Timedelta(minutes=1)).all():
        return None, None, None
    if any(column in window and window[column].nunique(dropna=False) != 1
           for column in ("segment_id", "contract")):
        return None, None, None
    hi = window.high.to_numpy(dtype=float)
    lo = window.low.to_numpy(dtype=float)
    cl = window.close.to_numpy(dtype=float)
    tr = np.maximum(hi[1:] - lo[1:], np.maximum(abs(hi[1:] - cl[:-1]), abs(lo[1:] - cl[:-1])))
    atr14 = float(tr[-14:].mean())
    range5 = float(hi[-5:].max() - lo[-5:].min())
    range20 = float(hi[-20:].max() - lo[-20:].min())
    compression = range5 / range20 if range20 > 0 else None
    prior20_high, prior20_low = float(hi[:-1].max()), float(lo[:-1].min())
    directional_break = bool(cl[-1] > prior20_high if direction == 1 else cl[-1] < prior20_low)
    return atr14, compression, directional_break


def analyze_trade_paths(bars: pd.DataFrame, trades: pd.DataFrame):
    """Return trade-level diagnostics and a descriptive summary; never alter P&L.

    Bars must be the exact indexed, audited pre-2025 training series used by the
    simulator, not the raw Kaggle CSV. Index values in trades are bar positions.
    """
    needed_bars = {"timestamp", "open", "high", "low", "close"}
    if not needed_bars.issubset(bars.columns):
        raise ValueError(f"Missing bar fields: {sorted(needed_bars - set(bars.columns))}")
    if not ENTRY_COLUMNS.issubset(trades.columns):
        raise ValueError(f"Missing trade fields: {sorted(ENTRY_COLUMNS - set(trades.columns))}")
    times = _time_column(bars.timestamp, "bars.timestamp")
    if (times >= TEST_START).any():
        raise ValueError("Holdout boundary violated: entry audit accepts pre-2025 bars only")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Bars must have unique increasing timestamps")
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Missing/nonpositive OHLC")
    o, h, l, c = prices.T
    if ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l > h)).any():
        raise ValueError("Invalid OHLC geometry")
    if len(trades) and trades["status"].isin(["unresolved"]).any():
        # Do not drop censored losses while reporting apparent performance.
        raise ValueError("Unresolved trades present: account for their exposure before inference")
    closed = trades[trades.status == "closed"].copy()
    for name in ("detection_index", "entry_index", "exit_index"):
        values = pd.to_numeric(closed[name], errors="raise")
        if values.isna().any() or not (values == values.round()).all():
            raise ValueError(f"{name} must be integer positions")
        closed[name] = values.astype(int)
    if ((closed.detection_index < 0) | (closed.entry_index <= closed.detection_index)
            | (closed.exit_index < closed.entry_index) | (closed.exit_index >= len(bars))).any():
        raise ValueError("Trade indices violate causal entry/exit order")
    entry_times = _time_column(closed.entry_timestamp, "entry_timestamp")
    exit_times = _time_column(closed.exit_timestamp, "exit_timestamp")
    changed = np.zeros(len(bars), dtype=np.int64)
    changed[1:] = (times.diff().iloc[1:] != pd.Timedelta(minutes=1)).to_numpy(dtype=int)
    for name in ("contract", "segment_id"):
        if name in bars:
            changed[1:] |= (bars[name].iloc[1:].to_numpy() != bars[name].iloc[:-1].to_numpy())
    breaks = np.cumsum(changed)
    records = []
    for ix, (_, t) in enumerate(closed.iterrows()):
        signal, start, end = (int(t[k]) for k in ("detection_index", "entry_index", "exit_index"))
        if entry_times.iloc[ix] != times.iloc[start] or exit_times.iloc[ix] != times.iloc[end] + pd.Timedelta(minutes=1):
            raise ValueError("Trade timestamps disagree with audited bar positions")
        direction = int(t.direction)
        if direction not in (-1, 1):
            raise ValueError("Trade direction must be -1 or +1")
        if any(not np.isfinite(float(t[k])) for k in ("entry_price", "exit_price", "net_dollars", "fees_dollars", "slippage_points")):
            raise ValueError("Trade P&L, fills and costs must be finite")
        valid_path = bool(breaks[start] == breaks[end])
        # An intrabar exit may occur BEFORE the exit candle's high or low.
        # Exclude its entire bar unless the exit is explicitly at bar close.
        inclusive_end = end + 1 if str(t.reason) in CLOSE_EXITS else end
        if valid_path and inclusive_end > start:
            view = prices[start: inclusive_end]
            favorable = float(view[:, 1].max() - t.entry_price if direction == 1
                              else t.entry_price - view[:, 2].min())
            adverse = float(t.entry_price - view[:, 2].min() if direction == 1
                            else view[:, 1].max() - t.entry_price)
            mfe, mae = max(0.0, favorable), max(0.0, adverse)
        else:
            mfe, mae = None, None
        atr14, compression, prior20_breakout = _prefix_features(bars, times, signal, direction)
        friction_points = float(t.fees_dollars) / POINT_VALUE + float(t.slippage_points)
        ratio = atr14 / friction_points if atr14 is not None and friction_points > 0 else None
        if ratio is None:
            ratio_band = "unknown"
        elif ratio < 2:
            ratio_band = "<2"
        elif ratio < 5:
            ratio_band = "2-5"
        else:
            ratio_band = ">=5"
        local = (times.iloc[signal] + pd.Timedelta(minutes=1)).tz_convert("America/New_York")
        mins = local.hour * 60 + local.minute
        session = ("open" if 570 <= mins < 690 else "midday" if 690 <= mins < 840
                   else "close" if 840 <= mins < 960 else "overnight")
        records.append({
            "trade_row": str(t.name), "signal_index": signal, "entry_index": start,
            "exit_index": end, "signal_et": local.isoformat(), "entry_et_date": entry_times.iloc[ix].tz_convert("America/New_York").date().isoformat(),
            "session": session, "side": "long" if direction == 1 else "short",
            "exit_reason": str(t.reason), "net_dollars": float(t.net_dollars),
            "signal_atr14_points": atr14, "compression_5_to_20": compression,
            "breakout_prior20": prior20_breakout, "friction_points": friction_points,
            "atr_to_friction": ratio, "atr_to_friction_band": ratio_band,
            "pre_exit_mfe_points_lower_bound": mfe, "pre_exit_mae_points_lower_bound": mae,
            "path_complete": valid_path,
            "intrabar_exit_bar_excluded": str(t.reason) not in CLOSE_EXITS,
        })
    rows = pd.DataFrame.from_records(records)
    def group(by):
        if rows.empty:
            return []
        result = []
        for name, frame in rows.groupby(by, dropna=False, sort=True):
            result.append({"cohort": str(name), "trades": len(frame),
                           "total_net_dollars": round(float(frame.net_dollars.sum()), 2),
                           "mean_net_dollars": round(float(frame.net_dollars.mean()), 3),
                           "profitable_fraction": round(float((frame.net_dollars > 0).mean()), 5)})
        return result
    summary = {
        "interpretation": "Exploratory attribution only; not a causal filter or profitability test",
        "closed_trades": len(rows), "rejected_or_nonclosed_rows": len(trades) - len(rows),
        "path_incomplete_rows": 0 if rows.empty else int((~rows.path_complete).sum()),
        "signal_features_are_causal": True, "excursions_are_post_entry_outcomes": True,
        "2025_holdout_opened": False,
        "total_net_dollars": 0 if rows.empty else round(float(rows.net_dollars.sum()), 2),
        "by_session": group("session"), "by_atr_to_friction": group("atr_to_friction_band"),
        "by_signal_breakout": group("breakout_prior20"),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trades", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON summary path; detailed CSV written alongside")
    args = parser.parse_args()
    # Verify frozen training/schedule checksums before loading audited bars.
    # Never read the raw CSV or held-out 2025 data.
    from nqscalp.research import INPUT_HASHES, digest, write_json

    for name, expected_sha in INPUT_HASHES.items():
        if digest(args.root / "data" / name) != expected_sha:
            raise ValueError(f"Verified research input changed: {name}")
    bars = pd.read_parquet(args.root / "data/training.parquet")
    ledgers = pd.read_csv(args.trades)
    detail, summary = analyze_trade_paths(bars, ledgers)
    summary["input_training_sha256"] = digest(args.root / "data/training.parquet")
    summary["input_trade_sha256"] = digest(args.trades)
    summary["detail_csv"] = str(args.output.with_suffix(".csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output.with_suffix(".csv"), index=False)
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
