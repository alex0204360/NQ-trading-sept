"""Conservative one-contract executions; scalar oracle and vector research path."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeSpec:
    direction: int
    target_points: float
    stop_points: float
    horizon: int
    entry_kind: str = "market"
    entry_offset_points: float = 0.0
    pending_bars: int = 1


@dataclass(frozen=True)
class ExecutionConfig:
    tick_size: float = 0.25
    point_value: float = 20.0
    fee_per_side: float = 5.0
    slippage_ticks: int = 1
    initial_balance: float = 50000.0
    max_risk_dollars: float = 500.0


@dataclass(frozen=True)
class PreparedBars:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    segment: np.ndarray

    def __len__(self):
        return len(self.open)


def prepare_bars(df):
    if isinstance(df, PreparedBars):
        return df
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if any(k not in df for k in required):
        raise ValueError("Expected standard timestamp/OHLCV bars")
    ts = pd.to_datetime(df.timestamp)
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or ts.isna().any():
        raise ValueError("Timestamp must be timezone aware and finite")
    times = ts.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
    if np.any(np.diff(times) <= 0) or np.any(times % 60_000_000_000):
        raise ValueError("Timestamps must increase uniquely on minute boundaries")
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    vol = df.volume.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite positive")
    if not np.isfinite(vol).all() or (vol < 0).any():
        raise ValueError("Volume must be finite nonnegative")
    if not np.allclose(prices * 4, np.rint(prices * 4), atol=1e-8, rtol=0):
        raise ValueError("NQ prices must align to quarter-point ticks")
    o, h, l, c = prices.T
    if ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l > h)).any():
        raise ValueError("Invalid OHLC ordering")
    changes = np.zeros(len(df), dtype=bool)
    for name in ("contract", "segment_id"):
        if name in df:
            if df[name].isna().any():
                raise ValueError("contract/segment identity cannot be null")
            changes |= df[name].ne(df[name].shift()).to_numpy()
    return PreparedBars(times, o, h, l, c, changes.cumsum())


def _reason(spec, costs):
    values = [costs.tick_size, costs.point_value, costs.initial_balance, costs.max_risk_dollars]
    if any(not np.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Positive finite execution configuration required")
    if (
        not np.isfinite(costs.fee_per_side)
        or costs.fee_per_side < 0
        or not isinstance(costs.slippage_ticks, (int, np.integer))
        or costs.slippage_ticks < 0
    ):
        raise ValueError("Nonnegative fees and integer slippage required")
    if not isinstance(spec.horizon, (int, np.integer)) or not 1 <= spec.horizon <= 45:
        return "rejected_out_of_scope"
    if spec.direction not in [-1, 1] or any(
        not np.isfinite(x) or x <= 0 for x in [spec.target_points, spec.stop_points]
    ):
        return "invalid_spec"
    if any(
        not np.isclose(x / costs.tick_size, round(x / costs.tick_size), atol=1e-8, rtol=0)
        for x in [spec.target_points, spec.stop_points]
    ):
        return "invalid_tick_distance"
    if spec.entry_kind != "market" or spec.entry_offset_points != 0 or spec.pending_bars != 1:
        return "unsupported_entry"
    risk = (
        spec.stop_points + 2 * costs.tick_size * costs.slippage_ticks
    ) * costs.point_value + 2 * costs.fee_per_side
    if risk > costs.max_risk_dollars:
        return "risk_limit"
    return None


def _initialize(b, indices, spec, costs):
    raw = np.asarray(indices)
    if raw.ndim != 1 or (
        raw.size and (raw.dtype.kind not in "iu" or (raw < 0).any() or (raw >= len(b)).any())
    ):
        raise ValueError("Detection indices must be in-range integers")
    idx = raw.astype(np.int64)
    n = len(idx)
    result = {
        "detection_index": idx,
        "entry_index": np.full(n, -1, dtype=int),
        "exit_index": np.full(n, -1, dtype=int),
        "termination_index": idx.copy(),
        "status": np.full(n, "unresolved", dtype=object),
        "reason": np.full(n, "missing_activation", dtype=object),
    }
    for key in [
        "reference_entry",
        "exit_reference_price",
        "entry_price",
        "exit_price",
        "stop_price",
        "target_price",
        "gross_points",
        "fill_pnl_points",
        "net_points",
        "net_dollars",
        "fees_dollars",
        "slippage_points",
        "holding_bars",
        "holding_minutes",
        "mae_points",
    ]:
        result[key] = np.full(n, np.nan)
    rejection = _reason(spec, costs)
    if rejection:
        result["status"][:] = "rejected"
        result["reason"][:] = rejection
    return result, rejection


def _frame(b, result, spec, costs):
    r = result
    closed = r["status"] == "closed"
    d = spec.direction
    slip = costs.tick_size * costs.slippage_ticks
    r["exit_price"][closed] = r["exit_reference_price"][closed] - d * slip
    r["gross_points"][closed] = d * (
        r["exit_reference_price"][closed] - r["reference_entry"][closed]
    )
    r["fill_pnl_points"][closed] = d * (r["exit_price"][closed] - r["entry_price"][closed])
    r["fees_dollars"][closed] = 2 * costs.fee_per_side
    r["slippage_points"][closed] = 2 * slip
    r["net_points"][closed] = (
        r["fill_pnl_points"][closed] - 2 * costs.fee_per_side / costs.point_value
    )
    r["net_dollars"][closed] = r["net_points"][closed] * costs.point_value
    r["holding_bars"][closed] = r["exit_index"][closed] - r["entry_index"][closed] + 1
    r["holding_minutes"][closed] = r["holding_bars"][closed]
    frame = pd.DataFrame(r)
    for field in ["detection", "entry", "exit"]:
        indexes = r[field + "_index"]
        times = np.full(len(indexes), np.iinfo(np.int64).min, dtype=np.int64)
        valid = indexes >= 0
        times[valid] = b.timestamp[indexes[valid]]
        if field in ("detection", "exit"):
            # Intrabar ordering is unknown; exits are observable by resolving-bar close.
            times[valid] += 60_000_000_000
        frame[field + "_timestamp"] = pd.to_datetime(times, utc=True)
    if isinstance(spec.horizon, (int, np.integer)) and 1 <= spec.horizon <= 45:
        frame["expiry_timestamp"] = pd.to_datetime(
            b.timestamp[r["detection_index"]] + (spec.horizon + 1) * 60_000_000_000, utc=True
        )
    else:
        frame["expiry_timestamp"] = pd.NaT
    return frame


DEFAULT_COSTS = ExecutionConfig()


def simulate_one(bars, index, spec, costs=DEFAULT_COSTS):
    """Independent scalar execution oracle, accepting only future consecutive bars."""
    b = prepare_bars(bars)
    r, rejection = _initialize(b, [index], spec, costs)
    if not rejection:
        entry = index + 1
        if (
            entry < len(b)
            and b.timestamp[entry] == b.timestamp[index] + 60_000_000_000
            and b.segment[entry] == b.segment[index]
        ):
            ref = b.open[entry]
            d = spec.direction
            actual = ref + d * costs.tick_size * costs.slippage_ticks
            stop, target = ref - d * spec.stop_points, ref + d * spec.target_points
            for key, value in [
                ("entry_index", entry),
                ("reference_entry", ref),
                ("entry_price", actual),
                ("stop_price", stop),
                ("target_price", target),
                ("mae_points", 0.0),
            ]:
                r[key][0] = value
            r["reason"][0] = "missing_future_bar"
            for step in range(spec.horizon):
                j = entry + step
                r["termination_index"][0] = min(j, len(b) - 1)
                if (
                    j >= len(b)
                    or b.timestamp[j] != b.timestamp[index] + (step + 1) * 60_000_000_000
                    or b.segment[j] != b.segment[index]
                ):
                    r["reason"][0] = "unexpected_gap" if j < len(b) else "missing_future_bar"
                    break
                adverse = actual - b.low[j] if d == 1 else b.high[j] - actual
                r["mae_points"][0] = max(r["mae_points"][0], adverse)
                hit_stop = b.low[j] <= stop if d == 1 else b.high[j] >= stop
                hit_target = (
                    b.high[j] >= target + costs.tick_size
                    if d == 1
                    else b.low[j] <= target - costs.tick_size
                )
                reason, price = None, np.nan
                open_target = (
                    b.open[j] >= target + costs.tick_size
                    if d == 1
                    else b.open[j] <= target - costs.tick_size
                )
                if open_target:
                    reason, price = "target", target
                elif hit_stop:
                    reason = "stop"
                    price = min(stop, b.open[j]) if d == 1 else max(stop, b.open[j])
                elif hit_target:
                    reason, price = "target", target
                elif step == spec.horizon - 1:
                    reason, price = "timeout", b.close[j]
                if reason:
                    r["status"][0], r["reason"][0] = "closed", reason
                    r["exit_index"][0], r["exit_reference_price"][0] = j, price
                    break
        elif entry < len(b):
            r["termination_index"][0] = entry
            r["reason"][0] = "unexpected_gap"
    return _frame(b, r, spec, costs).iloc[0].to_dict()


def simulate_batch(bars, indices, spec, costs=DEFAULT_COSTS):
    """Vector horizon scan; no Python loop over trade detections."""
    b = prepare_bars(bars)
    r, rejection = _initialize(b, indices, spec, costs)
    idx = r["detection_index"]
    if rejection or not len(idx):
        return _frame(b, r, spec, costs)
    entry = idx + 1
    safe_entry = np.minimum(entry, len(b) - 1)
    active = (entry < len(b)) & (b.timestamp[safe_entry] == b.timestamp[idx] + 60_000_000_000)
    active &= b.segment[safe_entry] == b.segment[idx]
    gap = (entry < len(b)) & ~active
    r["reason"][gap] = "unexpected_gap"
    r["termination_index"][gap] = entry[gap]
    d = spec.direction
    r["entry_index"][active] = entry[active]
    r["reference_entry"][active] = b.open[entry[active]]
    r["entry_price"][active] = (
        r["reference_entry"][active] + d * costs.tick_size * costs.slippage_ticks
    )
    r["stop_price"][active] = r["reference_entry"][active] - d * spec.stop_points
    r["target_price"][active] = r["reference_entry"][active] + d * spec.target_points
    r["mae_points"][active] = 0.0
    for step in range(spec.horizon):
        where = np.flatnonzero(active)
        if not len(where):
            break
        j = entry[where] + step
        safe = np.minimum(j, len(b) - 1)
        valid = (j < len(b)) & (
            b.timestamp[safe] == b.timestamp[idx[where]] + (step + 1) * 60_000_000_000
        )
        valid &= b.segment[safe] == b.segment[idx[where]]
        r["termination_index"][where] = safe
        bad = where[~valid]
        r["reason"][bad] = np.where(j[~valid] < len(b), "unexpected_gap", "missing_future_bar")
        active[bad] = False
        w, j = where[valid], j[valid]
        adverse = r["entry_price"][w] - b.low[j] if d == 1 else b.high[j] - r["entry_price"][w]
        r["mae_points"][w] = np.maximum(r["mae_points"][w], adverse)
        stop, target = r["stop_price"][w], r["target_price"][w]
        hit_stop = b.low[j] <= stop if d == 1 else b.high[j] >= stop
        hit_target = (
            b.high[j] >= target + costs.tick_size
            if d == 1
            else b.low[j] <= target - costs.tick_size
        )
        open_target = (
            b.open[j] >= target + costs.tick_size
            if d == 1
            else b.open[j] <= target - costs.tick_size
        )
        hit_stop = hit_stop & ~open_target
        exit_now = hit_stop | hit_target | (step == spec.horizon - 1)
        ew, ej = w[exit_now], j[exit_now]
        stop_fill = np.minimum(stop, b.open[j]) if d == 1 else np.maximum(stop, b.open[j])
        price = np.where(hit_stop, stop_fill, np.where(hit_target, target, b.close[j]))
        reason = np.where(hit_stop, "stop", np.where(hit_target, "target", "timeout"))
        r["status"][ew] = "closed"
        r["reason"][ew] = reason[exit_now]
        r["exit_index"][ew] = ej
        r["exit_reference_price"][ew] = price[exit_now]
        active[ew] = False
    return _frame(b, r, spec, costs)
