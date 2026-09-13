"""Completed-bar price trajectories and fixed, auditable event definitions."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate standard UTC interval-start bars without filling or removing rows."""
    required = ["timestamp", *PRICE_COLUMNS, "volume"]
    if any(c not in bars for c in required):
        raise ValueError("timestamp and standard OHLCV columns are required")
    result = bars.copy()
    try:
        timestamps = pd.DatetimeIndex(result.timestamp)
        if timestamps.tz is None or timestamps.hasnans:
            raise ValueError("timestamps must be aware and nonnull")
        timestamps = timestamps.tz_convert("UTC").as_unit("ns")
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError("timestamps must be unique and increasing")
        if (timestamps.asi8 % pd.Timedelta(minutes=1).value).any():
            raise ValueError("timestamps must be aligned to minute starts")
        prices = result[PRICE_COLUMNS].to_numpy(dtype=float)
        volumes = result.volume.to_numpy(dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid OHLCV types") from exc
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("OHLC must be finite and positive")
    if not np.allclose(prices * 4, np.rint(prices * 4), rtol=0, atol=1e-7):
        raise ValueError("OHLC must align to NQ 0.25 ticks")
    if not np.isfinite(volumes).all() or (volumes < 0).any():
        raise ValueError("volume must be finite and nonnegative")
    o, h, low, c = prices.T
    if ((h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low)).any():
        raise ValueError("inconsistent OHLC bounds")
    result["timestamp"] = timestamps
    return result


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Features use only completed bars; extrema/compression exclude current bar."""
    b = validate_bars(bars)
    # Work positionally so duplicated dataframe index labels cannot affect arithmetic.
    original_index = b.index
    b = b.reset_index(drop=True)
    boundary = b.timestamp.diff().ne(pd.Timedelta(minutes=1))
    for name in ("contract", "segment_id"):
        if name in b:
            if b[name].isna().any():
                raise ValueError(f"{name} cannot contain null values")
            boundary |= b[name].ne(b[name].shift())
    group = boundary.cumsum()

    def lag(series, periods=1):
        return series.groupby(group).shift(periods)

    def rolling(series, window, operation):
        return series.groupby(group).transform(lambda s: getattr(s.rolling(window), operation)())

    f = pd.DataFrame(index=b.index)
    for name in PRICE_COLUMNS:
        f[name] = b[name].astype(float)
    previous = lag(b.close)
    tr = pd.concat(
        [b.high - b.low, (b.high - previous).abs(), (b.low - previous).abs()], axis=1
    ).max(axis=1)
    atr = rolling(tr, 14, "mean").clip(lower=0.25)
    f["atr14"] = atr
    for k in (1, 2, 3, 5, 10, 20, 40):
        f[f"r{k}"] = (b.close - lag(b.close, k)) / atr
    for k in range(1, 13):
        f[f"seq{k}"] = (b.close - lag(b.close, k)) / atr
    span = (b.high - b.low).clip(lower=0.25)
    f["body_signed"] = (b.close - b.open) / atr
    f["body_fraction"] = (b.close - b.open).abs() / span
    f["upper_wick"] = (b.high - b[["open", "close"]].max(axis=1)) / span
    f["lower_wick"] = (b[["open", "close"]].min(axis=1) - b.low) / span
    f["range_atr"] = (b.high - b.low) / atr
    for k in (5, 20, 60):
        f[f"prior_high_{k}"] = rolling(lag(b.high), k, "max")
        f[f"prior_low_{k}"] = rolling(lag(b.low), k, "min")
    prior_range20 = (f.prior_high_20 - f.prior_low_20).clip(lower=0.25)
    f["compression5_20"] = (f.prior_high_5 - f.prior_low_5) / prior_range20
    distance10 = rolling((b.close - previous).abs(), 10, "sum")
    f["efficiency10"] = (b.close - lag(b.close, 10)).abs() / distance10.replace(0, np.nan)
    f["efficiency10"] = f.efficiency10.fillna(0)
    f["position20"] = (b.close - f.prior_low_20) / prior_range20
    prior_volume = rolling(lag(b.volume), 20, "mean")
    f["volume_ratio"] = b.volume / prior_volume.clip(lower=1)
    f["slope20"] = f.r20 / 20
    f["pullback_long"] = (f.prior_high_5 - b.close) / atr
    f["pullback_short"] = (b.close - f.prior_low_5) / atr
    f["breakout20"] = (b.close - f.prior_high_20) / atr
    f["breakdown20"] = (f.prior_low_20 - b.close) / atr
    f["high_sweep20"] = (
        ((b.high - f.prior_high_20) / atr).clip(lower=0).where(b.close < f.prior_high_20, 0)
    )
    f["low_sweep20"] = (
        ((f.prior_low_20 - b.low) / atr).clip(lower=0).where(b.close > f.prior_low_20, 0)
    )
    # Session classification uses detection time, when the bar is complete.
    et = (b.timestamp + pd.Timedelta(minutes=1)).dt.tz_convert("America/New_York")
    f["minute_et"] = et.dt.hour * 60 + et.dt.minute
    f["session_bucket"] = np.select(
        [
            f.minute_et.between(570, 659),
            f.minute_et.between(660, 839),
            f.minute_et.between(840, 959),
        ],
        ["open", "midday", "late"],
        default="outside_rth",
    )
    f["valid"] = b.groupby(group).cumcount().ge(60)
    f.index = original_index
    return f


def calendar_horizons(bars: pd.DataFrame, schedule: pd.DataFrame, period_end=None) -> np.ndarray:
    """Known time available after detection, independent of future observed bars."""
    times = pd.DatetimeIndex(bars.timestamp)
    if times.tz is None or times.hasnans:
        raise ValueError("aware nonnull timestamps required")
    activation = (times.tz_convert("UTC").as_unit("ns") + pd.Timedelta(minutes=1)).asi8
    ends = np.zeros(len(times), dtype=np.int64)
    minute = pd.Timedelta(minutes=1).value
    bound = None
    if period_end is not None:
        bound = pd.Timestamp(period_end)
        if bound.tzinfo is None:
            raise ValueError("period_end must be timezone aware")
        bound = bound.value
    for row in schedule.itertuples():
        start = pd.Timestamp(row.market_open)
        close = pd.Timestamp(row.market_close)
        if start.tzinfo is None or close.tzinfo is None or close <= start:
            raise ValueError("invalid schedule interval")
        bs = getattr(row, "break_start", pd.NaT)
        be = getattr(row, "break_end", pd.NaT)
        intervals = [(start, close)]
        if pd.notna(bs) and pd.notna(be):
            bs, be = pd.Timestamp(bs), pd.Timestamp(be)
            if not start <= bs <= be <= close:
                raise ValueError("invalid scheduled break")
            intervals = [(start, bs), (be, close)]
        for begin, end in intervals:
            endpoint = end.value if bound is None else min(end.value, bound)
            # Current completed bar must itself belong to this interval.
            eligible = (activation - minute >= begin.value) & (activation <= endpoint)
            remaining = np.clip((endpoint - activation) // minute, 0, 45)
            ends = np.maximum(ends, np.where(eligible, remaining, 0))
    return ends.astype(int)


def event_definitions() -> list[dict]:
    """Prospectively fixed grammar. No fitted thresholds or outcome access."""
    result = []
    for direction in (1, -1):
        positive = ">=" if direction == 1 else "<="
        side = "long" if direction == 1 else "short"
        breakout = "breakout20" if direction == 1 else "breakdown20"
        wick = "lower_wick" if direction == 1 else "upper_wick"
        sweep = "low_sweep20" if direction == 1 else "high_sweep20"
        families = {
            "momentum": [
                ("r5", positive, direction * 1.0),
                ("r20", positive, direction * 1.0),
                ("efficiency10", ">=", 0.5),
            ],
            "pullback": [
                ("r20", positive, direction * 1.0),
                (f"pullback_{side}", ">=", 0.5),
                ("body_signed", positive, direction * 0.2),
            ],
            "compression_breakout": [("compression5_20", "<=", 0.35), (breakout, ">=", 0.0)],
            "breakout": [(breakout, ">=", 0.25), ("range_atr", ">=", 1.0)],
            "failed_break_rejection": [(sweep, ">=", 0.25), (wick, ">=", 0.35)],
            "exhaustion_reversal": [
                ("r10", "<=" if direction == 1 else ">=", -direction * 2.0),
                ("body_signed", positive, direction * 0.2),
                (wick, ">=", 0.35),
            ],
        }
        for family, conditions in families.items():
            for context in ("all", "open", "midday", "late"):
                result.append(
                    {
                        "id": f"{family}_{side}_{context}",
                        "family": family,
                        "direction": direction,
                        "session_bucket": context,
                        "description": " AND ".join(f"{x} {op} {v}" for x, op, v in conditions)
                        + f"; detection session {context}",
                        "conditions": [
                            {"feature": x, "op": op, "value": v} for x, op, v in conditions
                        ],
                    }
                )
    return result


def match_definition(features: pd.DataFrame, definition: dict) -> np.ndarray:
    """Interpret a fixed conjunction without eval or future data."""
    operators = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
        "==": operator.eq,
    }
    hits = features.valid.to_numpy(dtype=bool).copy()
    context = definition["session_bucket"]
    if context != "all":
        if context not in ("open", "midday", "late", "outside_rth"):
            raise ValueError("unknown session bucket")
        hits &= features.session_bucket.to_numpy() == context
    for condition in definition["conditions"]:
        op = condition["op"]
        if op not in operators:
            raise ValueError("unsupported condition operator")
        values = features[condition["feature"]].to_numpy()
        hits &= np.isfinite(values) & operators[op](values, condition["value"])
    return hits
