"""Completed-bar features with an explicit finite, causal history.

Every valid row depends on at most the current bar and 60 previous completed
bars. Missing minutes, contract changes, and audited segment changes restart
the history. No publisher-provided indicator is consumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "body_fraction",
    "upper_wick_fraction",
    "lower_wick_fraction",
    "lag1_return_to_range",
    "range_to_atr14",
    "atr14",
    "atr14_fraction",
    "price_z20",
    "trend20_60",
    "volume_z20",
    "range_z20",
    "prior_high20",
    "prior_low20",
    "prior_high60",
    "prior_low60",
    "position60",
    "breakout_up20_atr",
    "breakout_down20_atr",
    "breakout_up60_atr",
    "breakout_down60_atr",
)
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
WARMUP_PRIOR_BARS = 60


def _validate_bars(bars: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"Missing canonical bar columns: {sorted(missing)}")
    try:
        timestamps = pd.DatetimeIndex(bars["timestamp"])
    except (ValueError, TypeError) as exc:
        raise ValueError("timestamp must contain timezone-aware timestamps") from exc
    if timestamps.tz is None or timestamps.hasnans:
        raise ValueError("timestamp must be timezone-aware and nonmissing")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("timestamp must be strictly increasing without duplicates")
    values = bars[list(REQUIRED_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV must be finite")
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("OHLC must be positive and volume nonnegative")
    if len(values):
        open_, high, low, close, _ = values.T
        if (
            (high < np.maximum.reduce([open_, close, low]))
            | (low > np.minimum.reduce([open_, close, high]))
        ).any():
            raise ValueError("Inconsistent OHLC geometry")
    return timestamps.as_unit("ns")


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Zero ranges/deviations have a neutral value; missing history stays NaN."""
    result = numerator / denominator.where(denominator != 0)
    return result.mask((denominator == 0) & numerator.notna(), 0.0)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return finite-history features, preserving the caller's index.

    ``valid`` becomes true after 60 prior consecutive minutes. Incomplete
    rolling windows are NaN. A flat observed window has neutral zero ratios.
    Timestamp validation accepts any aware timezone; elapsed time is UTC.
    """
    timestamps = _validate_bars(bars)
    if bars.empty:
        result = pd.DataFrame(index=bars.index, columns=FEATURE_COLUMNS, dtype=float)
        result["valid"] = pd.Series(index=bars.index, dtype=bool)
        return result

    # Use positional series internally so non-unique source indices cannot
    # accidentally align observations belonging to different bars.
    local = bars.reset_index(drop=True)
    breaks = np.ones(len(bars), dtype=bool)
    breaks[1:] = np.diff(timestamps.asi8) != 60_000_000_000
    for column in ("contract", "segment_id"):
        if column in local:
            identifiers = local[column].astype("string").fillna("<missing>")
            breaks[1:] |= identifiers.iloc[1:].to_numpy() != identifiers.iloc[:-1].to_numpy()
    positions = np.arange(len(bars))
    age = pd.Series(positions - np.maximum.accumulate(np.where(breaks, positions, 0)))
    open_, high, low, close, volume = (
        local[column].astype(float) for column in REQUIRED_COLUMNS[1:]
    )
    observed_range = high - low
    previous_close = close.shift().mask(breaks)
    true_range = pd.concat(
        [observed_range, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)

    def rolling(values: pd.Series, window: int, method: str, *, prior: bool = False):
        source = values.shift() if prior else values
        roller = source.rolling(window, min_periods=window)
        result = roller.std(ddof=0) if method == "std" else getattr(roller, method)()
        return result.where(age >= window - 1 + int(prior))

    atr14 = rolling(true_range, 14, "mean")
    mean20 = rolling(close, 20, "mean")
    mean60 = rolling(close, 60, "mean")
    high20 = rolling(high, 20, "max", prior=True)
    low20 = rolling(low, 20, "min", prior=True)
    high60 = rolling(high, 60, "max", prior=True)
    low60 = rolling(low, 60, "min", prior=True)
    features = {
        "body_fraction": _divide(close - open_, observed_range),
        "upper_wick_fraction": _divide(
            high - pd.concat([open_, close], axis=1).max(axis=1), observed_range
        ),
        "lower_wick_fraction": _divide(
            pd.concat([open_, close], axis=1).min(axis=1) - low, observed_range
        ),
        "lag1_return_to_range": _divide(close - previous_close, observed_range),
        "range_to_atr14": _divide(observed_range, atr14),
        "atr14": atr14,
        "atr14_fraction": _divide(atr14, close),
        "price_z20": _divide(close - mean20, rolling(close, 20, "std")),
        "trend20_60": _divide(mean20 - mean60, atr14),
        "volume_z20": _divide(volume - rolling(volume, 20, "mean"), rolling(volume, 20, "std")),
        "range_z20": _divide(
            observed_range - rolling(observed_range, 20, "mean"), rolling(observed_range, 20, "std")
        ),
        "prior_high20": high20,
        "prior_low20": low20,
        "prior_high60": high60,
        "prior_low60": low60,
        "position60": _divide(close - low60, high60 - low60),
        "breakout_up20_atr": _divide(close - high20, atr14),
        "breakout_down20_atr": _divide(low20 - close, atr14),
        "breakout_up60_atr": _divide(close - high60, atr14),
        "breakout_down60_atr": _divide(low60 - close, atr14),
    }
    result = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    result["valid"] = (age >= WARMUP_PRIOR_BARS) & np.isfinite(result.to_numpy()).all(axis=1)
    result.index = bars.index
    return result
