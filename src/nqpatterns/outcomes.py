"""First-passage labels on completed one-minute bars, without intrabar guesses.

The detection bar is index ``t`` and the reference entry is ``open[t + 1]``.
Availability is retrospective and must only be applied *after* causal thinning.
This module neither imputes missing bars nor uses outcomes to schedule detections.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

MAX_HORIZON = 45
TICK_SIZE = 0.25
_MINUTE_NS = 60_000_000_000
_OUTCOMES = ("up", "down", "no_clear_outcome", "ambiguous_both", "censored_data")


class OutOfScopeHorizon(ValueError):
    """A scope rejection, distinct from negative evidence about a pattern."""

    status = "rejected_out_of_scope"
    reason = "rejected — out of scope: holding window exceeds 45 bars"

    def __init__(self):
        super().__init__(self.reason)


def _horizon(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("horizon must be an integer from 1 through 45")
    if value > MAX_HORIZON:
        raise OutOfScopeHorizon()
    if value < 1:
        raise ValueError("horizon must be an integer from 1 through 45")
    return int(value)


def _magnitude(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("magnitude_points must be a positive tick-aligned number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("magnitude_points must be positive and finite")
    ticks = value / TICK_SIZE
    if not np.isclose(ticks, np.rint(ticks), rtol=0, atol=1e-9):
        raise ValueError("magnitude_points must align with the 0.25-point tick")
    return value


def _timestamps(values):
    try:
        timestamps = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be timezone-aware and strictly increasing") from exc
    if timestamps.tz is None or timestamps.hasnans:
        raise ValueError("timestamps must be timezone-aware and contain no missing times")
    timestamps = timestamps.tz_convert("UTC").as_unit("ns")
    if len(timestamps) > 1 and np.any(np.diff(timestamps.asi8) <= 0):
        raise ValueError("timestamps must be strictly increasing with no duplicates")
    return timestamps


def _bar_arrays(bars):
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a canonical OHLCV DataFrame")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing canonical bar columns: {sorted(missing)}")
    timestamps = _timestamps(bars["timestamp"])
    arrays = []
    for name in ("open", "high", "low", "close", "volume"):
        if not pd.api.types.is_numeric_dtype(bars[name].dtype):
            raise TypeError(f"{name} must be numeric; normalize and audit source bars first")
        value = bars[name].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values; audit source bars first")
        arrays.append(value)
    opening, high, low, close, volume = arrays
    if any(np.any(value <= 0) for value in (opening, high, low, close)):
        raise ValueError("OHLC prices must be positive; audit source bars first")
    if np.any(high < np.maximum(opening, close)) or np.any(low > np.minimum(opening, close)):
        raise ValueError("OHLC bounds are inconsistent; audit source bars first")
    if np.any(volume < 0):
        raise ValueError("volume cannot be negative")
    boundaries = np.diff(timestamps.asi8) != _MINUTE_NS
    for name in ("contract", "segment_id"):
        if name in bars:
            if bars[name].isna().any():
                raise ValueError(f"{name} cannot contain unresolved missing values")
            values = bars[name].to_numpy()
            boundaries |= values[1:] != values[:-1]
    return timestamps, opening, high, low, boundaries


@dataclass(frozen=True)
class FirstPassage:
    """Reusable immutable first touches for every scanned horizon.

    ``first_touch_bars`` includes ambiguous double touches. Empty directions and
    NaN first-touch times mean no touch in the available scanned bars. The arrays
    do not censor early touches at later gaps: ``labels_from_passage`` performs
    that operation separately for each requested complete horizon.
    """

    index: pd.Index
    first_direction: np.ndarray
    first_touch_bars: np.ndarray
    available_bars: np.ndarray
    eligible_45: np.ndarray
    max_horizon: int


def first_passage(bars, magnitude_points, max_horizon=MAX_HORIZON):
    """Scan at most 45 future bars in vectorized offsets for one magnitude.

    The common 45-bar availability mask is computed even when ``max_horizon``
    requests a shorter scan. Availability stops at a gap, contract/segment change,
    or the end of the supplied frame; it never depends on barrier outcomes.
    """
    max_horizon = _horizon(max_horizon)
    magnitude_points = _magnitude(magnitude_points)
    _, opening, high, low, boundaries = _bar_arrays(bars)
    count = len(bars)
    first_direction = np.full(count, "", dtype=object)
    first_touch_bars = np.full(count, np.nan)
    available_bars = np.zeros(count, dtype=np.uint8)
    if count:
        starts = np.r_[0, np.flatnonzero(boundaries) + 1]
        ends = np.r_[starts[1:] - 1, count - 1]
        remaining = np.repeat(ends, ends - starts + 1) - np.arange(count)
        available_bars = np.minimum(remaining, MAX_HORIZON).astype(np.uint8)
        entry = np.r_[opening[1:], np.nan]
        upper = entry + magnitude_points
        lower = entry - magnitude_points
        unresolved = np.ones(count, dtype=bool)
        for offset in range(1, min(max_horizon, count - 1) + 1):
            positions = np.flatnonzero(unresolved[:-offset] & (available_bars[:-offset] >= offset))
            if not len(positions):
                break
            future = positions + offset
            open_up = opening[future] >= upper[positions]
            open_down = opening[future] <= lower[positions]
            touched_up = high[future] >= upper[positions]
            touched_down = low[future] <= lower[positions]
            touched = open_up | open_down | touched_up | touched_down
            resolved_positions = positions[touched]
            direction_up = open_up | (~open_down & touched_up & ~touched_down)
            direction_down = open_down | (~open_up & touched_down & ~touched_up)
            first_direction[resolved_positions] = "ambiguous_both"
            first_direction[positions[direction_up]] = "up"
            first_direction[positions[direction_down]] = "down"
            first_touch_bars[resolved_positions] = offset
            unresolved[resolved_positions] = False
    eligible_45 = available_bars == MAX_HORIZON
    for array in (first_direction, first_touch_bars, available_bars, eligible_45):
        array.flags.writeable = False
    return FirstPassage(
        bars.index.copy(),
        first_direction,
        first_touch_bars,
        available_bars,
        eligible_45,
        max_horizon,
    )


def labels_from_passage(passage, horizon):
    """Materialize a horizon's labels; incomplete requested windows are censored."""
    horizon = _horizon(horizon)
    if not isinstance(passage, FirstPassage):
        raise TypeError("passage must be a FirstPassage result")
    if horizon > passage.max_horizon:
        raise ValueError("requested horizon exceeds the scanned max_horizon")
    eligible = passage.available_bars >= horizon
    touched = eligible & (passage.first_touch_bars <= horizon)
    outcome = np.full(len(passage.index), "censored_data", dtype=object)
    outcome[eligible] = "no_clear_outcome"
    outcome[touched] = passage.first_direction[touched]
    unambiguous = touched & ((outcome == "up") | (outcome == "down"))
    resolution = np.where(unambiguous, passage.first_touch_bars, np.nan)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "resolution_bars": pd.array(resolution, dtype="Float64"),
            "eligible_45": passage.eligible_45.copy(),
            "eligible_horizon": eligible,
        },
        index=passage.index,
    )


def label_outcomes(bars, magnitude_points, horizon):
    """Convenience wrapper for one window; use a reusable passage for a grid."""
    horizon = _horizon(horizon)
    return labels_from_passage(first_passage(bars, magnitude_points, max_horizon=horizon), horizon)


def nonoverlap_indices(timestamps, raw_mask, minutes=MAX_HORIZON):
    """Keep chronological detections at least ``minutes`` apart in elapsed time.

    Call this before any future-availability filter. Gaps do not reset the clock,
    and an otherwise valid signal with a censored outcome consumes its cooldown.
    The returned integers are positions, independent of pandas index labels.
    """
    if isinstance(minutes, (bool, np.bool_)) or not isinstance(minutes, Integral):
        raise TypeError("minutes must be a positive integer")
    if minutes < 1:
        raise ValueError("minutes must be a positive integer")
    timestamps = _timestamps(timestamps)
    mask = np.asarray(raw_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_ or len(mask) != len(timestamps):
        raise ValueError("raw_mask must be a one-dimensional boolean mask matching timestamps")
    raw = np.flatnonzero(mask)
    kept = []
    next_allowed = None
    cooldown = int(minutes) * _MINUTE_NS
    timestamp_ns = timestamps.asi8
    for position in raw:
        current = int(timestamp_ns[position])
        if next_allowed is None or current >= next_allowed:
            kept.append(position)
            next_allowed = current + cooldown
    return np.asarray(kept, dtype=np.int64)


def _distribution(times, horizon):
    histogram = {bar: int(np.count_nonzero(times == bar)) for bar in range(1, horizon + 1)}
    result = {name: None for name in ("mean", "std", "median", "p25", "p75", "p90", "min", "max")}
    result.update(count=len(times), histogram=histogram)
    if len(times):
        result.update(
            mean=float(np.mean(times)),
            std=float(np.std(times, ddof=1)) if len(times) > 1 else None,
            median=float(np.median(times)),
            p25=float(np.percentile(times, 25)),
            p75=float(np.percentile(times, 75)),
            p90=float(np.percentile(times, 90)),
            min=float(np.min(times)),
            max=float(np.max(times)),
        )
    return result


def resolution_time_summary(labels, horizon, favorable_direction):
    """Describe unambiguous resolution times and incidence among eligible labels.

    Callers select their event population first (including the common-45 filter
    for primary scoring). Censored rows are counted separately and excluded from
    incidence denominators. Unresolved rows never become zero-duration events.
    """
    horizon = _horizon(horizon)
    if favorable_direction not in {"up", "down"}:
        raise ValueError("favorable_direction must be up or down")
    if not {"outcome", "resolution_bars"}.issubset(labels.columns):
        raise ValueError("labels require outcome and resolution_bars columns")
    outcome = labels["outcome"].to_numpy()
    if not np.isin(outcome, _OUTCOMES).all():
        raise ValueError("labels contain an unknown outcome")
    times = labels["resolution_bars"].to_numpy(dtype=float, na_value=np.nan)
    resolved = (outcome == "up") | (outcome == "down")
    valid_times = (
        np.isfinite(times) & (times >= 1) & (times <= horizon) & (times == np.floor(times))
    )
    if np.any(resolved & ~valid_times) or np.any(~resolved & ~np.isnan(times)):
        raise ValueError(
            "resolution times must exist only for resolved outcomes within the horizon"
        )
    counts = {name: int(np.count_nonzero(outcome == name)) for name in _OUTCOMES}
    eligible_count = len(outcome) - counts["censored_data"]
    all_resolved = _distribution(times[resolved], horizon)
    favorable = _distribution(times[outcome == favorable_direction], horizon)

    def incidence(histogram):
        cumulative = 0
        result = {}
        for bar, frequency in histogram.items():
            cumulative += frequency
            result[bar] = cumulative / eligible_count if eligible_count else None
        return result

    return {
        "eligible_count": int(eligible_count),
        "censored_count": counts["censored_data"],
        "outcome_counts": counts,
        "all_resolved": all_resolved,
        "favorable": favorable,
        "cumulative_resolved_fraction": incidence(all_resolved["histogram"]),
        "cumulative_favorable_fraction": incidence(favorable["histogram"]),
    }
