"""Matched first-passage estimates and joint moving-block uncertainty.

The caller owns outcome construction, causal calibration, and the frozen trading
calendar. Every input row is an already eligible, scheduled event. A favorable
indicator means a confirmed first touch in the tested direction; ambiguities are
separate and never favorable to the candidate. Controls use the same outcome rule.

No event is independently resampled. Candidate and control counts for a trading
date always travel together, including shared events. Day aggregation is a
sufficient statistic for these probabilities and keeps all observations without
changing their weights. Complete zero-event dates remain in each explicit frame.

Research calls use 9,999 replicates for discovery and 99,999 for holdout, both
5-day and 20-day blocks. Smaller counts are exposed for tests only. Percentile
intervals are marginal, approximate intervals; this module does not claim that
adaptive discovery results are confirmatory or economically profitable.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import ceil

import numpy as np
import pandas as pd

ROOT_SEED = 20260906
EVENT_COLUMNS = ("day", "stratum", "favorable", "ambiguous")
METRICS = (
    "hit_rate",
    "baseline_hit_rate",
    "conservative_baseline_hit_rate",
    "lift",
    "conservative_lift",
)


@dataclass
class PeriodObservations:
    """Events plus an explicit calendar frame, normally one WF fold or holdout.

    ``day`` and ``days`` identify session dates, not necessarily UTC civil dates.
    ``stratum`` is a string identifier whose past-only calibration is performed
    upstream. ``favorable`` and ``ambiguous`` must be disjoint binary indicators.
    Period IDs and their frames must be stable across hypotheses so day draws are
    shared. Disjoint event intervals may split one session date across adjacent
    period frames. Resampling is separate by period; dependence across this
    partial boundary session remains an approximation.
    """

    candidate: pd.DataFrame
    control: pd.DataFrame
    days: Sequence
    period_id: str


@dataclass
class _CompiledPeriod:
    period_id: str
    days: list[str]
    strata: list[str]
    counts: np.ndarray
    event_days: list[str]


def _date(value) -> str:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing date")
        return stamp.date().isoformat()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("All event/frame days must be valid session dates") from exc


def _events(frame: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not set(EVENT_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Event data require columns {EVENT_COLUMNS}")
    result = frame.loc[:, EVENT_COLUMNS].copy()
    if result.isna().any().any():
        raise ValueError("Event fields cannot contain missing values")
    result["day"] = result["day"].map(_date)
    if not result["day"].isin(days).all():
        raise ValueError("An event day is outside its explicit calendar frame")
    if not result["stratum"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Stratum identifiers must be nonempty strings")
    for name in ("favorable", "ambiguous"):
        if not result[name].isin([0, 1]).all():
            raise ValueError("Outcome indicators must be binary")
        result[name] = result[name].astype(np.int64)
    if (result["favorable"] + result["ambiguous"] > 1).any():
        raise ValueError("Favorable and ambiguous indicators must be disjoint")
    return result


def _compile(periods: Sequence[PeriodObservations]) -> list[_CompiledPeriod]:
    if not periods:
        raise ValueError("At least one observation period is required")
    compiled, seen_ids = [], set()
    for period in periods:
        if not isinstance(period.period_id, str) or not period.period_id:
            raise ValueError("Each period needs a stable nonempty period_id")
        if period.period_id in seen_ids:
            raise ValueError("Period IDs must be unique")
        seen_ids.add(period.period_id)
        days = [_date(day) for day in period.days]
        if not days or days != sorted(set(days)):
            raise ValueError("Calendar days must be nonempty, unique, and ordered")
        candidate, control = _events(period.candidate, days), _events(period.control, days)
        strata = sorted(set(candidate["stratum"]) | set(control["stratum"]))
        day_ids = {day: i for i, day in enumerate(days)}
        stratum_ids = {stratum: i for i, stratum in enumerate(strata)}
        counts = np.zeros((len(days), len(strata), 6), dtype=np.int64)
        for frame, offset in ((candidate, 0), (control, 3)):
            if frame.empty:
                continue
            day_index = frame["day"].map(day_ids).to_numpy(dtype=np.int64)
            stratum_index = frame["stratum"].map(stratum_ids).to_numpy(dtype=np.int64)
            np.add.at(counts[:, :, offset], (day_index, stratum_index), 1)
            for addition, name in ((1, "favorable"), (2, "ambiguous")):
                np.add.at(
                    counts[:, :, offset + addition],
                    (day_index, stratum_index),
                    frame[name].to_numpy(dtype=np.int64),
                )
        compiled.append(
            _CompiledPeriod(period.period_id, days, strata, counts, sorted(set(candidate["day"])))
        )
    return compiled


def _rates(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return n/favorable/weighted ordinary/conservative numerators and validity."""
    candidate_n, control_n = totals[..., 0], totals[..., 3]
    undefined = np.any((candidate_n > 0) & (control_n == 0), axis=-1)
    ordinary = np.divide(
        totals[..., 4], control_n, out=np.zeros_like(control_n, dtype=float), where=control_n > 0
    )
    conservative = np.divide(
        totals[..., 4] + totals[..., 5],
        control_n,
        out=np.zeros_like(control_n, dtype=float),
        where=control_n > 0,
    )
    numerators = np.stack(
        (
            candidate_n.sum(axis=-1),
            totals[..., 1].sum(axis=-1),
            (candidate_n * ordinary).sum(axis=-1),
            (candidate_n * conservative).sum(axis=-1),
        ),
        axis=-1,
    )
    return numerators, undefined


def _probabilities(numerators: np.ndarray) -> np.ndarray:
    probabilities = numerators[..., 1:] / numerators[..., :1]
    return np.concatenate(
        (
            probabilities,
            probabilities[..., :1] - probabilities[..., 1:2],
            probabilities[..., :1] - probabilities[..., 2:3],
        ),
        axis=-1,
    )


def _iso_weeks(days: Sequence[str]) -> int:
    return len({tuple(pd.Timestamp(day).isocalendar()[:2]) for day in days})


def _point(compiled: list[_CompiledPeriod], min_controls: int) -> dict:
    if not isinstance(min_controls, int) or min_controls < 1:
        raise ValueError("min_controls must be a positive integer")
    period_results, pooled, reasons = [], np.zeros(4), []
    pooled_undefined = False
    all_event_days = []
    for period in compiled:
        totals = period.counts.sum(axis=0)
        numerators, undefined = _rates(totals)
        n = int(numerators[0])
        support = not bool(np.any((totals[:, 0] > 0) & (totals[:, 3] < min_controls)))
        rates = _probabilities(numerators) if n and not undefined else [None] * len(METRICS)
        period_result = {
            "period_id": period.period_id,
            "sample_size": n,
            "favorable_count": int(totals[:, 1].sum()),
            "ambiguous_count": int(totals[:, 2].sum()),
            "control_sample_size": int(totals[:, 3].sum()),
            "event_dates": len(period.event_days),
            "iso_weeks": _iso_weeks(period.event_days),
            "frame_dates": len(period.days),
            "baseline_support": support,
            "stratum_counts": [
                {
                    "stratum": name,
                    "sample_size": int(totals[i, 0]),
                    "control_size": int(totals[i, 3]),
                }
                for i, name in enumerate(period.strata)
                if totals[i, 0] > 0
            ],
            **{
                name: float(value) if value is not None else None
                for name, value in zip(METRICS, rates)
            },
        }
        period_results.append(period_result)
        if not support:
            reasons.append(
                f"{period.period_id}: occupied stratum has fewer than {min_controls} controls"
            )
        pooled += numerators
        pooled_undefined |= bool(undefined)
        all_event_days.extend(period.event_days)
    n = int(pooled[0])
    if not n:
        status = "no_candidate_events"
        reasons.append("No eligible candidate events")
    else:
        status = "baseline_support_failed" if reasons else "ok"
    rates = _probabilities(pooled) if n and not pooled_undefined else [None] * len(METRICS)
    return {
        "status": status,
        "reasons": reasons,
        "sample_size": n,
        "event_dates": len(set(all_event_days)),
        "iso_weeks": _iso_weeks(all_event_days),
        "baseline_support": all(p["baseline_support"] for p in period_results),
        **{
            name: float(value) if value is not None else None for name, value in zip(METRICS, rates)
        },
        "period_results": period_results,
    }


def matched_estimate(periods: Sequence[PeriodObservations], min_controls: int = 30) -> dict:
    """Match within period/stratum, then pool weighted by candidate event counts.

    Observed support below ``min_controls`` fails the baseline gate even when a
    descriptive rate can be calculated. An occupied stratum with no controls has
    no matched rate. No-clear events remain in the candidate denominator.
    """
    return _point(_compile(periods), min_controls)


def moving_block_indices(
    n_days: int,
    block_length: int,
    n_resamples: int,
    period_id: str,
    seed: int = ROOT_SEED,
) -> np.ndarray:
    """Shared day draws: admissible consecutive blocks, no wrap, final truncation.

    A SHA-256 digest supplies stable period entropy instead of Python's randomized
    hash. Randomness depends on the period identity/frame length, block length,
    and root seed, never on a hypothesis or its outcomes. Keep NumPy pinned when
    reproducing runs. The replicate-count prefix is stable.
    """
    for value in (n_days, block_length, n_resamples):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(
                "Day count, block length, and resample count must be positive integers"
            )
    if block_length > n_days:
        raise ValueError("Period has fewer days than the requested block length")
    if not isinstance(period_id, str) or not period_id or not isinstance(seed, int) or seed < 0:
        raise ValueError("A stable period ID and nonnegative integer seed are required")
    entropy = np.frombuffer(sha256(period_id.encode("utf-8")).digest(), dtype="<u4").tolist()
    rng = np.random.default_rng([seed, n_days, block_length, *entropy])
    starts = rng.integers(
        0,
        n_days - block_length + 1,
        size=(n_resamples, ceil(n_days / block_length)),
        dtype=np.int32,
    )
    indices = (starts[..., None] + np.arange(block_length, dtype=np.int32)).reshape(n_resamples, -1)
    return indices[:, :n_days].copy()


def _draw_numerators(
    period: _CompiledPeriod,
    indices: np.ndarray,
    block_length: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Block prefix sums reduce work while exactly preserving shared day draws."""
    n_resamples, n_days = indices.shape
    totals = np.zeros((n_resamples, 4))
    undefined = np.zeros(n_resamples, dtype=bool)
    prefix = np.concatenate(
        (np.zeros_like(period.counts[:1]), np.cumsum(period.counts, axis=0)), axis=0
    )
    offsets = np.arange(0, n_days, block_length)
    lengths = np.minimum(block_length, n_days - offsets)
    for start in range(0, n_resamples, chunk_size):
        end = min(start + chunk_size, n_resamples)
        block_starts = indices[start:end, offsets]
        sampled = (prefix[block_starts + lengths] - prefix[block_starts]).sum(axis=1)
        totals[start:end], undefined[start:end] = _rates(sampled)
    return totals, undefined


def _one_block(
    compiled: list[_CompiledPeriod],
    point: dict,
    block_length: int,
    n_resamples: int,
    seed: int,
    chunk_size: int,
) -> dict:
    result = {
        "block_length": block_length,
        "n_resamples": n_resamples,
        "status": "inference_failed",
        "reasons": [],
        "undefined_replicates": 0,
        "confidence_interval": None,
        "lift_confidence_interval": None,
        "p_value": 1.0,
    }
    numerators = np.zeros((n_resamples, 4))
    undefined = np.zeros(n_resamples, dtype=bool)
    for period in compiled:
        if len(period.days) < block_length:
            result["reasons"].append(
                f"{period.period_id}: too few frame days for {block_length}-day block"
            )
            return result
        indices = moving_block_indices(
            len(period.days), block_length, n_resamples, period.period_id, seed
        )
        contribution, bad = _draw_numerators(period, indices, block_length, chunk_size)
        numerators += contribution
        undefined |= bad
    undefined |= numerators[:, 0] == 0
    result["undefined_replicates"] = int(undefined.sum())
    if undefined.any():
        result["reasons"].append(
            "undefined candidate or occupied-stratum control support in a replicate"
        )
        return result
    probabilities = _probabilities(numerators)
    for metric in ("hit_rate", "conservative_baseline_hit_rate", "conservative_lift"):
        values = probabilities[:, METRICS.index(metric)]
        if np.ptp(values) <= 1e-12:
            result["reasons"].append(f"degenerate {metric} bootstrap distribution")
    if result["reasons"]:
        return result
    intervals = np.quantile(probabilities, [0.025, 0.975], axis=0, method="linear")
    observed = point["conservative_lift"]
    centered = probabilities[:, METRICS.index("conservative_lift")] - observed
    result.update(
        status="ok",
        confidence_interval=intervals[:, METRICS.index("hit_rate")].tolist(),
        lift_confidence_interval=intervals[:, METRICS.index("conservative_lift")].tolist(),
        metric_intervals={name: intervals[:, i].tolist() for i, name in enumerate(METRICS)},
        p_value=float((1 + np.count_nonzero(centered >= observed)) / (n_resamples + 1)),
    )
    return result


def block_bootstrap(
    periods: Sequence[PeriodObservations],
    n_resamples: int = 9999,
    block_lengths: Sequence[int] = (5, 20),
    seed: int = ROOT_SEED,
    min_controls: int = 30,
    chunk_size: int = 256,
) -> dict:
    """Joint per-period moving-block bootstrap with conservative lift inference.

    The original frame's observed occupied strata require at least 30 controls.
    Every resample recomputes weights; a zero denominator for any then-occupied
    stratum fails the whole inference rather than dropping that replicate. Empty
    candidate strata in a resample contribute no weight. Constant primary
    probability, conservative baseline, or lift distributions fail explicitly.

    The two confidence intervals returned at top level are the envelope across
    block lengths. The top-level p-value is their maximum centered, one-sided,
    plus-one p-value. Failure always carries p=1 for later frozen-family Holm.
    """
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    lengths = tuple(block_lengths)
    if (
        not lengths
        or any(not isinstance(x, int) or x < 1 for x in lengths)
        or len(set(lengths)) != len(lengths)
    ):
        raise ValueError("Block lengths must be distinct positive integers")
    compiled = _compile(periods)
    point = _point(compiled, min_controls)
    result = {
        "status": point["status"],
        "reasons": point["reasons"].copy(),
        "point_estimate": point,
        "confidence_interval": None,
        "lift_confidence_interval": None,
        "p_value": 1.0,
        "block_results": [],
        "method": "joint moving-block percentile envelope; marginal 95% intervals",
        "seed": seed,
        "n_resamples_per_block": n_resamples,
    }
    if point["status"] != "ok":
        return result
    for length in lengths:
        block = _one_block(compiled, point, length, n_resamples, seed, chunk_size)
        result["block_results"].append(block)
        result["reasons"].extend(f"{length}-day block: {reason}" for reason in block["reasons"])
    if result["reasons"]:
        result["status"] = "inference_failed"
        return result
    result["status"] = "ok"
    for key in ("confidence_interval", "lift_confidence_interval"):
        result[key] = [
            min(block[key][0] for block in result["block_results"]),
            max(block[key][1] for block in result["block_results"]),
        ]
    result["p_value"] = max(block["p_value"] for block in result["block_results"])
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment in original order; include every frozen failure."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be a one-dimensional sequence in [0, 1]")
    if not len(values):
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    original_order = np.empty_like(adjusted)
    original_order[order] = adjusted
    return original_order.tolist()
