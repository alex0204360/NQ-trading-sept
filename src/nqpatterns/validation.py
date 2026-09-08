"""Preregistered screening, deterministic selection and descriptive summaries.

Outcome codes are 0=no-clear, 1=up, 2=down and 3=ambiguous. Callers must
supply only the common 45-minute-complete, causally thinned event population.
These helpers never inspect the holdout or relax the protocol's gates.
"""

import math
from itertools import product

import numpy as np
import pandas as pd


def parameter_grid():
    """All 540 allowed magnitude, horizon, direction hypotheses per signature."""
    return product((1, 2, 4, 8, 16, 32), range(1, 46), ("up", "down"))


def point_estimates(
    candidate, candidate_strata, controls, control_strata, direction, min_controls=30
):
    """Vectorized candidate-weighted control probabilities within one fold.

    Integer stratum codes must share the same dictionary for both populations.
    Rates are descriptive even when an occupied stratum has too few controls;
    any occupied stratum with zero controls makes matched rates undefined.
    """
    candidate, controls = np.asarray(candidate), np.asarray(controls)
    cs, bs = np.asarray(candidate_strata), np.asarray(control_strata)
    if direction not in (1, 2) or isinstance(direction, bool):
        raise ValueError("direction must be code 1 or 2")
    if not isinstance(min_controls, int) or min_controls < 1:
        raise ValueError("min_controls must be a positive integer")
    for codes, strata in ((candidate, cs), (controls, bs)):
        if codes.ndim != 1 or strata.shape != codes.shape:
            raise ValueError("outcomes and strata require aligned one-dimensional arrays")
        if not np.isin(codes, [0, 1, 2, 3]).all():
            raise ValueError("only eligible outcome codes 0 through 3 are allowed")
        if strata.size and (not np.issubdtype(strata.dtype, np.integer) or (strata < 0).any()):
            raise ValueError("strata must be nonnegative integer codes")
    n = len(candidate)
    size = max(int(cs.max()) if cs.size else -1, int(bs.max()) if bs.size else -1) + 1
    cn = np.bincount(cs.astype(int), minlength=size)
    bn = np.bincount(bs.astype(int), minlength=size)
    bf = np.bincount(bs.astype(int), weights=controls == direction, minlength=size)
    ba = np.bincount(bs.astype(int), weights=controls == 3, minlength=size)
    occupied = cn > 0
    support = bool(n and np.all(bn[occupied] >= min_controls))
    defined = bool(n and np.all(bn[occupied] > 0))
    favorable = int(np.count_nonzero(candidate == direction))
    resolved = int(np.count_nonzero((candidate == 1) | (candidate == 2)))
    hit = favorable / n if n else None
    baseline = float(np.sum(cn[occupied] * bf[occupied] / bn[occupied]) / n) if defined else None
    conservative = (
        float(np.sum(cn[occupied] * (bf[occupied] + ba[occupied]) / bn[occupied]) / n)
        if defined
        else None
    )
    return {
        "sample_size": n,
        "favorable_count": favorable,
        "unfavorable_count": resolved - favorable,
        "ambiguous_count": int(np.count_nonzero(candidate == 3)),
        "no_clear_count": int(np.count_nonzero(candidate == 0)),
        "control_sample_size": len(controls),
        "baseline_support": support,
        "hit_rate": hit,
        "resolved_hit_rate": favorable / resolved if resolved else None,
        "baseline_hit_rate": baseline,
        "conservative_baseline_hit_rate": conservative,
        "lift": hit - baseline if defined else None,
        "conservative_lift": hit - conservative if defined else None,
    }


def _finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def _at_least(value, minimum):
    return _finite(value) and value >= minimum


def screen_reasons(row):
    """Cheap exploratory gates before the finite bootstrap shortlist."""
    reasons = []
    if not _at_least(row.get("sample_size"), 1000):
        reasons.append("sample_size")
    if not _at_least(row.get("conservative_lift"), 0.05):
        reasons.append("minimum_edge")
    folds = row.get("folds", row.get("period_results", []))
    if len(folds) != 4:
        reasons.append("four_folds_required")
    if any(not _at_least(f.get("sample_size"), 100) for f in folds):
        reasons.append("fold_sample_size")
    if any(not _at_least(f.get("event_dates"), 20) for f in folds):
        reasons.append("fold_event_dates")
    if any(not f.get("baseline_support", False) for f in folds):
        reasons.append("baseline_support")
    lifts = [f.get("conservative_lift") for f in folds]
    if sum(_finite(v) and v > 0 for v in lifts) < 3 or any(
        not _finite(v) or v < -0.02 for v in lifts
    ):
        reasons.append("fold_instability")
    return reasons


def _interval(value):
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) == 2
        and all(_finite(v) for v in value)
        and value[0] <= value[1]
    )


def _inference_reasons(row):
    reasons = []
    if row.get("status") != "ok":
        reasons.append("inference_failed")
    hit, lift = row.get("confidence_interval"), row.get("lift_confidence_interval")
    if not _interval(hit) or hit[0] < 0 or hit[1] > 1 or hit[1] - hit[0] > 0.10 + 1e-12:
        reasons.append("hit_rate_precision")
    if not _interval(lift) or lift[1] - lift[0] > 0.15 + 1e-12:
        reasons.append("lift_precision")
    if not _interval(lift) or lift[0] <= 0:
        reasons.append("interval_exclusion")
    blocks = row.get("block_results", [])
    if (
        (
            len(blocks) != 2
            or any(
                not _interval(b.get("lift_confidence_interval"))
                or b["lift_confidence_interval"][0] <= 0
                or b.get("status", "ok") != "ok"
                for b in blocks
            )
        )
        or any("block_length" in b for b in blocks)
        and {b.get("block_length") for b in blocks} != {5, 20}
    ):
        reasons.append("block_sensitivity")
    return reasons


def training_reasons(row):
    """Every exploratory support, stability and uncertainty gate."""
    return screen_reasons(row) + _inference_reasons(row)


def holdout_reasons(row):
    """Terminal holdout gate; caller must Holm-adjust all frozen submissions."""
    reasons = []
    for name, minimum in (("sample_size", 1000), ("event_dates", 100), ("iso_weeks", 40)):
        if not _at_least(row.get(name), minimum):
            reasons.append(name)
    if not _at_least(row.get("sample_size"), 30):
        reasons.append("low_confidence_n_below_30")
    if not row.get("baseline_support", False):
        reasons.append("baseline_support")
    if not _at_least(row.get("conservative_lift"), 0.05):
        reasons.append("minimum_edge")
    p = row.get("adjusted_p_value")
    if not _finite(p) or not 0 <= p <= 0.05:
        reasons.append("holm_significance")
    return reasons + _inference_reasons(row)


def choose_shortlist(rows, per_family=12):
    """One configuration per signature, up to 12 signatures per family."""
    if not isinstance(per_family, int) or not 1 <= per_family <= 12:
        raise ValueError("per_family must be between 1 and the protocol ceiling 12")
    eligible = [
        r for r in rows if r.get("screen_reasons") == [] and _finite(r.get("conservative_lift"))
    ]

    def key(r):
        resolution = r.get("mean_favorable_resolution")
        return (
            -r["conservative_lift"],
            -r["sample_size"],
            resolution if _finite(resolution) else math.inf,
            r["pattern_id"],
        )

    selected, seen, families = [], set(), {}
    for row in sorted(eligible, key=key):
        signature, family = row["signature_id"], row["family"]
        if signature in seen or families.get(family, 0) >= per_family:
            continue
        selected.append(row)
        seen.add(signature)
        families[family] = families.get(family, 0) + 1
    return selected


def resolution_summary(codes, resolution_bars, horizon, direction):
    """Unambiguous time distributions and cumulative incidence over all events."""
    if (
        not isinstance(horizon, (int, np.integer))
        or isinstance(horizon, bool)
        or not 1 <= horizon <= 45
    ):
        raise ValueError("horizon must be an integer between 1 and 45")
    if direction not in (1, 2):
        raise ValueError("direction must be code 1 or 2")
    codes, times = np.asarray(codes), np.asarray(resolution_bars, dtype=float)
    if codes.ndim != 1 or codes.shape != times.shape or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError("aligned eligible outcome and resolution arrays are required")
    resolved = (codes == 1) | (codes == 2)
    if np.any(
        resolved
        & (~np.isfinite(times) | (times < 1) | (times > horizon) | (times != np.floor(times)))
    ):
        raise ValueError("resolved outcomes require integer resolution times within the horizon")

    def distribution(mask):
        values = times[mask]
        hist = {str(i): int(np.count_nonzero(values == i)) for i in range(1, horizon + 1)}
        result = {"count": len(values), "histogram": hist}
        for name, value in [
            ("mean", np.mean),
            ("std", np.std),
            ("median", np.median),
            ("min", np.min),
            ("max", np.max),
        ]:
            result[name] = float(value(values)) if len(values) else None
        for name, q in [("p25", 0.25), ("p75", 0.75), ("p90", 0.90)]:
            result[name] = float(np.quantile(values, q)) if len(values) else None
        return result

    return {
        "all_resolved": distribution(resolved),
        "favorable": distribution(codes == direction),
        "cumulative_resolved_fraction": {
            str(i): float(np.count_nonzero(resolved & (times <= i)) / len(codes))
            if len(codes)
            else None
            for i in range(1, horizon + 1)
        },
    }


def stratum_codes(timestamps, relative_range, cutpoints):
    """Encode ET year-quarter, six-hour clock bucket and supplied past tertiles.

    Compute on the union of candidate and control timestamps before subsetting,
    since IDs are local to this call. Naive times and nonfinite ranges are errors.
    """
    times = pd.DatetimeIndex(timestamps)
    values = np.asarray(relative_range, dtype=float)
    cuts = np.asarray(cutpoints, dtype=float)
    if times.tz is None or times.hasnans:
        raise ValueError("timestamps must be aware and nonmissing")
    if values.shape != (len(times),) or not np.isfinite(values).all():
        raise ValueError("relative ranges must be aligned and finite")
    if cuts.shape != (2,) or not np.isfinite(cuts).all() or cuts[0] > cuts[1]:
        raise ValueError("two ordered finite past cutpoints are required")
    et = times.tz_convert("America/New_York")
    buckets = ("00-06", "06-12", "12-18", "18-24")
    tertiles = np.searchsorted(cuts, values, side="right")
    labels = np.array(
        [
            f"{y}Q{q}|{buckets[h // 6]}|{v}"
            for y, q, h, v in zip(et.year, et.quarter, et.hour, tertiles)
        ],
        dtype=str,
    )
    names, codes = np.unique(labels, return_inverse=True)
    return codes, names.tolist()
