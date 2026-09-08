"""Normalize one-minute OHLCV without hiding exclusions or provenance assumptions.

This module never fills a missing minute or chooses one side of conflicting data.
It does not establish a trading calendar or verify source timestamp conventions.
"""

from __future__ import annotations

from collections import Counter
from os import PathLike
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")
VALUE_COLUMNS = (*PRICE_COLUMNS, "volume")
MINUTE = pd.Timedelta(minutes=1)


def _timestamps(values: pd.Series, source_tz: str) -> tuple[pd.Series, dict[str, np.ndarray]]:
    """Vectorized mixed naive/offset parsing; DST diagnostic passes do not impute times."""
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    errors = {}
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        result = values.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    else:
        strings = values.astype("string")
        aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False).to_numpy(bool)
        if aware.any():
            result.loc[aware] = pd.to_datetime(
                values.loc[aware], format="mixed", utc=True, errors="coerce"
            )
        if (~aware).any():
            naive = pd.to_datetime(values.loc[~aware], format="mixed", errors="coerce")
            localized = naive.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
            # This second localization diagnoses an ambiguous time; its value is never retained.
            diagnostic = naive.dt.tz_localize(source_tz, ambiguous=True, nonexistent="NaT")
            ambiguous = naive.notna() & localized.isna() & diagnostic.notna()
            nonexistent = naive.notna() & diagnostic.isna()
            for name, mask in (
                ("ambiguous_local_time", ambiguous),
                ("nonexistent_local_time", nonexistent),
            ):
                full = np.zeros(len(values), dtype=bool)
                full[mask.index] = mask.to_numpy(bool)
                errors[name] = full
            result.loc[~aware] = localized.dt.tz_convert("UTC")
    invalid = result.isna().to_numpy()
    for mask in errors.values():
        invalid &= ~mask
    errors["invalid_timestamp"] = invalid
    return result, errors


def normalize_bars(
    source: pd.DataFrame | str | PathLike,
    *,
    timestamp_column: str = "timestamp",
    source_tz: str = "America/New_York",
    timestamp_convention: str = "start",
    tick_size: float = 0.25,
    source_timezone_verified: bool = False,
    source_convention_verified: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return a new canonical frame and an exhaustive, JSON-safe mechanical audit.

    ``source_row`` is the one-based data-row ordinal, excluding any CSV header.
    ``timestamp_convention='start'`` is an explicit caller assumption by default,
    not a claim that a publisher documented it. End timestamps shift back one
    minute. Verification flags record external evidence and never infer it from
    plausible-looking prices. An unresolved calendar remains unverified.

    Invalid bars and every conflicting duplicate are quarantined. An identical
    duplicate requires equality of *all* source fields except its timestamp.
    Extra vendor indicators are compared for duplication but are not features.
    """
    if source_tz is None:
        raise ValueError("source_tz must identify an explicit timezone")
    if not isinstance(source_tz, str):
        raise TypeError("source_tz must be a timezone name string")
    try:
        ZoneInfo(source_tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("source_tz must identify a recognized timezone") from exc
    if timestamp_convention not in {"start", "end"}:
        raise ValueError("timestamp_convention must be explicitly 'start' or 'end'")
    if not np.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("tick_size must be finite and positive")
    raw = source.copy(deep=True) if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    raw = raw.reset_index(drop=True)
    missing = {timestamp_column, *VALUE_COLUMNS} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not raw.columns.is_unique:
        raise ValueError("Source column names must be unique")
    count = len(raw)
    reasons: dict[int, list[str]] = {}

    def exclude(mask: np.ndarray | pd.Series, reason: str) -> None:
        for pos in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons.setdefault(int(pos) + 1, []).append(reason)

    timestamps, parse_errors = _timestamps(raw[timestamp_column], source_tz)
    for reason, mask in parse_errors.items():
        exclude(mask, reason)
    nonminute = timestamps.notna() & (timestamps.astype("int64") % MINUTE.value != 0)
    exclude(nonminute, "not_minute_aligned")
    unresolved_ids = np.flatnonzero(timestamps.isna().to_numpy() | nonminute.to_numpy()) + 1
    if timestamp_convention == "end":
        timestamps = timestamps - MINUTE
    frame = pd.DataFrame({"timestamp": timestamps, "source_row": np.arange(1, count + 1)})
    for name in VALUE_COLUMNS:
        frame[name] = pd.to_numeric(raw[name], errors="coerce").astype(float)
        values = frame[name].to_numpy()
        bad = ~np.isfinite(values) | (values < 0 if name == "volume" else values <= 0)
        exclude(bad, f"invalid_{name}")
    exclude(frame.high < frame[["open", "close", "low"]].max(axis=1), "high_below_ohlc")
    exclude(frame.low > frame[["open", "close", "high"]].min(axis=1), "low_above_ohlc")
    if "contract" in raw:
        frame["contract"] = raw.contract.copy()
        exclude(
            raw.contract.isna() | raw.contract.astype("string").str.strip().eq("").fillna(True),
            "invalid_contract",
        )

    # Inspect duplicate groups before any bad OHLCV row is removed. Otherwise an
    # invalid alternate record could leave whichever valid version is favorable.
    duplicate = timestamps.notna() & timestamps.duplicated(keep=False)
    if duplicate.any():
        duplicate_values = raw.loc[duplicate].drop(columns=timestamp_column)
        duplicate_values.index = pd.DatetimeIndex(timestamps.loc[duplicate])
        different = duplicate_values.groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        conflicting_times = different.index[different]
        conflicting = timestamps.isin(conflicting_times)
        exclude(conflicting, "conflicting_duplicate")
        exclude(
            duplicate & ~conflicting & timestamps.duplicated(keep="first"), "identical_duplicate"
        )

    ticks = frame[list(PRICE_COLUMNS)].to_numpy() / tick_size
    bad_ticks = np.isfinite(ticks) & ~np.isclose(ticks, np.rint(ticks), atol=1e-8, rtol=0)
    tick_violations = [
        {
            "source_row": int(pos) + 1,
            "fields": [name for name, bad in zip(PRICE_COLUMNS, bad_ticks[pos]) if bad],
        }
        for pos in np.flatnonzero(bad_ticks.any(axis=1))
    ]
    excluded_ids = sorted(reasons)
    frame = frame.loc[~frame.source_row.isin(excluded_ids)]
    original_ids = frame.source_row.to_numpy().copy()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    source_ids = frame.source_row.to_numpy()
    moved_ids = np.sort(original_ids[original_ids != source_ids]).tolist()

    elapsed = frame.timestamp.diff()
    gap = (elapsed.notna() & elapsed.ne(MINUTE)).to_numpy()
    contract_change = np.zeros(len(frame), dtype=bool)
    if "contract" in frame and len(frame):
        contract_change[1:] = (
            frame.contract.iloc[1:].to_numpy() != frame.contract.iloc[:-1].to_numpy()
        )
    unresolved_between = np.zeros(len(frame), dtype=bool)
    left = np.zeros(len(frame), dtype=int)
    right = np.zeros(len(frame), dtype=int)
    if len(frame) > 1 and len(unresolved_ids):
        starts = np.minimum(source_ids[:-1], source_ids[1:])
        ends = np.maximum(source_ids[:-1], source_ids[1:])
        left[1:] = np.searchsorted(unresolved_ids, starts, side="right")
        right[1:] = np.searchsorted(unresolved_ids, ends, side="left")
        unresolved_between = right > left
    starts_segment = gap | contract_change | unresolved_between
    frame["segment_id"] = np.cumsum(starts_segment, dtype=np.int64)
    boundaries = [
        {
            "previous_source_row": int(source_ids[pos - 1]),
            "source_row": int(source_ids[pos]),
            "previous_timestamp_utc": frame.timestamp.iloc[pos - 1].isoformat(),
            "timestamp_utc": frame.timestamp.iloc[pos].isoformat(),
            "missing_minutes": max(0, int(elapsed.iloc[pos] / MINUTE) - 1),
            "contract_change": bool(contract_change[pos]),
            "unresolved_source_rows": unresolved_ids[left[pos] : right[pos]].tolist(),
            "calendar_classification": "unverified",
        }
        for pos in np.flatnonzero(starts_segment)
    ]

    segment = frame.segment_id
    previous_close = frame.close.shift().where(segment.eq(segment.shift()))
    true_range = pd.concat(
        [
            frame.high - frame.low,
            (frame.high - previous_close).abs(),
            (frame.low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if len(frame):
        prior_range = true_range.groupby(segment).shift()
        reference = (
            prior_range.groupby(segment)
            .rolling(60, min_periods=60)
            .median()
            .droplevel(0)
            .reindex(frame.index)
        )
        frame["warmup_available"] = frame.groupby("segment_id").cumcount().ge(60)
    else:
        reference = pd.Series(dtype=float, index=frame.index)
        frame["warmup_available"] = pd.Series(dtype=bool, index=frame.index)
    frame["extreme_range"] = (frame.high - frame.low).gt(20 * reference)
    frame["extreme_jump"] = (frame.close - previous_close).abs().gt(20 * reference)
    outliers = [
        {
            "source_row": int(source_ids[pos]),
            "extreme_range": bool(frame.extreme_range.iloc[pos]),
            "extreme_jump": bool(frame.extreme_jump.iloc[pos]),
        }
        for pos in np.flatnonzero((frame.extreme_range | frame.extreme_jump).to_numpy())
    ]
    reason_counts = Counter(reason for row_reasons in reasons.values() for reason in row_reasons)
    audit = {
        "schema_version": "1.0",
        "input_rows": count,
        "output_rows": len(frame),
        "excluded_rows": len(excluded_ids),
        "source_row_numbering": "one_based_data_rows_excluding_header",
        "excluded_row_ids": excluded_ids,
        "exclusions": [{"source_row": row, "reasons": reasons[row]} for row in excluded_ids],
        "reason_counts": dict(sorted(reason_counts.items())),
        "sort": {"reordered": bool(moved_ids), "moved_row_ids": moved_ids},
        "timestamp_normalization": {
            "source_timezone": source_tz,
            "source_convention": timestamp_convention,
            "canonical_timezone": "UTC",
            "canonical_convention": "start",
            "source_timezone_verified": bool(source_timezone_verified),
            "source_convention_verified": bool(source_convention_verified),
            "end_to_start_rows": int(timestamps.notna().sum())
            if timestamp_convention == "end"
            else 0,
        },
        "ignored_source_columns": [
            name
            for name in raw.columns
            if name not in {timestamp_column, *VALUE_COLUMNS, "contract"}
        ],
        "tick_size": float(tick_size),
        "tick_grid_violations": tick_violations,
        "boundaries": boundaries,
        "segment_count": int(segment.nunique()),
        "warmup_unavailable_rows": int((~frame.warmup_available).sum()),
        "outliers": outliers,
        "calendar_status": "unverified",
        "contract_roll_methodology_status": "unverified",
        "confirmatory_metadata_ready": False,
        "imputed_rows": 0,
    }
    return frame, audit
