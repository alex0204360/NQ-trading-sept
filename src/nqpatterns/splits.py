"""Fixed chronological partitions from the committed research protocol."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

HOLDOUT_CUTOFF = pd.Timestamp("2025-01-01T00:00:00-05:00")
PURGE = pd.Timedelta(minutes=45)
VALIDATION_BOUNDARIES = tuple(
    pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")
    for value in ("2023-04-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01")
)


@dataclass(frozen=True)
class Fold:
    """Positional arrays index the unchanged caller's training frame."""

    number: int
    fit_start: pd.Timestamp | None
    fit_end_exclusive: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    purged_indices: np.ndarray


def _validate(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame:
        raise ValueError("A canonical UTC timestamp column is required")
    times = frame.timestamp
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC":
        raise ValueError("Canonical timestamps must be timezone-aware UTC")
    if times.isna().any() or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError("Canonical timestamps must be non-null, unique, and increasing")
    return times


def split_training_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy bars before/at the fixed cutoff; never select a split from outcomes."""
    times = _validate(frame)
    training = frame.loc[times < HOLDOUT_CUTOFF].copy(deep=True).reset_index(drop=True)
    holdout = frame.loc[times >= HOLDOUT_CUTOFF].copy(deep=True).reset_index(drop=True)
    return training, holdout


def _positions(mask: pd.Series) -> np.ndarray:
    values = np.flatnonzero(mask.to_numpy())
    values.flags.writeable = False
    return values


def walk_forward_folds(training: pd.DataFrame) -> Iterator[Fold]:
    """Yield all four fixed folds, including explicitly empty unsupported folds.

    Fit rows are strictly earlier than validation start minus 45 elapsed minutes.
    This splitter does not score outcomes or allow access to any holdout row.
    """
    times = _validate(training)
    if (times >= HOLDOUT_CUTOFF).any():
        raise ValueError("walk_forward_folds cannot accept holdout rows")
    for number, (start, end) in enumerate(pairwise(VALIDATION_BOUNDARIES), start=1):
        fit_end = start - PURGE
        yield Fold(
            number=number,
            fit_start=times.iloc[0] if len(times) else None,
            fit_end_exclusive=fit_end,
            validation_start=start,
            validation_end=end,
            fit_indices=_positions(times < fit_end),
            validation_indices=_positions((times >= start) & (times < end)),
            purged_indices=_positions((times >= fit_end) & (times < start)),
        )
