"""Fixed protocol boundaries are tested on synthetic timestamps only."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from nqpatterns.splits import HOLDOUT_CUTOFF, split_training_holdout, walk_forward_folds


def frame(timestamps):
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True),
                         "sentinel": np.arange(len(timestamps))})


def test_fixed_cutoff_is_midnight_eastern_inclusive_for_holdout():
    source = frame(["2025-01-01T04:59:00Z", "2025-01-01T05:00:00Z", "2025-01-01T05:01:00Z"])
    original = source.copy(deep=True)
    training, holdout = split_training_holdout(source)
    assert HOLDOUT_CUTOFF == pd.Timestamp("2025-01-01T00:00:00-05:00")
    assert training.sentinel.tolist() == [0]
    assert holdout.sentinel.tolist() == [1, 2]
    assert isinstance(training.index, pd.RangeIndex)
    assert isinstance(holdout.index, pd.RangeIndex)
    training.loc[0, "sentinel"] = -99
    pd.testing.assert_frame_equal(source, original)


def test_four_expanding_folds_use_exact_et_boundaries_and_strict_45_minute_purge():
    starts = pd.to_datetime([
        "2023-04-01T04:00:00Z", "2023-07-01T04:00:00Z",
        "2024-01-01T05:00:00Z", "2024-07-01T04:00:00Z",
    ])
    ends = list(starts[1:]) + [pd.Timestamp("2025-01-01T05:00:00Z")]
    timestamps = {pd.Timestamp("2022-12-27T00:00:00Z")}
    for start, end in zip(starts, ends):
        timestamps.update([start - pd.Timedelta(minutes=46), start - pd.Timedelta(minutes=45),
                           start - pd.Timedelta(minutes=1), start, end - pd.Timedelta(minutes=1)])
    training = frame(sorted(timestamps))
    folds = list(walk_forward_folds(training))
    assert [fold.number for fold in folds] == [1, 2, 3, 4]
    for fold, start, end in zip(folds, starts, ends):
        assert fold.fit_start == training.timestamp.iloc[0]
        assert fold.fit_end_exclusive == start - pd.Timedelta(minutes=45)
        assert fold.validation_start == start
        assert fold.validation_end == end
        fit_times = training.timestamp.iloc[fold.fit_indices]
        valid_times = training.timestamp.iloc[fold.validation_indices]
        purged_times = training.timestamp.iloc[fold.purged_indices]
        assert (fit_times < start - pd.Timedelta(minutes=45)).all()
        assert start - pd.Timedelta(minutes=46) in set(fit_times)
        assert start - pd.Timedelta(minutes=45) in set(purged_times)
        assert start - pd.Timedelta(minutes=1) in set(purged_times)
        assert ((valid_times >= start) & (valid_times < end)).all()
        assert start in set(valid_times)
        assert end not in set(valid_times)
        assert not set(fold.fit_indices) & set(fold.validation_indices)
    assert all(set(a.fit_indices) < set(b.fit_indices) for a, b in pairwise(folds))


def test_walk_forward_rejects_any_holdout_row():
    with pytest.raises(ValueError, match="holdout"):
        list(walk_forward_folds(frame(["2024-01-01T00:00Z", "2025-01-01T05:00Z"])))


@pytest.mark.parametrize("operation", [split_training_holdout, lambda f: list(walk_forward_folds(f))])
@pytest.mark.parametrize("timestamps", [
    ["2024-01-02T00:01Z", "2024-01-02T00:00Z"],
    ["2024-01-02T00:00Z", "2024-01-02T00:00Z"],
    ["2024-01-02T00:00Z", None],
])
def test_chronological_functions_reject_unsupported_ordering(operation, timestamps):
    with pytest.raises(ValueError):
        operation(frame(timestamps))


@pytest.mark.parametrize("operation", [split_training_holdout, lambda f: list(walk_forward_folds(f))])
def test_naive_timestamps_are_rejected_before_split(operation):
    source = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02 09:30"])})
    with pytest.raises(ValueError, match="UTC"):
        operation(source)


def test_empty_input_preserves_four_fixed_folds_and_empty_partitions():
    source = frame([])
    training, holdout = split_training_holdout(source)
    assert training.empty and holdout.empty
    folds = list(walk_forward_folds(training))
    assert len(folds) == 4
    assert all(fold.fit_start is None and len(fold.fit_indices) == 0
               and len(fold.validation_indices) == 0 for fold in folds)


def test_missing_timestamp_column_is_explicitly_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        split_training_holdout(pd.DataFrame({"close": [100]}))
