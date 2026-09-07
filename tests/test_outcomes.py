"""Behavioral contracts for first-passage labels and causal signal thinning."""

import numpy as np
import pandas as pd
import pytest

from nqpatterns.outcomes import (
    OutOfScopeHorizon,
    first_passage,
    label_outcomes,
    labels_from_passage,
    nonoverlap_indices,
    resolution_time_summary,
)


def bars(count=60):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=count, freq="min", tz="UTC"),
            "open": np.full(count, 100.0),
            "high": np.full(count, 100.5),
            "low": np.full(count, 99.5),
            "close": np.full(count, 100.0),
            "volume": np.full(count, 10.0),
            "contract": np.full(count, "NQH4", dtype=object),
        }
    )


@pytest.mark.parametrize(
    ("high", "low", "expected"),
    [(101.0, 99.5, "up"), (100.5, 99.0, "down"), (100.75, 99.25, "no_clear_outcome")],
)
def test_horizon_one_includes_next_bars_intrabar_range_and_exact_touches(high, low, expected):
    frame = bars()
    frame.loc[1, ["high", "low"]] = [high, low]
    result = label_outcomes(frame, magnitude_points=1.0, horizon=1)
    assert result.loc[0, "outcome"] == expected
    assert result.loc[0, "eligible_45"]
    assert result.loc[0, "eligible_horizon"]
    if expected in {"up", "down"}:
        assert result.loc[0, "resolution_bars"] == 1.0
    else:
        assert pd.isna(result.loc[0, "resolution_bars"])


def test_detection_bar_is_completed_and_entry_is_next_open():
    frame = bars()
    frame.loc[0, ["open", "high", "low", "close"]] = [150.0, 160.0, 140.0, 150.0]
    frame.loc[1, "high"] = 101.0
    assert label_outcomes(frame, 1.0, 1).loc[0, "outcome"] == "up"


def test_horizon_45_includes_45th_future_bar_and_stops_there():
    frame = bars(47)
    frame.loc[45, "high"] = 101.0
    assert label_outcomes(frame, 1.0, 44).loc[0, "outcome"] == "no_clear_outcome"
    result = label_outcomes(frame, 1.0, 45)
    assert result.loc[0, "outcome"] == "up"
    assert result.loc[0, "resolution_bars"] == 45.0
    frame.loc[45, "high"] = 100.5
    frame.loc[46, "high"] = 101.0
    assert label_outcomes(frame, 1.0, 45).loc[0, "outcome"] == "no_clear_outcome"


def test_horizon_46_is_a_scope_rejection_before_market_data_access():
    with pytest.raises(OutOfScopeHorizon) as caught:
        label_outcomes(None, magnitude_points=1.0, horizon=46)
    assert caught.value.status == "rejected_out_of_scope"
    assert caught.value.reason == "rejected — out of scope: holding window exceeds 45 bars"
    assert str(caught.value) == caught.value.reason


@pytest.mark.parametrize("horizon", [0, -1, 1.0, 1.5, True, "1", None])
def test_horizon_requires_an_integer_from_one_through_45(horizon):
    with pytest.raises((TypeError, ValueError)):
        label_outcomes(bars(), 1.0, horizon)


@pytest.mark.parametrize("magnitude", [0, -1, np.nan, np.inf, -np.inf, 0.1, 1.125, True, "1"])
def test_magnitude_requires_a_finite_positive_tick_aligned_number(magnitude):
    with pytest.raises((TypeError, ValueError)):
        label_outcomes(bars(), magnitude, 1)


def test_quarter_point_magnitude_is_supported():
    frame = bars()
    frame.loc[1, ["high", "low"]] = [100.25, 100.0]
    assert label_outcomes(frame, 0.25, 1).loc[0, "outcome"] == "up"


def test_double_touch_is_ambiguous_and_never_resolves_from_a_later_bar():
    frame = bars()
    frame.loc[1, ["high", "low"]] = [101.0, 99.0]
    frame.loc[2, "high"] = 102.0
    result = label_outcomes(frame, 1.0, 3)
    assert result.loc[0, "outcome"] == "ambiguous_both"
    assert pd.isna(result.loc[0, "resolution_bars"])
    passage = first_passage(frame, 1.0)
    assert passage.first_direction[0] == "ambiguous_both"
    assert passage.first_touch_bars[0] == 1.0


@pytest.mark.parametrize(
    ("opening", "high", "low", "expected"),
    [(102.0, 103.0, 98.0, "up"), (98.0, 103.0, 97.0, "down"), (101.0, 102.0, 99.0, "up")],
)
def test_later_opening_barrier_precedes_high_low_even_when_both_touch(opening, high, low, expected):
    frame = bars()
    frame.loc[2, ["open", "high", "low"]] = [opening, high, low]
    result = label_outcomes(frame, 1.0, 2)
    assert result.loc[0, "outcome"] == expected
    assert result.loc[0, "resolution_bars"] == 2.0


def test_first_touch_direction_is_retained_when_opposite_barrier_hits_later():
    frame = bars()
    frame.loc[2, "low"] = 99.0
    frame.loc[3, "high"] = 101.0
    result = label_outcomes(frame, 1.0, 4)
    assert result.loc[0, "outcome"] == "down"
    assert result.loc[0, "resolution_bars"] == 2.0


def test_missing_minute_censors_window_even_when_barrier_was_reached_before_gap():
    frame = bars()
    frame.loc[1, "high"] = 101.0
    frame = frame.drop(index=2).reset_index(drop=True)
    short = label_outcomes(frame, 1.0, 1)
    long = label_outcomes(frame, 1.0, 2)
    assert short.loc[0, "outcome"] == "up"
    assert not short.loc[0, "eligible_45"]
    assert long.loc[0, "outcome"] == "censored_data"
    assert pd.isna(long.loc[0, "resolution_bars"])
    assert not long.loc[0, "eligible_horizon"]
    assert long.loc[2, "eligible_45"]


def test_entry_must_be_the_next_consecutive_minute():
    frame = bars().drop(index=1).reset_index(drop=True)
    frame.loc[1, "high"] = 110.0
    assert label_outcomes(frame, 1.0, 1).loc[0, "outcome"] == "censored_data"


@pytest.mark.parametrize("boundary", [1, 2])
def test_contract_change_censors_crossing_window(boundary):
    frame = bars()
    frame.loc[boundary:, "contract"] = "NQM4"
    frame.loc[1, "high"] = 101.0
    result = label_outcomes(frame, 1.0, 2)
    assert result.loc[0, "outcome"] == "censored_data"
    assert not result.loc[0, "eligible_45"]
    assert result.loc[boundary, "eligible_45"]


def test_contract_column_is_optional():
    frame = bars().drop(columns="contract")
    assert label_outcomes(frame, 1.0, 1).loc[0, "outcome"] == "no_clear_outcome"


def test_tail_and_common_population_distinguish_short_horizon_availability():
    frame = bars(46)
    result = label_outcomes(frame, 1.0, 1)
    assert result["eligible_45"].tolist() == [True] + [False] * 45
    assert result["eligible_horizon"].tolist() == [True] * 45 + [False]
    assert result.loc[44, "outcome"] == "no_clear_outcome"
    assert result.loc[45, "outcome"] == "censored_data"
    assert result["resolution_bars"].isna().all()


def test_reusable_passage_matches_labels_at_every_horizon_without_mutating_input():
    frame = bars(100)
    frame.loc[8, "high"] = 101.0
    frame.loc[11, ["high", "low"]] = [102.0, 98.0]
    frame.loc[17, "low"] = 99.0
    frame.loc[40:, "contract"] = "NQM4"
    before = frame.copy(deep=True)
    passage = first_passage(frame, 1.0)
    for horizon in range(1, 46):
        pd.testing.assert_frame_equal(
            labels_from_passage(passage, horizon), label_outcomes(frame, 1.0, horizon)
        )
    pd.testing.assert_frame_equal(frame, before)
    assert passage.available_bars[0] == 39
    assert passage.available_bars[40] == 45
    assert passage.first_direction[-1] == ""
    assert np.isnan(passage.first_touch_bars[-1])
    assert isinstance(labels_from_passage(passage, 1)["resolution_bars"].dtype, pd.Float64Dtype)


def test_first_passage_arrays_are_read_only():
    passage = first_passage(bars(), 1.0)
    for field in ("first_direction", "first_touch_bars", "available_bars", "eligible_45"):
        array = getattr(passage, field)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = array[0]


def test_short_scan_still_computes_45_minute_eligibility_and_rejects_unscanned_labels():
    frame = bars()
    passage = first_passage(frame, 1.0, max_horizon=2)
    assert passage.eligible_45[0]
    assert passage.available_bars[0] == 45
    assert labels_from_passage(passage, 2).loc[0, "outcome"] == "no_clear_outcome"
    with pytest.raises(ValueError, match="scan|scanned|max_horizon"):
        labels_from_passage(passage, 3)
    with pytest.raises(OutOfScopeHorizon):
        first_passage(None, 1.0, max_horizon=46)


@pytest.mark.parametrize("change", ["naive", "duplicate", "reversed", "nat"])
def test_unsupported_timestamps_are_rejected(change):
    frame = bars()
    if change == "naive":
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    elif change == "duplicate":
        frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    elif change == "reversed":
        frame = frame.iloc[::-1].reset_index(drop=True)
    else:
        frame.loc[2, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="timestamp|time|ordered|increasing"):
        label_outcomes(frame, 1.0, 1)


def test_empty_canonical_bars_return_empty_label_and_passage_arrays():
    frame = bars(0)
    result = label_outcomes(frame, 1.0, 45)
    assert result.empty
    assert result.index.equals(frame.index)
    assert list(result.columns) == ["outcome", "resolution_bars", "eligible_45", "eligible_horizon"]
    assert first_passage(frame, 1.0).available_bars.size == 0


def test_nonoverlap_uses_elapsed_minutes_and_allows_exact_boundary():
    timestamps = pd.to_datetime(
        ["2024-01-02T00:00Z", "2024-01-02T00:44Z", "2024-01-02T00:45Z", "2024-01-02T01:29Z", "2024-01-02T01:30Z"]
    )
    assert nonoverlap_indices(timestamps, np.ones(5, dtype=bool)).tolist() == [0, 2, 4]


def test_censored_signal_consumes_cooldown_before_future_eligibility_filter():
    frame = bars(120).drop(index=10).reset_index(drop=True)
    raw = np.zeros(len(frame), dtype=bool)
    raw[[0, 5, 44]] = True
    retained = nonoverlap_indices(frame["timestamp"], raw)
    assert retained.tolist() == [0, 44]
    labels = label_outcomes(frame, 1.0, 45)
    assert labels.loc[0, "outcome"] == "censored_data"
    eligible = retained[labels.loc[retained, "eligible_45"].to_numpy()]
    assert eligible.tolist() == [44]


def test_thinning_is_prefix_invariant_and_does_not_restart_at_a_gap():
    frame = bars(130).drop(index=range(12, 20)).reset_index(drop=True)
    raw = np.arange(len(frame)) % 3 == 0
    retained = nonoverlap_indices(frame["timestamp"], raw)
    for stop in (1, 11, 12, 20, 40, 60, 100, len(frame)):
        prefix = nonoverlap_indices(frame["timestamp"].iloc[:stop], raw[:stop])
        np.testing.assert_array_equal(prefix, retained[retained < stop])
    assert retained[1] == 39


def test_no_raw_signals_and_empty_schedule():
    assert nonoverlap_indices(bars()["timestamp"], np.zeros(60, dtype=bool)).size == 0
    assert nonoverlap_indices(bars(0)["timestamp"], np.zeros(0, dtype=bool)).size == 0


@pytest.mark.parametrize("mask", [np.zeros(2, dtype=bool), np.zeros(60, dtype=int), [None] * 60])
def test_thinning_rejects_invalid_raw_masks(mask):
    with pytest.raises((TypeError, ValueError)):
        nonoverlap_indices(bars()["timestamp"], mask)


@pytest.mark.parametrize("minutes", [0, -1, 1.5, True])
def test_thinning_requires_a_positive_integer_cooldown(minutes):
    with pytest.raises((TypeError, ValueError)):
        nonoverlap_indices(bars()["timestamp"], np.ones(60, dtype=bool), minutes=minutes)


def test_resolution_summary_excludes_ambiguity_no_clear_and_censored_from_times():
    labels = pd.DataFrame(
        {
            "outcome": ["up", "down", "up", "no_clear_outcome", "ambiguous_both", "censored_data"],
            "resolution_bars": pd.array([1.0, 2.0, 4.0, None, None, None], dtype="Float64"),
        }
    )
    result = resolution_time_summary(labels, horizon=4, favorable_direction="up")
    assert result["eligible_count"] == 5
    assert result["all_resolved"]["count"] == 3
    assert result["all_resolved"]["mean"] == pytest.approx(7 / 3)
    assert result["all_resolved"]["std"] == pytest.approx(np.std([1, 2, 4], ddof=1))
    assert result["all_resolved"]["median"] == 2.0
    assert result["all_resolved"]["p25"] == 1.5
    assert result["all_resolved"]["p75"] == 3.0
    assert result["all_resolved"]["p90"] == pytest.approx(3.6)
    assert result["all_resolved"]["min"] == 1.0
    assert result["all_resolved"]["max"] == 4.0
    assert result["all_resolved"]["histogram"] == {1: 1, 2: 1, 3: 0, 4: 1}
    assert result["favorable"]["histogram"] == {1: 1, 2: 0, 3: 0, 4: 1}
    assert result["favorable"]["mean"] == 2.5
    assert result["cumulative_resolved_fraction"] == {1: 0.2, 2: 0.4, 3: 0.4, 4: 0.6}
    assert result["cumulative_favorable_fraction"] == {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.4}


def test_empty_resolution_distributions_use_null_summaries_and_exact_zero_histogram():
    labels = pd.DataFrame(
        {"outcome": ["censored_data"], "resolution_bars": pd.array([None], dtype="Float64")}
    )
    result = resolution_time_summary(labels, horizon=2, favorable_direction="down")
    assert result["eligible_count"] == 0
    for name in ("all_resolved", "favorable"):
        assert result[name]["count"] == 0
        for statistic in ("mean", "std", "median", "p25", "p75", "p90", "min", "max"):
            assert result[name][statistic] is None
        assert result[name]["histogram"] == {1: 0, 2: 0}
    assert result["cumulative_resolved_fraction"] == {1: None, 2: None}


def test_single_resolution_has_undefined_sample_standard_deviation():
    labels = pd.DataFrame({"outcome": ["down"], "resolution_bars": [1.0]})
    result = resolution_time_summary(labels, horizon=1, favorable_direction="down")
    assert result["all_resolved"]["std"] is None
    assert result["favorable"]["std"] is None


def test_resolution_summary_rejects_invalid_favorable_direction():
    labels = label_outcomes(bars(), 1.0, 1)
    with pytest.raises(ValueError, match="direction"):
        resolution_time_summary(labels, horizon=1, favorable_direction="sideways")
