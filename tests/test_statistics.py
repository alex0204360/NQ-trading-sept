"""Behavioral tests for matched, dependence-aware inference, before implementation."""

import json

import numpy as np
import pandas as pd
import pytest

from nqpatterns.statistics import (
    PeriodObservations,
    block_bootstrap,
    holm_adjust,
    matched_estimate,
    moving_block_indices,
)


COLUMNS = ["day", "stratum", "favorable", "ambiguous"]


def events(day, stratum, count, favorable, ambiguous=0):
    return [
        (day, stratum, int(i < favorable), int(favorable <= i < favorable + ambiguous))
        for i in range(count)
    ]


def period(candidate, control, days, name="fold_1"):
    return PeriodObservations(
        candidate=pd.DataFrame(candidate, columns=COLUMNS),
        control=pd.DataFrame(control, columns=COLUMNS),
        days=list(days),
        period_id=name,
    )


def weighted_period():
    days = pd.bdate_range("2023-04-03", periods=40).strftime("%Y-%m-%d").tolist()
    candidate = events(days[0], "A", 3, 1, 1) + events(days[1], "B", 1, 1)
    controls = events(days[0], "A", 30, 3, 6) + events(days[1], "B", 30, 24, 3)
    return period(candidate, controls, days)


def varying_period(name="fold_1", start="2023-04-03", null=False, empty_days=0):
    days = pd.bdate_range(start, periods=80 + empty_days).strftime("%Y-%m-%d").tolist()
    candidate, controls = [], []
    rng = np.random.default_rng(214)
    for i, day in enumerate(days[:80]):
        stratum = "A" if i % 3 else "B"
        controls += events(day, stratum, 30, int(rng.binomial(30, 0.35)))
        probability = 0.35 if null else 0.75
        candidate += events(day, stratum, 15 + i % 4, int(rng.binomial(15, probability)))
    return period(candidate, controls, days, name)


def test_baseline_uses_candidate_weights_and_keeps_ambiguous_in_denominator():
    result = matched_estimate([weighted_period()])
    assert result["status"] == "ok"
    assert result["sample_size"] == 4
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["baseline_hit_rate"] == pytest.approx(0.275)
    assert result["conservative_baseline_hit_rate"] == pytest.approx(0.45)
    assert result["lift"] == pytest.approx(0.225)
    assert result["conservative_lift"] == pytest.approx(0.05)
    assert result["period_results"][0]["event_dates"] == 2
    assert result["period_results"][0]["frame_dates"] == 40
    assert result["period_results"][0]["baseline_support"] is True


def test_pooling_matches_within_each_fold_then_weights_by_candidate_support():
    first = weighted_period()
    days = pd.bdate_range("2023-07-03", periods=40).strftime("%Y-%m-%d").tolist()
    second = period(events(days[0], "A", 4, 3), events(days[0], "A", 30, 6), days, "fold_2")
    result = matched_estimate([first, second])
    assert result["sample_size"] == 8
    assert result["hit_rate"] == pytest.approx(5 / 8)
    assert result["baseline_hit_rate"] == pytest.approx((0.275 + 0.2) / 2)
    assert result["conservative_lift"] == pytest.approx(0.625 - (0.45 + 0.2) / 2)
    assert [x["sample_size"] for x in result["period_results"]] == [4, 4]


def test_controls_in_unoccupied_strata_do_not_fail_observed_support():
    p = weighted_period()
    p.control.loc[len(p.control)] = [p.days[2], "unused", 1, 0]
    assert matched_estimate([p])["status"] == "ok"


def test_insufficient_control_support_is_explicit_and_bootstrap_is_not_run():
    p = weighted_period()
    p.control = p.control.iloc[1:].reset_index(drop=True)
    result = matched_estimate([p])
    assert result["status"] == "baseline_support_failed"
    assert result["period_results"][0]["baseline_support"] is False
    inference = block_bootstrap([p], n_resamples=39)
    assert inference["status"] == "baseline_support_failed"
    assert inference["p_value"] == 1.0
    assert inference["confidence_interval"] is None
    assert inference["block_results"] == []


def test_empty_candidate_is_null_not_fabricated_zero_and_json_is_finite():
    p = weighted_period()
    p.candidate = p.candidate.iloc[:0]
    point = matched_estimate([p])
    assert point["sample_size"] == 0
    assert point["hit_rate"] is None
    assert point["conservative_lift"] is None
    result = block_bootstrap([p], n_resamples=39)
    assert result["status"] == "no_candidate_events"
    assert result["p_value"] == 1.0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["outside_frame", "overlap", "missing", "null", "fraction"])
def test_invalid_events_fail_before_statistics(fault):
    p = weighted_period()
    if fault == "outside_frame":
        p.candidate.loc[0, "day"] = "2030-01-01"
    elif fault == "overlap":
        p.candidate.loc[0, ["favorable", "ambiguous"]] = [1, 1]
    elif fault == "missing":
        p.control = p.control.drop(columns="ambiguous")
    elif fault == "null":
        p.candidate.loc[0, "stratum"] = None
    else:
        p.candidate["favorable"] = p.candidate["favorable"].astype(float)
        p.candidate.loc[0, "favorable"] = 0.5
    with pytest.raises(ValueError):
        matched_estimate([p])


@pytest.mark.parametrize("days", [["2023-04-03", "2023-04-03"], ["2023-04-04", "2023-04-03"], []])
def test_explicit_trading_date_frame_must_be_unique_ordered_and_nonempty(days):
    p = period([], [], days)
    with pytest.raises(ValueError):
        matched_estimate([p])


def test_period_identity_is_unambiguous():
    with pytest.raises(ValueError):
        matched_estimate([])
    with pytest.raises(ValueError):
        matched_estimate([weighted_period(), weighted_period()])


def test_day_draws_are_deterministic_shared_truncated_blocks_without_wrap():
    indices = moving_block_indices(23, 5, 99, "fold_1")
    assert indices.shape == (99, 23)
    assert indices.min() >= 0
    assert indices.max() < 23
    for offset in range(0, 23, 5):
        assert np.all(np.diff(indices[:, offset:min(offset + 5, 23)], axis=1) == 1)
    np.testing.assert_array_equal(indices, moving_block_indices(23, 5, 99, "fold_1"))
    np.testing.assert_array_equal(indices[:7], moving_block_indices(23, 5, 7, "fold_1"))
    assert not np.array_equal(indices, moving_block_indices(23, 5, 99, "fold_2"))
    assert not np.array_equal(indices, moving_block_indices(23, 5, 99, "fold_1", seed=19))


@pytest.mark.parametrize("n_days, block, count", [(19, 20, 9), (25, 0, 9), (25, 5, 0)])
def test_invalid_block_configuration_is_rejected(n_days, block, count):
    with pytest.raises(ValueError):
        moving_block_indices(n_days, block, count, "fold_1")


def test_directional_known_data_produces_positive_bounded_intervals_and_plus_one_p():
    result = block_bootstrap([varying_period()], n_resamples=399)
    assert result["status"] == "ok"
    assert result["confidence_interval"][0] > 0.6
    assert result["lift_confidence_interval"][0] > 0.25
    assert result["p_value"] >= 1 / 400
    assert result["p_value"] < 0.02
    assert [x["block_length"] for x in result["block_results"]] == [5, 20]
    for key in ["confidence_interval", "lift_confidence_interval"]:
        assert result[key] == [
            min(b[key][0] for b in result["block_results"]),
            max(b[key][1] for b in result["block_results"]),
        ]
    assert result["p_value"] == max(b["p_value"] for b in result["block_results"])
    assert all(b["n_resamples"] == 399 for b in result["block_results"])
    json.dumps(result, allow_nan=False)


def test_null_data_has_no_fabricated_positive_significance():
    result = block_bootstrap([varying_period(null=True)], n_resamples=399)
    assert result["status"] == "ok"
    assert result["lift_confidence_interval"][0] < 0
    assert result["p_value"] > 0.05


def test_bootstrap_recomputes_candidate_stratum_weights_in_every_replicate():
    days = pd.bdate_range("2023-04-03", periods=40).strftime("%Y-%m-%d").tolist()
    candidate, controls = [], []
    for i, day in enumerate(days):
        candidate += events(day, "A" if i < 20 else "B", 10, 6 if i % 2 else 8)
        controls += events(day, "A", 30, 3 + i % 3)
        controls += events(day, "B", 30, 23 + i % 4)
    p = period(candidate, controls, days)
    result = block_bootstrap([p], n_resamples=99, block_lengths=(5,))
    indices = moving_block_indices(len(days), 5, 99, "fold_1")
    expected = []
    for row in indices:
        a_count = 10 * np.count_nonzero(row < 20)
        b_count = 10 * np.count_nonzero(row >= 20)
        favorable = sum(6 if i % 2 else 8 for i in row)
        a_baseline = sum(3 + i % 3 for i in row) / (30 * len(days))
        b_baseline = sum(23 + i % 4 for i in row) / (30 * len(days))
        expected.append((favorable - a_count * a_baseline - b_count * b_baseline) / 400)
    np.testing.assert_allclose(result["lift_confidence_interval"], np.quantile(expected, [0.025, 0.975]))
    observed = result["point_estimate"]["conservative_lift"]
    expected_p = (1 + np.count_nonzero(np.asarray(expected) - observed >= observed)) / 100
    assert result["p_value"] == pytest.approx(expected_p)


def test_bootstrap_is_chunk_invariant_and_preserves_full_zero_event_day_frame():
    p = varying_period(empty_days=15)
    a = block_bootstrap([p], n_resamples=99, chunk_size=7)
    b = block_bootstrap([p], n_resamples=99, chunk_size=128)
    assert a == b
    assert a["point_estimate"]["period_results"][0]["frame_dates"] == 95
    assert a["point_estimate"]["period_results"][0]["event_dates"] == 80
    assert a["confidence_interval"] != block_bootstrap([varying_period()], n_resamples=99)["confidence_interval"]


def test_resample_without_controls_is_failed_not_discarded():
    days = pd.bdate_range("2023-04-03", periods=40).strftime("%Y-%m-%d").tolist()
    candidate = sum([events(day, "A", 10, 4 + i % 4) for i, day in enumerate(days)], [])
    p = period(candidate, events(days[0], "A", 40, 10), days)
    result = block_bootstrap([p], n_resamples=99)
    assert result["point_estimate"]["status"] == "ok"
    assert result["status"] == "inference_failed"
    assert result["p_value"] == 1.0
    assert result["confidence_interval"] is None
    assert any("undefined" in reason for reason in result["reasons"])
    assert any(b["undefined_replicates"] > 0 for b in result["block_results"])


def test_degenerate_probabilities_fail_inference_instead_of_narrow_perfect_ci():
    p = varying_period()
    p.candidate["favorable"] = 1
    result = block_bootstrap([p], n_resamples=99)
    assert result["status"] == "inference_failed"
    assert result["p_value"] == 1.0
    assert any("degenerate" in reason for reason in result["reasons"])


def test_identical_candidate_and_control_events_preserve_joint_dependence():
    p = varying_period()
    p.candidate = p.control.copy()
    result = block_bootstrap([p], n_resamples=99)
    assert result["point_estimate"]["conservative_lift"] == pytest.approx(0)
    assert result["status"] == "inference_failed"
    assert all(b["p_value"] == 1 for b in result["block_results"])
    assert any("degenerate" in reason for reason in result["reasons"])


def test_too_short_period_has_an_explicit_inference_failure():
    p = varying_period()
    p.days = p.days[:19]
    p.candidate = p.candidate[p.candidate.day.isin(p.days)]
    p.control = p.control[p.control.day.isin(p.days)]
    result = block_bootstrap([p], n_resamples=99)
    assert result["status"] == "inference_failed"
    assert any("block" in reason for reason in result["reasons"])


def test_two_fold_joint_bootstrap_is_reproducible_with_distinct_date_frames():
    periods = [varying_period(), varying_period("fold_2", "2023-10-02")]
    result = block_bootstrap(periods, n_resamples=99)
    assert result["status"] == "ok"
    assert result == block_bootstrap(periods, n_resamples=99)
    assert len(result["point_estimate"]["period_results"]) == 2


def test_holm_counts_failures_preserves_order_and_monotonic_adjustment():
    assert holm_adjust([0.04, 0.001, 1.0, 0.03]) == pytest.approx([0.09, 0.004, 1.0, 0.09])
    assert holm_adjust([1.0, 1.0]) == [1.0, 1.0]
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])
    with pytest.raises(ValueError):
        holm_adjust([-0.01])
