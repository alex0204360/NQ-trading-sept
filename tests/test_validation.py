"""Protocol gates, event population, and deterministic selection behavior."""

import numpy as np
import pandas as pd
import pytest

from nqpatterns.validation import (
    choose_shortlist,
    holdout_reasons,
    parameter_grid,
    point_estimates,
    resolution_summary,
    screen_reasons,
    stratum_codes,
    training_reasons,
)


def test_grid_contains_every_allowed_horizon_six_thresholds_and_two_directions():
    grid = list(parameter_grid())
    assert len(grid) == 540
    assert {g[1] for g in grid} == set(range(1, 46))
    assert {g[0] for g in grid} == {1, 2, 4, 8, 16, 32}
    assert {g[2] for g in grid} == {"up", "down"}


def test_matched_baseline_weights_candidate_strata_and_penalizes_ambiguity():
    r = point_estimates(
        np.array([1, 1, 2, 0]),
        np.array([0, 0, 1, 1]),
        np.array([1, 2, 3, 0]),
        np.array([0, 0, 1, 1]),
        1,
        min_controls=2,
    )
    assert r["sample_size"] == 4
    assert r["hit_rate"] == 0.5
    assert r["baseline_hit_rate"] == 0.25
    assert r["conservative_baseline_hit_rate"] == 0.5
    assert r["conservative_lift"] == 0
    assert r["baseline_support"] is True
    assert not point_estimates(np.array([1]), np.array([7]), np.array([1]), np.array([0]), 1)[
        "baseline_support"
    ]


def valid_screen():
    return {
        "sample_size": 1500,
        "conservative_lift": 0.08,
        "folds": [
            {
                "sample_size": 375,
                "event_dates": 30,
                "baseline_support": True,
                "conservative_lift": 0.08,
            }
            for _ in range(4)
        ],
    }


def test_all_four_folds_required_and_nonzero_directional_edges():
    r = valid_screen()
    assert screen_reasons(r) == []
    r["folds"][0]["sample_size"] = 99
    assert "fold_sample_size" in screen_reasons(r)
    r = valid_screen()
    r["folds"][0]["conservative_lift"] = -0.021
    assert "fold_instability" in screen_reasons(r)
    r = valid_screen()
    r["folds"] = r["folds"][:3]
    assert "four_folds_required" in screen_reasons(r)
    r = valid_screen()
    r["sample_size"] = 999
    assert "sample_size" in screen_reasons(r)


def test_precision_and_both_block_sensitivities_are_required():
    r = valid_screen()
    r.update(
        status="ok",
        confidence_interval=[0.5, 0.59],
        lift_confidence_interval=[0.02, 0.12],
        block_results=[
            {"lift_confidence_interval": [0.02, 0.11]},
            {"lift_confidence_interval": [0.03, 0.12]},
        ],
    )
    assert training_reasons(r) == []
    r["block_results"][1]["lift_confidence_interval"][0] = 0
    assert "block_sensitivity" in training_reasons(r)
    r["confidence_interval"] = [0.4, 0.51]
    assert "hit_rate_precision" in training_reasons(r)


def test_holdout_gates_include_failed_hypotheses_and_low_support():
    r = {
        "sample_size": 1000,
        "event_dates": 100,
        "iso_weeks": 40,
        "baseline_support": True,
        "conservative_lift": 0.06,
        "confidence_interval": [0.5, 0.59],
        "lift_confidence_interval": [0.01, 0.12],
        "adjusted_p_value": 0.04,
        "status": "ok",
        "block_results": [{"lift_confidence_interval": [0.01, 0.1]}] * 2,
    }
    assert holdout_reasons(r) == []
    r["adjusted_p_value"] = 0.051
    assert "holm_significance" in holdout_reasons(r)
    r["sample_size"] = 29
    assert "low_confidence_n_below_30" in holdout_reasons(r)


def test_shortlist_preserves_signature_diversity_and_family_cap():
    rows = []
    for family in ["a", "b"]:
        for i in range(14):
            for h in [1, 2]:
                rows.append(
                    {
                        "pattern_id": f"{family}{i:02d}_{h}",
                        "signature_id": f"{family}{i:02d}",
                        "family": family,
                        "conservative_lift": 0.1 - i * 0.001,
                        "sample_size": 1000,
                        "mean_favorable_resolution": h,
                        "screen_reasons": [],
                    }
                )
    selected = choose_shortlist(rows)
    assert len(selected) == 24
    assert len({x["signature_id"] for x in selected}) == 24
    assert all(x["pattern_id"].endswith("_1") for x in selected)


def test_resolution_summary_does_not_treat_no_clear_or_ambiguity_as_resolution():
    r = resolution_summary(np.array([1, 2, 3, 0]), np.array([1, 3, 2, np.nan]), 3, 1)
    assert r["all_resolved"]["count"] == 2
    assert r["all_resolved"]["mean"] == 2
    assert r["favorable"]["histogram"] == {"1": 1, "2": 0, "3": 0}
    assert r["cumulative_resolved_fraction"] == {"1": 0.25, "2": 0.25, "3": 0.5}
    assert (
        resolution_summary(np.array([0]), np.array([np.nan]), 1, 1)["all_resolved"]["mean"] is None
    )


def test_strata_use_et_clock_quarter_and_only_supplied_past_cuts():
    t = pd.Series(pd.to_datetime(["2023-06-30T23:59Z", "2023-07-01T04:00Z"]))
    code, names = stratum_codes(t, np.array([0.1, 0.3]), (0.15, 0.25))
    assert names[code[0]] == "2023Q2|18-24|0"
    assert names[code[1]] == "2023Q3|00-06|2"


def test_empty_and_unmatched_estimates_are_json_null_not_nan():
    import json

    for candidate, strata in [
        (np.array([], dtype=int), np.array([], dtype=int)),
        (np.array([1]), np.array([1])),
    ]:
        result = point_estimates(candidate, strata, np.array([2]), np.array([0]), 1)
        assert result["baseline_support"] is False
        assert result["baseline_hit_rate"] is None
        assert result["conservative_lift"] is None
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "codes,strata,direction",
    [([4], [0], 1), ([1], [-1], 1), ([1], [0.5], 1), ([1], [0, 1], 1), ([1], [0], 3)],
)
def test_invalid_or_censored_codes_cannot_enter_eligible_population(codes, strata, direction):
    with pytest.raises(ValueError):
        point_estimates(np.array(codes), np.array(strata), np.array([1]), np.array([0]), direction)


def test_missing_inference_cannot_pass_and_wrong_block_lengths_fail():
    reasons = holdout_reasons({})
    assert {
        "sample_size",
        "event_dates",
        "iso_weeks",
        "low_confidence_n_below_30",
        "baseline_support",
        "minimum_edge",
        "holm_significance",
        "inference_failed",
        "hit_rate_precision",
        "lift_precision",
        "interval_exclusion",
        "block_sensitivity",
    } <= set(reasons)
    row = valid_screen()
    row.update(
        status="ok",
        confidence_interval=[0.5, 0.59],
        lift_confidence_interval=[0.02, 0.12],
        block_results=[
            {"block_length": 5, "lift_confidence_interval": [0.02, 0.12]},
            {"block_length": 10, "lift_confidence_interval": [0.02, 0.12]},
        ],
    )
    assert "block_sensitivity" in training_reasons(row)
    row["folds"][0].update(event_dates=19, baseline_support=False)
    assert {"fold_event_dates", "baseline_support"} <= set(screen_reasons(row))


def test_gate_boundaries_and_statistics_period_alias():
    row = valid_screen()
    row["period_results"] = row.pop("folds")
    row["sample_size"] = 1000
    row["conservative_lift"] = 0.05
    row["period_results"][0]["conservative_lift"] = -0.02
    row.update(
        status="ok",
        confidence_interval=[0.5, 0.6],
        lift_confidence_interval=[0.01, 0.16],
        block_results=[
            {"block_length": length, "lift_confidence_interval": [0.01, 0.16]} for length in (5, 20)
        ],
    )
    assert training_reasons(row) == []


def test_resolution_rejects_missing_or_impossible_confirmed_times():
    for times in ([np.nan], [0], [4], [1.5]):
        with pytest.raises(ValueError):
            resolution_summary([1], times, 3, 1)
    with pytest.raises(ValueError):
        resolution_summary([1], [1], 46, 1)
    import json

    empty = resolution_summary([], [], 3, 1)
    assert empty["all_resolved"]["count"] == 0
    assert empty["cumulative_resolved_fraction"]["3"] is None
    json.dumps(empty, allow_nan=False)


def test_strata_reject_unresolved_timezone_and_missing_calibration_inputs():
    with pytest.raises(ValueError):
        stratum_codes(pd.to_datetime(["2023-01-01"]), [0.1], (0.1, 0.2))
    aware = pd.to_datetime(["2023-01-01Z".replace("01Z", "01T00:00Z")])
    with pytest.raises(ValueError):
        stratum_codes(aware, [np.nan], (0.1, 0.2))
    with pytest.raises(ValueError):
        stratum_codes(aware, [0.1], (0.2, 0.1))
