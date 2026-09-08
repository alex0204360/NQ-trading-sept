"""Frozen definitions, finite search, and matching of causal signatures."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nqpatterns.candidates import generate_candidates, match_candidate
from nqpatterns.features import compute_features


def reference_bars(n=260, start="2023-03-31T18:00:00Z"):
    i = np.arange(n, dtype=float)
    close = 12000 + np.sin(i / 3) * 2 + np.sin(i / 13)
    open_ = close - np.cos(i / 5) * 0.5
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="min"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.2 + np.sin(i / 7) ** 2,
            "low": np.minimum(open_, close) - 0.2 - np.cos(i / 11) ** 2,
            "close": close,
            "volume": 100 + (np.arange(n) * 11) % 73,
        }
    )


@pytest.mark.parametrize(
    ("round_number", "families"),
    [
        (1, {"candle_shape_cluster", "price_state"}),
        (2, {"swing_breakout", "volume_range"}),
        (3, {"price_state", "swing_breakout"}),
        (4, {"candle_geometry", "volume_range"}),
    ],
)
def test_each_initial_round_has_exactly_two_finite_distinct_families(round_number, families):
    candidates = generate_candidates(reference_bars(), round_number)
    assert {candidate["family"] for candidate in candidates} == families
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    for family in families:
        assert 1 <= sum(candidate["family"] == family for candidate in candidates) <= 32
    json.dumps(candidates, allow_nan=False)
    assert all("direction" not in candidate["signature"] for candidate in candidates)


def test_round_one_has_eight_frozen_candle_clusters():
    candidates = generate_candidates(reference_bars(), 1)
    shape = [c for c in candidates if c["family"] == "candle_shape_cluster"]
    assert len(shape) == 8
    assert {c["signature"]["cluster"] for c in shape} == set(range(8))
    artifact = shape[0]["signature"]["artifact"]
    assert all(c["signature"]["artifact"] == artifact for c in shape)
    assert len(artifact["features"]) == 5
    assert len(artifact["mean"]) == len(artifact["scale"]) == 5
    assert np.asarray(artifact["centroids"]).shape == (8, 5)
    assert artifact["seed"] == 20260906
    assert artifact["fit_end_exclusive"] == "2023-04-01T04:00:00Z"
    assert all(type(value) is float for value in artifact["mean"] + artifact["scale"])


def test_candidate_id_hashes_the_exact_canonical_definition():
    for candidate in generate_candidates(reference_bars(), 1):
        canonical = json.dumps(
            {"family": candidate["family"], "signature": candidate["signature"]},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert candidate["id"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_candidates_are_reproducible_without_global_random_state_dependence():
    np.random.seed(1)
    first = generate_candidates(reference_bars(), 1)
    np.random.seed(912)
    second = generate_candidates(reference_bars(), 1)
    assert first == second


def test_scaler_and_centroids_ignore_all_bars_at_or_after_initial_fit_boundary():
    frame = reference_bars(800)
    expected = generate_candidates(frame, 1)
    after_fit = frame["timestamp"] >= pd.Timestamp("2023-04-01T04:00:00Z")
    altered = frame.copy()
    altered.loc[after_fit, ["open", "high", "low", "close"]] *= 4
    altered.loc[after_fit, "volume"] *= 9
    assert generate_candidates(altered, 1) == expected
    assert generate_candidates(frame.loc[~after_fit], 1) == expected


def test_matching_json_roundtrip_uses_the_same_frozen_centroids(monkeypatch):
    candidates = generate_candidates(reference_bars(), 1)
    features = compute_features(reference_bars(400))
    before = json.dumps(candidates, allow_nan=False, sort_keys=True)

    def fail_refit(*args, **kwargs):
        raise AssertionError("Frozen candidate matching must not fit another model")

    from sklearn.cluster import KMeans

    monkeypatch.setattr(KMeans, "fit", fail_refit)
    restored = json.loads(before)
    for original, loaded in zip(candidates, restored, strict=True):
        left = match_candidate(features, original["signature"])
        right = match_candidate(features, loaded["signature"])
        assert isinstance(left, np.ndarray)
        assert left.dtype == bool
        np.testing.assert_array_equal(left, right)
        assert not left[:60].any()
    assert json.dumps(candidates, allow_nan=False, sort_keys=True) == before
    assert {c["family"] for c in generate_candidates(reference_bars(), 2)} == {
        "swing_breakout", "volume_range"
    }


def test_each_valid_bar_matches_exactly_one_shape_cluster():
    frame = reference_bars()
    features = compute_features(frame)
    candidates = generate_candidates(frame, 1)
    matches = [
        match_candidate(features, c["signature"])
        for c in candidates if c["family"] == "candle_shape_cluster"
    ]
    np.testing.assert_array_equal(np.sum(matches, axis=0), features["valid"].astype(int))


@pytest.mark.parametrize("round_number", [1, 2])
def test_matches_are_causal_under_prefix_and_future_perturbation(round_number):
    frame = reference_bars()
    candidates = generate_candidates(frame, round_number)
    future = frame.copy()
    future.loc[150:, ["open", "high", "low", "close"]] += 1000
    future.loc[150:, "volume"] *= 10
    full_features = compute_features(frame)
    prefix_features = compute_features(frame.iloc[:150])
    changed_features = compute_features(future)
    for candidate in candidates:
        signature = candidate["signature"]
        full = match_candidate(full_features, signature)
        np.testing.assert_array_equal(full[:150], match_candidate(prefix_features, signature))
        np.testing.assert_array_equal(full[:150], match_candidate(changed_features, signature)[:150])


def test_rule_comparisons_obey_exact_boundaries_and_feature_validity():
    features = pd.DataFrame(
        {"valid": [True, True, True, True, False], "price_z20": [-2, -1, 0, 1, 0]}
    )
    signature = {
        "version": 1,
        "kind": "rules",
        "conditions": [
            {"feature": "price_z20", "op": "ge", "value": -1.0},
            {"feature": "price_z20", "op": "lt", "value": 1.0},
        ],
    }
    np.testing.assert_array_equal(match_candidate(features, signature), [False, True, True, False, False])


def test_no_volume_at_price_or_vwap_is_claimed_by_bar_volume_signatures():
    signatures = generate_candidates(reference_bars(), 2)
    encoded = json.dumps(signatures).lower()
    assert "vwap" not in encoded
    assert "volume_profile" not in encoded
    assert "volume_at_price" not in encoded
    volume_family = [c for c in signatures if c["family"] == "volume_range"]
    assert all("volume_z20" in json.dumps(c["signature"]) for c in volume_family)


@pytest.mark.parametrize("round_number", [0, 5, -1])
def test_rejects_rounds_outside_finite_tournament(round_number):
    with pytest.raises(ValueError):
        generate_candidates(reference_bars(), round_number)


def test_cluster_generation_fails_explicitly_without_initial_fit_support():
    with pytest.raises(ValueError, match="(?i)(initial|reference|fit|warmup)"):
        generate_candidates(reference_bars(60), 1)


@pytest.mark.parametrize(
    "signature",
    [
        {"version": 1, "kind": "unknown"},
        {"version": 1, "kind": "rules", "conditions": [{"feature": "price_z20", "op": "future", "value": 0}]},
        {"version": 1, "kind": "rules", "conditions": [{"feature": "future_price", "op": "ge", "value": 0}]},
    ],
)
def test_unknown_signatures_fail_instead_of_silently_matching(signature):
    with pytest.raises(ValueError):
        match_candidate(compute_features(reference_bars()), signature)


def test_later_rounds_preserve_frozen_definitions_regardless_of_previous_outcomes():
    frame = reference_bars()
    for round_number in (3, 4):
        expected = generate_candidates(frame, round_number)
        assert generate_candidates(frame, round_number, {"future_holdout_hit_rate": 1}) == expected
        assert any(match_candidate(compute_features(frame), c["signature"]).any() for c in expected)


@pytest.mark.parametrize("changes", [
    {"version": 2},
    {"conditions": []},
    {"conditions": ["bad condition"]},
    {"conditions": [{"feature": "price_z20", "op": "ge", "value": float("nan")}]},
    {"conditions": [{"feature": "price_z20", "op": "ge", "value": True}]},
])
def test_malformed_rules_fail_before_matching(changes):
    signature = {"version": 1, "kind": "rules", "conditions": [
        {"feature": "price_z20", "op": "ge", "value": 0.0}
    ]}
    signature.update(changes)
    with pytest.raises(ValueError):
        match_candidate(compute_features(reference_bars()), signature)


@pytest.mark.parametrize("defect", ["artifact", "features", "numeric", "shape", "scale", "cluster"])
def test_malformed_frozen_cluster_artifacts_fail_before_matching(defect):
    signature = {
        "version": 1, "kind": "cluster", "cluster": 0,
        "artifact": {"features": ["body_fraction"], "mean": [0.0], "scale": [1.0],
                     "centroids": [[0.0], [1.0]]},
    }
    if defect == "artifact":
        signature["artifact"] = None
    elif defect == "features":
        signature["artifact"]["features"] = ["tomorrow"]
    elif defect == "numeric":
        signature["artifact"]["mean"] = ["bad value"]
    elif defect == "shape":
        signature["artifact"]["centroids"] = [[0.0, 1.0]]
    elif defect == "scale":
        signature["artifact"]["scale"] = [0.0]
    else:
        signature["cluster"] = 2
    with pytest.raises(ValueError):
        match_candidate(compute_features(reference_bars()), signature)


def test_missing_feature_or_invalid_valid_column_is_rejected():
    signature = {"version": 1, "kind": "rules", "conditions": [
        {"feature": "price_z20", "op": "ge", "value": 0.0}
    ]}
    for frame in [pd.DataFrame({"valid": [True]}), pd.DataFrame({"valid": [1], "price_z20": [2]})]:
        with pytest.raises(ValueError):
            match_candidate(frame, signature)


def test_cluster_matching_has_no_matches_before_warmup_and_rejects_missing_features():
    signature = generate_candidates(reference_bars(), 1)[0]["signature"]
    features = compute_features(reference_bars(20))
    assert not match_candidate(features, signature).any()
    with pytest.raises(ValueError, match="feature"):
        match_candidate(features.drop(columns="body_fraction"), signature)


def test_naive_initial_fit_dates_are_rejected():
    frame = reference_bars()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="aware"):
        generate_candidates(frame, 1)
