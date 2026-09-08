# ruff: noqa: TRY004 -- schema errors consistently use ValueError
"""Finite, JSON-native candidate definitions and frozen matching.

Candidate directions and outcome parameters are deliberately separate from
their signatures: the tournament evaluates every preregistered outcome pair.
"""

from __future__ import annotations

import hashlib
import json
import operator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .features import FEATURE_COLUMNS, compute_features

INITIAL_FIT_END = "2023-04-01T04:00:00Z"
ROOT_SEED = 20260906
SHAPE_FEATURES = (
    "body_fraction",
    "upper_wick_fraction",
    "lower_wick_fraction",
    "lag1_return_to_range",
    "range_to_atr14",
)
OPERATORS = {
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "eq": operator.eq,
}


def validate_signature(signature: dict[str, Any]) -> None:
    """Reject unsupported or nonfinite signatures before any matching."""
    if not isinstance(signature, dict) or signature.get("version") != 1:
        raise ValueError("Unsupported signature version")
    kind = signature.get("kind")
    if kind == "rules":
        conditions = signature.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("A rules signature requires nonempty conditions")
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError("Each condition must be a dictionary")
            if condition.get("feature") not in FEATURE_COLUMNS:
                raise ValueError("Unknown or noncausal feature")
            if condition.get("op") not in OPERATORS:
                raise ValueError("Unknown comparison operator")
            value = condition.get("value")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError("Rule threshold must be finite")
    elif kind == "cluster":
        artifact = signature.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("Cluster requires a frozen artifact")
        names = artifact.get("features")
        if (
            not isinstance(names, list)
            or not names
            or any(name not in FEATURE_COLUMNS for name in names)
        ):
            raise ValueError("Unknown cluster features")
        try:
            mean = np.asarray(artifact.get("mean"), dtype=float)
            scale = np.asarray(artifact.get("scale"), dtype=float)
            centers = np.asarray(artifact.get("centroids"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid numeric cluster artifact") from exc
        if (
            mean.shape != (len(names),)
            or scale.shape != mean.shape
            or centers.ndim != 2
            or centers.shape[1] != len(names)
            or len(centers) < 1
        ):
            raise ValueError("Cluster artifact dimensions disagree")
        if (
            not all(np.isfinite(values).all() for values in (mean, scale, centers))
            or (scale <= 0).any()
        ):
            raise ValueError("Cluster artifact must have finite values and positive scales")
        cluster = signature.get("cluster")
        if type(cluster) is not int or not 0 <= cluster < len(centers):
            raise ValueError("Cluster index is outside the frozen artifact")
    else:
        raise ValueError("Unknown signature kind")


def match_candidate(features: pd.DataFrame, signature: dict[str, Any]) -> np.ndarray:
    """Match fixed rules or nearest centroids; never fit or change artifacts."""
    validate_signature(signature)
    if "valid" not in features or features["valid"].dtype != bool:
        raise ValueError("features must include a boolean valid column")
    result = features["valid"].to_numpy(copy=True)
    if signature["kind"] == "rules":
        for condition in signature["conditions"]:
            name = condition["feature"]
            if name not in features:
                raise ValueError(f"Missing required feature: {name}")
            values = features[name].to_numpy(dtype=float)
            result &= np.isfinite(values) & OPERATORS[condition["op"]](values, condition["value"])
        return result
    artifact = signature["artifact"]
    if any(name not in features for name in artifact["features"]):
        raise ValueError("Missing required cluster features")
    positions = np.flatnonzero(result)
    values = features.loc[:, artifact["features"]].to_numpy(dtype=float)[positions]
    result[:] = False
    if not len(values):
        return result
    finite = np.isfinite(values).all(axis=1)
    values = (values[finite] - np.asarray(artifact["mean"])) / np.asarray(artifact["scale"])
    centers = np.asarray(artifact["centroids"])
    # One distance per center avoids an N x K x F intermediate on long history.
    distance = np.column_stack([np.square(values - center).sum(axis=1) for center in centers])
    result[positions[finite]] = np.argmin(distance, axis=1) == signature["cluster"]
    return result


def _candidate(family: str, signature: dict[str, Any]) -> dict[str, Any]:
    validate_signature(signature)
    definition = {"family": family, "signature": signature}
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), **definition}


def _rule(family: str, *conditions: tuple[str, str, float]) -> dict[str, Any]:
    return _candidate(
        family,
        {
            "version": 1,
            "kind": "rules",
            "conditions": [
                {"feature": name, "op": op, "value": float(value)} for name, op, value in conditions
            ],
        },
    )


def _shape_clusters(reference_bars: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = pd.DatetimeIndex(reference_bars["timestamp"])
    if timestamps.tz is None:
        raise ValueError("Initial reference timestamps must be timezone aware")
    initial = reference_bars.loc[timestamps < pd.Timestamp(INITIAL_FIT_END)]
    features = compute_features(initial)
    values = features.loc[features["valid"], list(SHAPE_FEATURES)].to_numpy(dtype=float)
    if len(values) < 8 or len(np.unique(values, axis=0)) < 8:
        raise ValueError("Insufficient distinct initial-fit observations after 60-bar warmup")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    model = KMeans(n_clusters=8, random_state=ROOT_SEED, n_init=10, algorithm="lloyd")
    # Stable JSON artifact hashes require deterministic floating reductions as
    # well as a fixed random seed. Parallel BLAS/OpenMP reductions may differ.
    with threadpool_limits(limits=1):
        model.fit((values - mean) / scale)
    artifact = {
        "features": list(SHAPE_FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "centroids": model.cluster_centers_.tolist(),
        "seed": ROOT_SEED,
        "fit_end_exclusive": INITIAL_FIT_END,
    }
    return [
        _candidate(
            "candle_shape_cluster",
            {
                "version": 1,
                "kind": "cluster",
                "cluster": cluster,
                "artifact": artifact,
            },
        )
        for cluster in range(8)
    ]


def generate_candidates(
    reference_bars: pd.DataFrame,
    round_number: int,
    previous_results: Any = None,
) -> list[dict[str, Any]]:
    """Register two bounded families per round before accessing their outcomes.

    Round 3 deepens price state and swing geometry. Round 4 deepens candle
    geometry and volume/range. Their thresholds are deterministic and use no
    performance feedback; ``previous_results`` is reserved and ignored. Only
    round 1 learns a scaler/model, exclusively before the initial fit cutoff.
    """
    if type(round_number) is not int or round_number not in (1, 2, 3, 4):
        raise ValueError("round_number must be an integer from 1 through 4")
    candidates: list[dict[str, Any]] = []
    if round_number == 1:
        candidates.extend(_shape_clusters(reference_bars))
        for edge in (0.5, 1.0, 1.5, 2.0):
            for side in (-1, 1):
                for trend in (-1, 1):
                    candidates.append(
                        _rule(
                            "price_state",
                            ("price_z20", "ge" if side > 0 else "le", side * edge),
                            ("trend20_60", "ge" if trend > 0 else "lt", 0),
                        )
                    )
    elif round_number == 2:
        for window in (20, 60):
            for side in ("up", "down"):
                for edge in (0.0, 0.25, 0.5, 1.0):
                    candidates.append(
                        _rule(
                            "swing_breakout",
                            (f"breakout_{side}{window}_atr", "ge", edge),
                        )
                    )
        for volume_edge in (0.0, 1.0, 2.0):
            for range_edge in (0.0, 1.0):
                for side in (-1, 1):
                    candidates.append(
                        _rule(
                            "volume_range",
                            ("volume_z20", "ge", volume_edge),
                            ("range_z20", "ge", range_edge),
                            ("body_fraction", "ge" if side > 0 else "lt", 0),
                        )
                    )
    elif round_number == 3:
        for side in (-1, 1):
            for edge in (0.75, 1.25, 1.75, 2.25):
                for wick in (0.25, 0.5):
                    candidates.append(
                        _rule(
                            "price_state",
                            ("price_z20", "ge" if side > 0 else "le", side * edge),
                            (
                                "upper_wick_fraction" if side > 0 else "lower_wick_fraction",
                                "ge",
                                wick,
                            ),
                        )
                    )
        for window in (20, 60):
            for side in ("up", "down"):
                for distance in (0.25, 0.5):
                    for body in (0.25, 0.5):
                        candidates.append(
                            _rule(
                                "swing_breakout",
                                (f"breakout_{side}{window}_atr", "ge", -distance),
                                (f"breakout_{side}{window}_atr", "lt", 0.0),
                                (
                                    "body_fraction",
                                    "le" if side == "up" else "ge",
                                    -body if side == "up" else body,
                                ),
                            )
                        )
    else:
        for side in (-1, 1):
            for body in (0.25, 0.5, 0.75):
                for range_edge in (1.0, 1.5):
                    candidates.append(
                        _rule(
                            "candle_geometry",
                            ("body_fraction", "ge" if side > 0 else "le", side * body),
                            ("range_to_atr14", "ge", range_edge),
                        )
                    )
        for volume_edge in (-0.5, 0.5, 1.5):
            for wick in (0.25, 0.5):
                for side in ("upper", "lower"):
                    candidates.append(
                        _rule(
                            "volume_range",
                            ("volume_z20", "ge", volume_edge),
                            (f"{side}_wick_fraction", "ge", wick),
                            ("range_to_atr14", "ge", 1.0),
                        )
                    )
    return candidates
