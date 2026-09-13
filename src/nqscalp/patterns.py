"""Frozen trajectory signatures independent of outcome-based state shrinkage."""

import hashlib
import json

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .features import match_definition

TRAJECTORY_COLUMNS = [f"seq{k}" for k in range(1, 13)] + [
    "body_signed",
    "range_atr",
    "compression5_20",
    "efficiency10",
]


def prototype_assignments(features, encoder):
    if encoder.get("columns") != TRAJECTORY_COLUMNS:
        raise ValueError("frozen trajectory columns differ")
    mean, scale, centers = (np.asarray(encoder[k], float) for k in ("mean", "scale", "centers"))
    if (
        mean.shape != (16,)
        or scale.shape != (16,)
        or centers.ndim != 2
        or centers.shape[1] != 16
        or len(centers) < 1
        or (scale <= 0).any()
        or not all(np.isfinite(x).all() for x in (mean, scale, centers))
    ):
        raise ValueError("invalid frozen trajectory encoder")
    values = features[TRAJECTORY_COLUMNS].to_numpy(float)
    valid = features.valid.to_numpy(bool) & np.isfinite(values).all(axis=1)
    indices = np.flatnonzero(valid)
    labels = np.full(len(features), -1, dtype=int)
    for start in range(0, len(indices), 4096):
        ix = indices[start : start + 4096]
        scaled = (values[ix] - mean) / scale
        labels[ix] = np.square(scaled[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
    return labels


def fit_prototypes(features, n_clusters=32, seed=20260911):
    if not isinstance(n_clusters, int) or not 2 <= n_clusters <= 128:
        raise ValueError("n_clusters must be an integer in 2..128")
    values = features.loc[features.valid, TRAJECTORY_COLUMNS].to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < n_clusters:
        raise ValueError("insufficient fit observations")
    if len(values) > 50_000:
        indices = np.random.default_rng(seed).choice(len(values), 50_000, replace=False)
        values = values[np.sort(indices)]
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(
            (values - mean) / scale
        )
    encoder = {
        "columns": TRAJECTORY_COLUMNS,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "centers": model.cluster_centers_.tolist(),
        "fit_rows": len(values),
        "seed": seed,
    }
    identity = hashlib.sha256(json.dumps(encoder, sort_keys=True).encode()).hexdigest()[:12]
    return [
        {
            "id": f"trajectory_{identity}_{k:02d}",
            "family": "trajectory",
            "kind": "prototype",
            "cluster": k,
            "encoder": encoder,
            "description": f"Frozen normalized price trajectory {k}",
        }
        for k in range(n_clusters)
    ]


def match_signature(features, signature):
    if signature.get("kind") == "prototype":
        cluster = signature["cluster"]
        if not isinstance(cluster, int) or not 0 <= cluster < len(signature["encoder"]["centers"]):
            raise ValueError("invalid prototype cluster")
        return prototype_assignments(features, signature["encoder"]) == cluster
    return match_definition(features, signature)
