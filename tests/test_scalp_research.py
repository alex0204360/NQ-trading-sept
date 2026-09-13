import numpy as np
import pandas as pd
import pytest

from nqscalp.patterns import fit_prototypes, match_signature
from nqscalp.research import CacheInvalid, load_cache, nonoverlapping, save_cache


def test_occupancy_ends_at_actual_exit_not_fixed_45_minutes():
    rows = pd.DataFrame(
        [
            {"detection_index": 0, "exit_index": 3, "termination_index": 3, "status": "closed"},
            {"detection_index": 1, "exit_index": 2, "termination_index": 2, "status": "closed"},
            {"detection_index": 3, "exit_index": 4, "termination_index": 4, "status": "closed"},
        ]
    )
    assert nonoverlapping(rows).detection_index.tolist() == [0, 3]


def test_unresolved_trade_stays_in_ledger_and_blocks_following_signal():
    rows = pd.DataFrame(
        [
            {
                "detection_index": 0,
                "exit_index": -1,
                "termination_index": 5,
                "status": "unresolved",
            },
            {"detection_index": 1, "exit_index": 2, "termination_index": 2, "status": "closed"},
        ]
    )
    assert nonoverlapping(rows).status.tolist() == ["unresolved"]


def test_cache_detects_truncation_and_identity_change(tmp_path):
    path = tmp_path / "payoff.npy"
    identity = {"source": "training", "code": "v1", "action": "long"}
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    save_cache(path, values, identity)
    np.testing.assert_array_equal(load_cache(path, identity), values)
    with pytest.raises(CacheInvalid, match="identity"):
        load_cache(path, {**identity, "code": "v2"})
    path.write_bytes(path.read_bytes()[:140])
    with pytest.raises(CacheInvalid, match="corrupt|size|digest"):
        load_cache(path, identity)


def test_prototype_matching_does_not_refit_using_future_rows():
    rng = np.random.default_rng(8)
    f = pd.DataFrame({f"seq{k}": rng.normal(size=100) for k in range(1, 13)})
    for col in ["body_signed", "range_atr", "compression5_20", "efficiency10"]:
        f[col] = rng.normal(size=100)
    f["valid"] = True
    signatures = fit_prototypes(f, n_clusters=4)
    for signature in signatures:
        np.testing.assert_array_equal(
            match_signature(f.iloc[:20], signature), match_signature(f, signature)[:20]
        )
    assert np.stack([match_signature(f, s) for s in signatures]).sum(axis=0).tolist() == [1] * 100
