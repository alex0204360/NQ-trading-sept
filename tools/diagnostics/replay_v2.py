"""Read-only replay of the separate, existing V2 checkout's frozen policies."""

import hashlib
import json
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT.parent / "nq-profitability-research"
sys.path.insert(0, str(V2 / "src"))
import numpy as np
import pandas as pd
from nqprofit.features import compute_features
from nqprofit.policies import encode_states
from nqprofit.replay import decision_indices, feasible_horizons, replay_policy
from nqprofit.research import BOUNDARIES, _baseline

started = time.monotonic()
manifest = json.loads((V2 / "reports/input_manifest.json").read_text())
input_status = {}
for name, expected in manifest["input_hashes"].items():
    source = ROOT / "data" / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    v2_source = V2 / "data" / name
    input_status[name] = {
        "used": str(source),
        "sha256": expected,
        "v2_copy_sha256": hashlib.sha256(v2_source.read_bytes()).hexdigest(),
    }
bars = pd.read_parquet(ROOT / "data/training.parquet")
assert (bars.timestamp < BOUNDARIES[-1]).all()
schedule = pd.read_parquet(ROOT / "data/schedule.parquet")
features = compute_features(bars)
indices = decision_indices(bars, features)
close_times = pd.DatetimeIndex(bars.timestamp) + pd.Timedelta(minutes=1)
report = {
    "holdout_accessed": False,
    "training_bars": len(bars),
    "valid_features": int(features.valid.sum()),
    "opportunities": len(indices),
    "folds": [],
    "status": "running",
    "models_refit": False,
    "input_status": input_status,
}
output = ROOT / "reports/rebuild/v2-replay.json"
for family in ["regime8", "prototype8", "regime16", "prototype16"]:
    encoder = json.loads((V2 / f"models/{family}.encoder.json").read_text())
    states = np.full(len(bars), -1, dtype=int)
    states[indices] = encode_states(features.iloc[indices], close_times[indices], encoder)
    for fold, (start, end) in enumerate(pairwise(BOUNDARIES), 1):
        model = json.loads((V2 / f"models/{family}.fold{fold}.json").read_text())
        validation = indices[(close_times[indices] >= start) & (close_times[indices] < end)]
        horizons = feasible_horizons(bars, schedule, end)
        replay = replay_policy(bars, features, validation, horizons, states, model)
        baseline = replay_policy(
            bars, features, validation, horizons, np.zeros(len(bars), dtype=int), _baseline(model)
        )
        scores = []
        for state in model["states"]:
            scores.append(
                {
                    "state_id": state["state_id"],
                    "supported": state["supported"],
                    "sample_size": state["sample_size"],
                    "date_count": state["date_count"],
                    "maximum_ranking_score": max(state["ranking_scores"])
                    if state["ranking_scores"]
                    else None,
                }
            )
        row = {
            "family": family,
            "fold": fold,
            "policy": {k: v for k, v in replay.items() if k != "trades"},
            "baseline": {k: v for k, v in baseline.items() if k != "trades"},
            "state_scores": scores,
        }
        report["folds"].append(row)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "state_scores"}), flush=True)
report["status"] = "complete"
report["elapsed_seconds"] = round(time.monotonic() - started, 3)
output.write_text(json.dumps(report, indent=2) + "\n")
