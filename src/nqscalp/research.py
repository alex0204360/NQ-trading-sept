"""Registered, resumable development experiments. Never reads held-out prices."""

import hashlib
import itertools
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import acceptance, day_evidence, summarize
from .execution import ExecutionConfig, TradeSpec, prepare_bars, simulate_batch
from .features import calendar_horizons, compute_features, event_definitions
from .patterns import fit_prototypes, match_signature, prototype_assignments

FIT_END = pd.Timestamp("2024-01-01T05:00Z")
TEST_START = pd.Timestamp("2025-01-01T05:00Z")
INPUT_HASHES = {
    "training.parquet": "1722334651520716155d7d14aca334342319e4431c084b5d6be8d24bd31b6f08",
    "schedule.parquet": "fccb7ac0bc7b5c709f76ca5528b303b861b42ef8d78c6d36703252771c9c24a3",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with tmp.open("w") as stream:
        stream.write(raw + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


class CacheInvalid(ValueError):
    pass


def save_cache(path, values, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    write_json(
        path.with_suffix(".meta.json"),
        {
            "identity": identity,
            "size": path.stat().st_size,
            "sha256": digest(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        },
    )


def load_cache(path, identity):
    path = Path(path)
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta["identity"] != identity:
            raise CacheInvalid("cache identity changed")
        if path.stat().st_size != meta["size"] or digest(path) != meta["sha256"]:
            raise CacheInvalid("corrupt cache size or digest")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != meta["shape"] or str(array.dtype) != meta["dtype"]:
            raise CacheInvalid("corrupt cache schema")
        return array
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, CacheInvalid):
            raise
        raise CacheInvalid(f"corrupt cache: {exc}") from exc


def nonoverlapping(outcomes):
    if outcomes.empty:
        return outcomes.copy()
    frame = outcomes.sort_values("detection_index", kind="stable")
    indices = frame.detection_index.to_numpy()
    termination = frame.termination_index.to_numpy()
    status = frame.status.to_numpy()
    chosen, occupied = [], -1
    for position, index in enumerate(indices):
        if index < occupied:
            continue
        chosen.append(position)
        if status[position] != "rejected":
            occupied = max(index + 1, int(termination[position]))
    return frame.iloc[chosen].copy()


def source_hashes(root):
    return {
        p.name: digest(p)
        for p in sorted((Path(root) / "src/nqscalp").glob("*.py"))
        if p.name not in ("cli.py", "__main__.py", "matcher.py")
    }


def training_context(root):
    root = Path(root)
    for name, expected in INPUT_HASHES.items():
        if digest(root / "data" / name) != expected:
            raise ValueError(f"verified input hash differs: {name}")
    bars = pd.read_parquet(root / "data/training.parquet")
    if (bars.timestamp >= TEST_START).any():
        raise ValueError("development cannot contain held-out timestamps")
    schedule = pd.read_parquet(root / "data/schedule.parquet")
    return bars, schedule, compute_features(bars)


def calendar_dates(schedule, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = set()
    for row in schedule.itertuples():
        left, right = max(row.market_open, start), min(row.market_close, end)
        if left < right:
            left = left.tz_convert("America/New_York").normalize()
            right = (right - pd.Timedelta(nanoseconds=1)).tz_convert("America/New_York").normalize()
            days.update(pd.date_range(left, right, freq="D").strftime("%Y-%m-%d"))
    return sorted(days)


def prepare_experiment(root, experiment_id="r001"):
    root = Path(root).resolve()
    if not re.fullmatch(r"r\d{3}", experiment_id):
        raise ValueError("experiment ID must be rNNN")
    path = root / "research" / f"{experiment_id}.json"
    if path.exists():
        raise FileExistsError("registered experiment already exists")
    bars, _, features = training_context(root)
    fit = features.loc[bars.timestamp + pd.Timedelta(minutes=46) < FIT_END]
    signatures = event_definitions() + fit_prototypes(fit)
    actions = [
        asdict(TradeSpec(d, tp, sl, h))
        for d in (1, -1)
        for tp in (4, 8, 12, 20, 32)
        for sl in (4, 8, 12, 20)
        for h in (1, 3, 5, 10, 15, 20, 30, 45)
    ]
    manifest = {
        "experiment_id": experiment_id,
        "source_hashes": source_hashes(root),
        "input_hashes": INPUT_HASHES,
        "costs": asdict(ExecutionConfig()),
        "fit_end": FIT_END.isoformat(),
        "validation_end": TEST_START.isoformat(),
        "signatures": signatures,
        "actions": actions,
        "status": "registered_before_outcomes",
        "evidence": "exploratory_pre2025_reused_data",
        "holdout_accessed": False,
    }
    write_json(path, manifest)
    return manifest


def fit_statistics(trades):
    priced = trades.loc[trades.status.eq("closed")]
    values = priced.net_points.to_numpy(float)
    n = len(values)
    dates = (
        pd.DatetimeIndex(priced.entry_timestamp).tz_convert("America/New_York").strftime("%Y-%m-%d")
    )
    unique, inverse = np.unique(dates, return_inverse=True)
    mean = float(values.mean()) if n else None
    se = score = None
    if n and len(unique) > 1:
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        se = float(
            np.sqrt(len(unique) / (len(unique) - 1) * np.square(sums - mean * counts).sum()) / n
        )
        score = mean - 1.645 * se
    return {
        "n": n,
        "dates": len(unique),
        "mean_net_points": mean,
        "day_cluster_se": se,
        "score": score,
        "unresolved": int(trades.status.eq("unresolved").sum()),
        "rejected": int(trades.status.eq("rejected").sum()),
        "supported": n >= 100 and len(unique) >= 20,
    }


def _rank(row):
    a, s = row["action"], row["fit"]
    return (
        -(s["score"] if s["score"] is not None else -1e100),
        a["horizon"],
        a["stop_points"],
        a["target_points"],
        a["direction"],
    )


def write_trades(path, trades):
    frame = trades.copy()
    frame["balance_after_dollars"] = np.nan
    balance = 50_000.0
    for index, row in frame.iterrows():
        if row.status == "unresolved":
            balance = np.nan
        elif row.status == "closed":
            balance += row.net_dollars
        frame.loc[index, "balance_after_dollars"] = balance
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def run_experiment(root, experiment_id="r001", max_signatures=None):
    root = Path(root).resolve()
    manifest_path = root / "research" / f"{experiment_id}.json"
    manifest = json.loads(manifest_path.read_text())
    committed = subprocess.run(
        ["git", "show", f"HEAD:research/{experiment_id}.json"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if committed.returncode or committed.stdout != manifest_path.read_bytes():
        raise ValueError("commit registered manifest before evaluation")
    if manifest["source_hashes"] != source_hashes(root):
        raise ValueError("registered source changed; use a prospective amended experiment")
    if manifest["input_hashes"] != INPUT_HASHES:
        raise ValueError("registered input differs")
    bars, schedule, features = training_context(root)
    prepared = prepare_bars(bars)
    availability = bars.timestamp + pd.Timedelta(minutes=1)
    fit_mask = (availability < FIT_END - pd.Timedelta(minutes=45)).to_numpy()
    val_mask = ((availability >= FIT_END) & (availability < TEST_START)).to_numpy()
    fit_h = calendar_horizons(bars, schedule, FIT_END)
    val_h = calendar_horizons(bars, schedule, TEST_START)
    days = calendar_dates(schedule, FIT_END, TEST_START)
    boundaries = [
        pd.Timestamp(t, tz="America/New_York").tz_convert("UTC")
        for t in ("2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01")
    ]
    output = root / "reports/scalping" / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    identity = digest(manifest_path)
    reports, new_count = [], 0
    labels = None
    for number, signature in enumerate(manifest["signatures"]):
        checkpoint = output / "checkpoints" / (signature["id"] + ".json")
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
            if record["manifest_sha256"] != identity:
                raise ValueError("checkpoint identity differs")
            reports.append(record)
            continue
        if max_signatures is not None and new_count >= max_signatures:
            break
        print(
            f"{experiment_id} signature {number + 1}/{len(manifest['signatures'])}: {signature['id']}",
            flush=True,
        )
        if signature.get("kind") == "prototype":
            if labels is None:
                labels = prototype_assignments(features, signature["encoder"])
            hits = labels == signature["cluster"]
        else:
            hits = match_signature(features, signature)
        evaluations = []
        for action in manifest["actions"]:
            if signature.get("direction", action["direction"]) != action["direction"]:
                continue
            positions = np.flatnonzero(hits & fit_mask & (fit_h >= action["horizon"]))
            outcomes = simulate_batch(prepared, positions, TradeSpec(**action))
            trades = nonoverlapping(outcomes)
            evaluations.append(
                {
                    "action": action,
                    "fit": fit_statistics(trades),
                    "raw_events": int((hits & fit_mask).sum()),
                    "eligible": len(positions),
                    "occupied_suppressed": len(outcomes) - len(trades),
                }
            )
        chosen = sorted([r for r in evaluations if r["fit"]["supported"]], key=_rank)[:2]
        validation = []
        for candidate in chosen:
            action = candidate["action"]
            candidate_id = (
                signature["id"]
                + "_"
                + hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()[:12]
            )
            positions = np.flatnonzero(hits & val_mask & (val_h >= action["horizon"]))
            trades = nonoverlapping(simulate_batch(prepared, positions, TradeSpec(**action)))
            trades["pattern_id"] = candidate_id
            for key, value in action.items():
                trades[key] = value
            summary = summarize(trades, days)
            evidence = day_evidence(trades, days)
            windows = []
            for start, end in itertools.pairwise(boundaries):
                subset = trades.loc[
                    (trades.entry_timestamp >= start) & (trades.entry_timestamp < end)
                ]
                windows.append(summarize(subset, calendar_dates(schedule, start, end)))
            reasons = acceptance(summary, evidence, windows)
            if candidate["fit"]["score"] <= 0:
                reasons.append("nonpositive_fit_rank")
            if candidate["fit"]["unresolved"]:
                reasons.append("unresolved_fit_executions")
            record = {
                "pattern_id": candidate_id,
                "signature_id": signature["id"],
                "family": signature["family"],
                "action": action,
                "fit": candidate["fit"],
                "summary": summary,
                "evidence": evidence,
                "windows": windows,
                "reasons": reasons,
                "status": "development_finalist" if not reasons else "rejected",
                "raw_events": int((hits & val_mask).sum()),
                "eligible": len(positions),
                "accepted_orders": len(trades),
                "occupied_suppressed": len(positions) - len(trades),
            }
            write_trades(output / "trades" / (candidate_id + ".csv"), trades)
            validation.append(record)
        record = {
            "signature_id": signature["id"],
            "manifest_sha256": identity,
            "fit_evaluations": evaluations,
            "validation": validation,
            "status": "completed" if chosen else "no_supported_fit_action",
        }
        write_json(checkpoint, record)
        reports.append(record)
        new_count += 1
        write_json(
            output / "progress.json",
            {
                "completed_signatures": len(reports),
                "total_signatures": len(manifest["signatures"]),
                "objective": "unfinished",
                "holdout_accessed": False,
            },
        )
    candidates = [c for report in reports for c in report["validation"]]
    finalists = sorted(
        [c for c in candidates if not c["reasons"]],
        key=lambda c: (
            -c["evidence"]["lower95"],
            -c["summary"]["mean_net_points"],
            -c["summary"]["trade_dates"],
            c["pattern_id"],
        ),
    )[:2]
    result = {
        "experiment_id": experiment_id,
        "manifest_sha256": identity,
        "status": "completed" if len(reports) == len(manifest["signatures"]) else "checkpoint",
        "objective": "unfinished",
        "evidence": "exploratory_development_only",
        "holdout_accessed": False,
        "completed_signatures": len(reports),
        "fit_actions": sum(len(r["fit_evaluations"]) for r in reports),
        "candidates": candidates,
        "finalists": finalists,
    }
    write_json(output / "summary.json", result)
    return result
