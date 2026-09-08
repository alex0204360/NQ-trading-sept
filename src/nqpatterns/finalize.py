"""Immutable, one-shot historical finalization with explicit empty-library output."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import match_candidate
from .features import compute_features
from .outcomes import first_passage, nonoverlap_indices
from .research import assert_committed
from .splits import HOLDOUT_CUTOFF
from .statistics import PeriodObservations, block_bootstrap, holm_adjust
from .tournament import append_event, freeze_finalists, verify_freeze, write_json
from .validation import holdout_reasons, point_estimates, resolution_summary, stratum_codes


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_finalization(root, state):
    """Freeze cumulative finalists and past-only calibration; never open holdout."""
    root = Path(root).resolve()
    path = root / "results/frozen_manifest.json"
    if path.exists():
        raise FileExistsError("A frozen evaluation manifest cannot be replaced")
    rows = []
    for summary in sorted((root / "results").glob("round_*/summary.json")):
        rows.extend(json.loads(summary.read_text())["survivors"])
    provenance = {
        "stop_reason": state.get("stop_reason"),
        "state": state,
        "baseline_cutpoints": None,
        "input_hashes": {},
        "source_hashes": {},
        "confirmatory_metadata_ready": False,
    }
    pipeline = root / "reports/pipeline_freeze.json"
    if pipeline.exists():
        provenance["pipeline"] = json.loads(pipeline.read_text())
        provenance["confirmatory_metadata_ready"] = provenance["pipeline"].get(
            "confirmatory_metadata_ready", False
        )
        provenance["input_hashes"]["reports/pipeline_freeze.json"] = _hash(pipeline)
    for source in sorted((root / "src/nqpatterns").glob("*.py")):
        provenance["source_hashes"][str(source.relative_to(root))] = _hash(source)
    if rows:
        bars = pd.read_parquet(root / "data/training.parquet")
        if (bars.timestamp >= HOLDOUT_CUTOFF).any():
            raise ValueError("Training file contains heldout timestamps")
        features = compute_features(bars)
        reference = features.loc[features.valid, "atr14_fraction"].to_numpy()
        if not len(reference):
            raise ValueError("No valid training observations for frozen baseline calibration")
        provenance["baseline_cutpoints"] = np.quantile(reference, [1 / 3, 2 / 3]).tolist()
        for name in ("data/training.parquet", "data/schedule.parquet", "data/holdout.parquet"):
            provenance["input_hashes"][name] = _hash(root / name)
    return freeze_finalists(rows, path, provenance)


def apply_confirmation(rows, metadata_ready):
    """Adjust every submitted hypothesis, then apply statistical and provenance gates."""
    pvalues = []
    for row in rows:
        value = row.get("p_value", 1.0)
        prechecks = holdout_reasons({**row, "adjusted_p_value": 0.0})
        pvalues.append(value if not prechecks else 1.0)
    adjusted = holm_adjust(pvalues)
    results = []
    for source, pvalue in zip(rows, adjusted):
        row = {**source, "adjusted_p_value": pvalue}
        reasons = holdout_reasons(row)
        row["statistical_pass"] = not reasons
        if not reasons and not metadata_ready:
            reasons.append("unverified_data_provenance")
            row["status"] = "unconfirmed_provenance"
        else:
            row["status"] = "rejected_holdout" if reasons else "confirmed"
        row["gate_reasons"] = reasons
        row["low_confidence"] = row.get("sample_size", 0) < 30
        results.append(row)
    return results


def _score_period(bars, schedule, finalist, cuts, start=None, seed_detection=None, bootstrap=False):
    features = compute_features(bars)
    raw = match_candidate(features, finalist["signature"])
    period = (
        np.ones(len(bars), dtype=bool) if start is None else (bars.timestamp >= start).to_numpy()
    )
    raw &= period
    if seed_detection is not None:
        # A training event seeds cooldown but is excluded from heldout estimates.
        raw |= (bars.timestamp == seed_detection).to_numpy()
    retained = nonoverlap_indices(bars.timestamp, raw)
    passage = first_passage(bars, finalist["magnitude_points"], max_horizon=45)
    positions = retained[period[retained] & passage.eligible_45[retained]]
    minute = pd.DatetimeIndex(bars.timestamp).as_unit("ns").asi8 // 60_000_000_000
    controls = np.flatnonzero(
        (minute % 45 == 0) & features.valid.to_numpy() & period & passage.eligible_45
    )
    codes = np.zeros(len(bars), dtype=np.int8)
    within = passage.first_touch_bars <= finalist["horizon_bars"]
    for label, code in [("up", 1), ("down", 2), ("ambiguous_both", 3)]:
        codes[within & (passage.first_direction == label)] = code
    strata, names = stratum_codes(
        bars.timestamp, features.atr14_fraction.fillna(0).to_numpy(), cuts
    )
    target = 1 if finalist["direction"] == "up" else 2
    row = point_estimates(
        codes[positions], strata[positions], codes[controls], strata[controls], target
    )
    days = pd.to_datetime(bars.trading_date.iloc[positions])
    row.update(
        pattern_id=finalist["pattern_id"],
        event_dates=int(days.nunique()),
        iso_weeks=len({tuple(x.isocalendar()[:2]) for x in days}),
        time_to_resolution=resolution_summary(
            codes[positions], passage.first_touch_bars[positions], finalist["horizon_bars"], target
        ),
        outcome_counts={
            label: int((codes[positions] == code).sum())
            for label, code in [
                ("up", 1),
                ("down", 2),
                ("ambiguous_both", 3),
                ("no_clear_outcome", 0),
            ]
        },
        detection_counts={
            "raw_detections": int((raw & period).sum()),
            "retained_before_censoring": int(period[retained].sum()),
            "eligible_events": len(positions),
            "suppressed_detections": int((raw & period).sum()) - int(period[retained].sum()),
        },
        status="descriptive_training_fit",
        confidence_interval=None,
        lift_confidence_interval=None,
    )
    if bootstrap:

        def events(indices):
            return pd.DataFrame(
                {
                    "day": bars.trading_date.iloc[indices].to_numpy(),
                    "stratum": np.asarray(names)[strata[indices]],
                    "favorable": (codes[indices] == target).astype(int),
                    "ambiguous": (codes[indices] == 3).astype(int),
                }
            )

        frame = schedule.loc[
            (schedule.market_close > start) & (schedule.market_open <= bars.timestamp.iloc[-1])
        ]
        observation = PeriodObservations(
            events(positions), events(controls), [str(d.date()) for d in frame.index], "holdout"
        )
        row.update(block_bootstrap([observation], n_resamples=99999, block_lengths=(5, 20)))
    last_detection = bars.timestamp.iloc[retained[-1]] if len(retained) else None
    return row, last_detection


def evaluate_finalization(root):
    """Verify published freeze, mark irreversible access, and evaluate at most once."""
    root = Path(root).resolve()
    path = root / "results/frozen_manifest.json"
    assert_committed(root, path)
    manifest = verify_freeze(path)
    provenance = manifest["provenance"]
    for section in ("input_hashes", "source_hashes"):
        for relative, expected in provenance[section].items():
            if _hash(root / relative) != expected:
                raise ValueError(f"Frozen input changed: {relative}")
    finalists = manifest["finalists"]
    stage1 = {
        "status": "not_run_no_survivors",
        "patterns": [],
        "walk_forward_evidence": "exploratory reused training validation",
        "causality_verification": "prefix and future-perturbation regression tests",
    }
    stage2 = {
        "status": "not_run_no_survivors",
        "patterns": [],
        "holdout_opened": False,
        "frozen_manifest_sha256": manifest["content_sha256"],
    }
    stage3 = {"status": "out_of_scope_not_run", "feed_connected": False}
    library = {
        "schema_version": 1,
        "instrument": "NQ",
        "bar_interval_minutes": 1,
        "timestamp_convention": "interval_start",
        "tick_size": 0.25,
        "patterns": [],
        "stop_reason": provenance["stop_reason"],
        "frozen_manifest_sha256": manifest["content_sha256"],
    }
    if finalists:
        marker = root / "history/heldout_access.json"
        if marker.exists():
            raise FileExistsError(
                "Heldout evaluation already accessed; no second evaluation permitted"
            )
        training = pd.read_parquet(root / "data/training.parquet")
        schedule = pd.read_parquet(root / "data/schedule.parquet")
        cuts = provenance["baseline_cutpoints"]
        seeds = {}
        for finalist in finalists:
            fit, seed = _score_period(training, schedule, finalist, cuts)
            stage1["patterns"].append(
                {
                    "pattern_id": finalist["pattern_id"],
                    "training_fit": fit,
                    "walk_forward": finalist,
                }
            )
            seeds[finalist["pattern_id"]] = seed
        stage1["status"] = "completed_descriptive_fit"
        write_json(root / "reports/stage1.json", stage1)
        access = {
            "status": "started",
            "opened_at_utc": datetime.now(UTC).isoformat(),
            "frozen_manifest_sha256": manifest["content_sha256"],
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create also protects against concurrent evaluation processes.
        with marker.open("x") as stream:
            json.dump(access, stream)
        try:
            holdout = pd.read_parquet(root / "data/holdout.parquet")
            if holdout.empty or (holdout.timestamp < HOLDOUT_CUTOFF).any():
                raise ValueError("Heldout partition is empty or contains training timestamps")
            combined = pd.concat([training.tail(60), holdout], ignore_index=True)
            rows = []
            for finalist in finalists:
                try:
                    row, _ = _score_period(
                        combined,
                        schedule,
                        finalist,
                        cuts,
                        start=HOLDOUT_CUTOFF,
                        seed_detection=seeds[finalist["pattern_id"]],
                        bootstrap=True,
                    )
                except Exception as exc:  # noqa: BLE001 -- preserve terminal failure evidence
                    row = {
                        "pattern_id": finalist["pattern_id"],
                        "status": "evaluation_error",
                        "p_value": 1.0,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
            rows = apply_confirmation(rows, provenance["confirmatory_metadata_ready"])
            stage2.update(
                status="completed",
                patterns=rows,
                holdout_opened=True,
                data_sha256=_hash(root / "data/holdout.parquet"),
            )
            for finalist, row, fit in zip(finalists, rows, stage1["patterns"]):
                if row["status"] == "confirmed":
                    library["patterns"].append(
                        {
                            **finalist,
                            "status": "confirmed",
                            "historical_stats": {
                                "training_fit": fit["training_fit"],
                                "walk_forward": finalist,
                                "holdout": row,
                            },
                            "nonoverlap_minutes": 45,
                            "feature_warmup_bars": 60,
                        }
                    )
                append_event(root / "history/candidates.jsonl", {"stage": "heldout", **row})
            if provenance["stop_reason"] == "early_quality_attempt":
                strong = sum(
                    r["status"] == "confirmed" and r["conservative_lift"] >= 0.07 for r in rows
                )
                library["stop_reason"] = (
                    "quality_gate_reached"
                    if strong >= 3
                    else "holdout_spent_quality_gate_not_reached"
                )
        except Exception as exc:  # noqa: BLE001 -- preserve terminal failure evidence
            stage2.update(
                status="evaluation_failed_holdout_spent",
                holdout_opened=True,
                reason=f"{type(exc).__name__}: {exc}",
            )
            append_event(root / "history/events.jsonl", {"stage": "heldout", **stage2})
    for number, report in enumerate((stage1, stage2, stage3), 1):
        write_json(root / f"reports/stage{number}.json", report)
    write_json(root / "patterns/library.json", library)
    return {"stage1": stage1, "stage2": stage2, "stage3": stage3, "library": library}
