"""Reproducible exploratory rounds; heldout bytes are never loaded here."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nqpatterns.candidates import generate_candidates, match_candidate
from nqpatterns.features import compute_features
from nqpatterns.outcomes import first_passage, nonoverlap_indices
from nqpatterns.splits import HOLDOUT_CUTOFF, walk_forward_folds
from nqpatterns.statistics import PeriodObservations, block_bootstrap
from nqpatterns.tournament import append_event, canonical, write_json
from nqpatterns.validation import (
    choose_shortlist,
    point_estimates,
    resolution_summary,
    screen_reasons,
    stratum_codes,
    training_reasons,
)


def assert_committed(root, path):
    relative = str(Path(path).relative_to(root))
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=root, capture_output=True, check=False
    )
    if result.returncode or result.stdout != Path(path).read_bytes():
        raise ValueError("Manifest must be committed unchanged before evaluation")
    remote = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if remote.returncode:
        raise ValueError("Manifest commit must be published before evaluation")


class ResearchRunner:
    def __init__(self, root, require_committed=True):
        self.root = Path(root).resolve()
        self.require_committed = require_committed
        self.bars = pd.read_parquet(self.root / "data/training.parquet")
        if (self.bars.timestamp >= HOLDOUT_CUTOFF).any():
            raise ValueError("Training input contains heldout timestamps")
        self.schedule = pd.read_parquet(self.root / "data/schedule.parquet")
        self.features = compute_features(self.bars)
        self.folds = list(walk_forward_folds(self.bars))
        self.passages = {}
        self.code_cache = {}
        self.contexts = []
        self.retained = {}
        self.short_counts = {}
        utc_minutes = pd.DatetimeIndex(self.bars.timestamp).as_unit("ns").asi8 // 60_000_000_000
        self.grid = (utc_minutes % 45 == 0) & self.features.valid.to_numpy()
        for fold in self.folds:
            fit = self.features.iloc[fold.fit_indices]
            reference = fit.loc[fit.valid, "atr14_fraction"].to_numpy()
            if len(reference):
                cuts = np.quantile(reference, [1 / 3, 2 / 3]).tolist()
            else:
                cuts = [0.0, 0.0]
            strata, names = stratum_codes(
                self.bars.timestamp, self.features.atr14_fraction.fillna(0).to_numpy(), cuts
            )
            mask = (
                (self.bars.timestamp >= fold.validation_start)
                & (self.bars.timestamp < fold.validation_end)
            ).to_numpy()
            label_safe = (
                self.bars.timestamp + pd.Timedelta(minutes=45) < fold.validation_end
            ).to_numpy()
            schedule = self.schedule.loc[
                (self.schedule.market_close > fold.validation_start)
                & (self.schedule.market_open < fold.validation_end)
            ]
            self.contexts.append(
                {
                    "mask": mask,
                    "label_safe": label_safe,
                    "strata": strata,
                    "names": names,
                    "cuts": cuts,
                    "days": [str(x.date()) for x in schedule.index],
                    "period_id": f"wf_{fold.number}",
                }
            )

    def _provenance(self):
        paths = sorted((self.root / "src/nqpatterns").glob("*.py"))
        paths += [self.root / "data/training.parquet", self.root / "data/schedule.parquet"]
        freeze = self.root / "reports/pipeline_freeze.json"
        if freeze.exists():
            paths.append(freeze)
        return {
            str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }

    def prepare_round(self, number):
        path = self.root / f"results/round_{number:02d}/manifest.json"
        if path.exists():
            raise FileExistsError("Round manifest already exists")
        reference = self.bars.loc[self.bars.timestamp < self.folds[0].fit_end_exclusive]
        candidates = generate_candidates(reference, number)
        if len({c["family"] for c in candidates}) != 2:
            raise ValueError("Exactly two distinct approaches required")
        manifest = {
            "round": number,
            "candidates": candidates,
            "horizons": list(range(1, 46)),
            "magnitudes_points": [1, 2, 4, 8, 16, 32],
            "directions": ["up", "down"],
            "reference_fit_end_exclusive": self.folds[0].fit_end_exclusive.isoformat(),
            "baseline_cutpoints": [c["cuts"] for c in self.contexts],
            "evidence_status": "exploratory_reused_walk_forward",
            "definitions_sha256": hashlib.sha256(canonical(candidates).encode()).hexdigest(),
            "provenance": self._provenance(),
        }
        write_json(path, manifest)
        return manifest

    def _passage(self, magnitude):
        if magnitude not in self.passages:
            self.passages[magnitude] = first_passage(self.bars, magnitude, max_horizon=45)
        return self.passages[magnitude]

    def _codes(self, magnitude, horizon):
        key = (magnitude, horizon)
        if key in self.code_cache:
            return self.code_cache[key]
        p = self._passage(magnitude)
        codes = np.zeros(len(self.bars), dtype=np.int8)
        touch = p.first_touch_bars <= horizon
        for label, code in [("up", 1), ("down", 2), ("ambiguous_both", 3)]:
            codes[touch & (p.first_direction == label)] = code
        self.code_cache[key] = codes
        return codes

    def _populations(self, candidate):
        raw = match_candidate(self.features, candidate["signature"])
        retained = nonoverlap_indices(self.bars.timestamp, raw)
        self.retained[candidate["id"]] = retained
        selected = np.zeros(len(self.bars), dtype=bool)
        selected[retained] = True
        eligible = self._passage(1).eligible_45
        populations = []
        for context in self.contexts:
            period = context["mask"]
            safe = context["label_safe"] & eligible
            candidate_positions = np.flatnonzero(selected & period & safe)
            control_positions = np.flatnonzero(self.grid & period & safe)
            populations.append((candidate_positions, control_positions))
        fold_counts = []
        for context, (positions, controls) in zip(self.contexts, populations):
            period = context["mask"]
            r = int((raw & period).sum())
            k = int((selected & period).sum())
            fold_counts.append(
                {
                    "raw": r,
                    "retained_before_censoring": k,
                    "suppressed": r - k,
                    "eligible": len(positions),
                    "censored_or_boundary_excluded": k - len(positions),
                }
            )
        counts = {
            "folds": fold_counts,
            "raw_detections": int(raw.sum()),
            "retained_before_censoring": len(retained),
            "suppressed_detections": int(raw.sum()) - len(retained),
            "eligible_events": sum(len(a) for a, b in populations),
            "population_scope": "folds contain comparable WF counts; top-level raw counts span full training",
        }
        return populations, counts

    def _score(self, candidate, populations, counts, magnitude, horizon, direction):
        codes = self._codes(magnitude, horizon)
        target = 1 if direction == "up" else 2
        folds = []
        all_positions = []
        for context, (positions, controls) in zip(self.contexts, populations):
            row = point_estimates(
                codes[positions],
                context["strata"][positions],
                codes[controls],
                context["strata"][controls],
                target,
            )
            row["event_dates"] = int(self.bars.trading_date.iloc[positions].nunique())
            folds.append(row)
            all_positions.extend(positions.tolist())
        n = sum(r["sample_size"] for r in folds)

        def pooled(key):
            if not n or any(r[key] is None and r["sample_size"] for r in folds):
                return None
            return sum((r[key] or 0) * r["sample_size"] for r in folds) / n

        positions = np.asarray(all_positions, dtype=int)
        resolution = resolution_summary(
            codes[positions], self._passage(magnitude).first_touch_bars[positions], horizon, target
        )
        pattern_id = f"{candidate['id']}_m{magnitude}_h{horizon}_{direction}"
        row = {
            "pattern_id": pattern_id,
            "signature_id": candidate["id"],
            "family": candidate["family"],
            "direction": direction,
            "magnitude_points": magnitude,
            "magnitude_ticks": magnitude * 4,
            "horizon_bars": horizon,
            "sample_size": n,
            "folds": folds,
            "detection_counts": counts,
            "hit_rate": pooled("hit_rate"),
            "baseline_hit_rate": pooled("baseline_hit_rate"),
            "conservative_baseline_hit_rate": pooled("conservative_baseline_hit_rate"),
            "lift": pooled("lift"),
            "conservative_lift": pooled("conservative_lift"),
            "time_to_resolution": resolution,
            "mean_favorable_resolution": resolution["favorable"]["mean"],
            "outcome_counts": {
                name: int((codes[positions] == value).sum())
                for name, value in [
                    ("up", 1),
                    ("down", 2),
                    ("ambiguous_both", 3),
                    ("no_clear_outcome", 0),
                ]
            },
            "confidence_interval": None,
            "lift_confidence_interval": None,
            "evidence_status": "exploratory_reused_walk_forward",
        }
        key = (candidate["id"], horizon)
        if key not in self.short_counts:
            retained = self.retained[candidate["id"]]
            p = self._passage(magnitude)
            short_count = 0
            for context, fold in zip(self.contexts, self.folds):
                safe = (
                    self.bars.timestamp.iloc[retained] + pd.Timedelta(minutes=horizon)
                    < fold.validation_end
                ).to_numpy()
                short_count += int(
                    (
                        context["mask"][retained] & safe & (p.available_bars[retained] >= horizon)
                    ).sum()
                )
            self.short_counts[key] = short_count
        row["short_horizon_eligible_excluded_by_common45"] = self.short_counts[key] - n
        resolved = row["outcome_counts"]["up"] + row["outcome_counts"]["down"]
        row["resolved_hit_rate"] = row["outcome_counts"][direction] / resolved if resolved else None
        row["screen_reasons"] = screen_reasons(row)
        row["status"] = (
            "rejected_screen" if row["screen_reasons"] else "not_evaluated_inference_budget"
        )
        return row

    def _bootstrap(self, row, populations):
        codes = self._codes(row["magnitude_points"], row["horizon_bars"])
        target = 1 if row["direction"] == "up" else 2
        periods = []
        for context, (positions, controls) in zip(self.contexts, populations):

            def events(indices, context=context):
                return pd.DataFrame(
                    {
                        "day": self.bars.trading_date.iloc[indices].to_numpy(),
                        "stratum": np.asarray(context["names"])[context["strata"][indices]],
                        "favorable": (codes[indices] == target).astype(int),
                        "ambiguous": (codes[indices] == 3).astype(int),
                    }
                )

            periods.append(
                PeriodObservations(
                    events(positions), events(controls), context["days"], context["period_id"]
                )
            )
        return block_bootstrap(periods, n_resamples=9999)

    def evaluate_round(self, number):
        directory = self.root / f"results/round_{number:02d}"
        if (directory / "summary.json").exists():
            raise FileExistsError("A completed/partial round cannot be overwritten")
        path = directory / "manifest.json"
        if self.require_committed:
            assert_committed(self.root, path)
            for code in (self.root / "src/nqpatterns").glob("*.py"):
                assert_committed(self.root, code)
            assert_committed(self.root, self.root / "reports/pipeline_freeze.json")
        manifest = json.loads(path.read_text())
        candidates = manifest["candidates"]
        if manifest["provenance"] != self._provenance():
            raise ValueError("Frozen pipeline/data/code changed")
        if (
            hashlib.sha256(canonical(candidates).encode()).hexdigest()
            != manifest["definitions_sha256"]
        ):
            raise ValueError("Candidate definitions changed")
        rows = []
        populations = {}
        failures = []
        for candidate in candidates:
            try:
                population, counts = self._populations(candidate)
                populations[candidate["id"]] = population
                checkpoint = directory / "checkpoints" / f"{candidate['id']}.json"
                if checkpoint.exists():
                    rows.extend(json.loads(checkpoint.read_text()))
                    continue
                candidate_rows = []
                for magnitude in [1, 2, 4, 8, 16, 32]:
                    for horizon in range(1, 46):
                        for direction in ["up", "down"]:
                            candidate_rows.append(
                                self._score(
                                    candidate, population, counts, magnitude, horizon, direction
                                )
                            )
                write_json(checkpoint, candidate_rows)
                rows.extend(candidate_rows)
                print(
                    json.dumps(
                        {
                            "round": number,
                            "family": candidate["family"],
                            "signature": candidate["id"][:12],
                            "scored": 540,
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 -- log candidate failure and continue per protocol
                failure = {
                    "round": number,
                    "signature_id": candidate["id"],
                    "status": "candidate_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                append_event(self.root / "history/candidates.jsonl", failure)
        shortlist = choose_shortlist(rows)
        for row in shortlist:
            try:
                checkpoint = directory / "inference" / f"{row['pattern_id']}.json"
                if checkpoint.exists():
                    inference = json.loads(checkpoint.read_text())
                else:
                    inference = self._bootstrap(row, populations[row["signature_id"]])
                    write_json(checkpoint, inference)
                row.update(inference)
                row["gate_reasons"] = training_reasons(row)
                row["status"] = (
                    "provisional_survivor" if not row["gate_reasons"] else "rejected_inference"
                )
            except Exception as exc:  # noqa: BLE001 -- log candidate failure and continue per protocol
                row["status"] = "inference_error"
                row["gate_reasons"] = [f"{type(exc).__name__}: {exc}"]
            print(json.dumps({"bootstrap": row["pattern_id"], "status": row["status"]}), flush=True)
        lookup = {c["id"]: c for c in candidates}
        survivors = []
        scores_path = directory / "scores.jsonl"
        if scores_path.exists():
            scores_path.rename(
                directory / f"scores.interrupted.{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.jsonl"
            )
        history_path = self.root / "history/candidates.jsonl"
        seen = set()
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                event = json.loads(line)
                seen.add((event.get("round"), event.get("pattern_id"), event.get("status")))
        for row in rows:
            if row["status"] == "provisional_survivor":
                row["signature"] = lookup[row["signature_id"]]["signature"]
                survivors.append(row)
            append_event(directory / "scores.jsonl", row)
            if (number, row["pattern_id"], row["status"]) not in seen:
                append_event(
                    self.root / "history/candidates.jsonl",
                    {
                        "round": number,
                        "pattern_id": row["pattern_id"],
                        "status": row["status"],
                        "reasons": row.get("gate_reasons", row["screen_reasons"]),
                        "sample_size": row["sample_size"],
                        "artifact": f"results/round_{number:02d}/scores.jsonl",
                    },
                )
        scope = {
            "round": number,
            "status": "rejected_out_of_scope",
            "requested_horizon": 46,
            "reason": "rejected — out of scope: holding window exceeds 45 bars",
            "experiment": "boundary guard only; no long-horizon market scoring",
        }
        append_event(self.root / "history/candidates.jsonl", scope)
        from collections import Counter

        summary = {
            "round": number,
            "hypotheses_scored": len(rows),
            "signatures": len(candidates),
            "families": sorted({c["family"] for c in candidates}),
            "inference_shortlist": len(shortlist),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "survivors": survivors,
            "failures": failures,
            "evaluable": bool(rows),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(directory / "summary.json", summary)
        return summary
