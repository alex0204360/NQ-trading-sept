"""Read-only baseline reproduction; never reads held-out bars or alters research outputs.

Run from repository root:
PYTHONPATH=src:.venv/lib/python3.12/site-packages python tools/diagnostics/reproduce.py
"""

import collections
import json
import subprocess
import time
from pathlib import Path

from nqpatterns.research import ResearchRunner

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/rebuild/recomputed.json"


def main():
    started = time.monotonic()
    runner = ResearchRunner(ROOT)
    result = {
        "purpose": "Recompute every frozen V1 candidate using current unchanged production engine",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": "PYTHONPATH=src:.venv/lib/python3.12/site-packages python tools/diagnostics/reproduce.py",
        "holdout_accessed": False,
        "training_bars": len(runner.bars),
        "valid_feature_rows": int(runner.features.valid.sum()),
        "rounds": [],
        "status": "running",
    }

    def save():
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for number in [1, 2, 3]:
        directory = ROOT / f"results/round_{number:02d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        archived = {
            r["pattern_id"]: r
            for r in map(json.loads, (directory / "scores.jsonl").read_text().splitlines())
        }
        current = runner._provenance()
        summary = {
            "round": number,
            "signatures": len(manifest["candidates"]),
            "scored": 0,
            "exact_matches": 0,
            "mismatches": [],
            "status_counts": {},
            "screen_reason_counts": {},
            "candidate_counts": [],
            "provenance_differences": sorted(
                k
                for k in set(current) | set(manifest["provenance"])
                if current.get(k) != manifest["provenance"].get(k)
            ),
        }
        statuses, reasons = collections.Counter(), collections.Counter()
        result["rounds"].append(summary)
        for candidate in manifest["candidates"]:
            population, counts = runner._populations(candidate)
            summary["candidate_counts"].append(
                {"id": candidate["id"], "family": candidate["family"], **counts}
            )
            for magnitude in manifest["magnitudes_points"]:
                for horizon in manifest["horizons"]:
                    for direction in manifest["directions"]:
                        row = runner._score(
                            candidate, population, counts, magnitude, horizon, direction
                        )
                        summary["scored"] += 1
                        statuses[row["status"]] += 1
                        reasons.update(row["screen_reasons"])
                        previous = archived[row["pattern_id"]]
                        if row == previous:
                            summary["exact_matches"] += 1
                        elif len(summary["mismatches"]) < 20:
                            summary["mismatches"].append(
                                {
                                    "pattern_id": row["pattern_id"],
                                    "fields": sorted(
                                        k
                                        for k in set(row) | set(previous)
                                        if row.get(k) != previous.get(k)
                                    ),
                                }
                            )
            summary["status_counts"] = dict(statuses)
            summary["screen_reason_counts"] = dict(reasons)
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save()
            print(
                json.dumps(
                    {
                        "round": number,
                        "scored": summary["scored"],
                        "exact_matches": summary["exact_matches"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    result["status"] = "complete"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save()


if __name__ == "__main__":
    main()
