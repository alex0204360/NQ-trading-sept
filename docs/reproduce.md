# Use and reproduce the project

Use Python 3.12. The dependency versions used for the reported run are pinned in `requirements.lock`. The repository contains source, model definitions, audits and derived statistics. It does not contain the raw Kaggle CSV or a live feed.

## Install

```sh
git clone https://github.com/alex0204360/NQ-trading-sept.git
cd NQ-trading-sept
python -m venv .venv
```

Activate `.venv` with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```sh
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

## Test the matcher without a data feed

```sh
python examples/match_bars.py examples/demo_library.json examples/demo_bars.jsonl --diagnostic
```

The fixture is invented data and an invented provisional rule. Its 160 bars produce three accepted diagnostic detections, with 97 further matches explicitly marked suppressed by the 45-minute cooldown. Without `--diagnostic`, this provisional fixture produces no matches. Neither result is market evidence.

To use the delivered research library with your own completed bars:

```python
from nqpatterns.matcher import LiveMatcher

matcher = LiveMatcher("patterns/library.json")
matches = matcher.update({
    "timestamp": "2026-01-05T14:30:00Z",
    "open": 24000.0,
    "high": 24002.0,
    "low": 23999.75,
    "close": 24001.0,
    "volume": 250,
    "completed": True,
})
print(matches)
```

These prices are format examples. Supply 60 preceding contiguous bars for warmup. Timestamps are timezone-aware interval starts, and each bar is supplied only after its minute has completed. Prices must respect the 0.25-point NQ grid. An optional `contract` marks roll boundaries. Out-of-order and malformed bars raise errors. Gaps restart feature warmup while preserving elapsed-time cooldown. `match_bars(iterable)` processes a batch using the same state and rules as repeated `update` calls. Use a new matcher instance for a separate replay.

The default library contains only confirmed patterns. An empty list of matches is expected when no pattern was confirmed. Historical statistics returned with a match describe the research sample, not live performance.

## Reproduce acquisition

Download the pinned Kaggle version yourself or let KaggleHub obtain it:

```sh
python -m nqpatterns.cli audit --source /path/to/Dataset_NQ_1min_2022_2025.csv
```

Omit `--source` to use KaggleHub. If authentication is required, set `KAGGLE_API_TOKEN` in your local environment. Never add credentials to repository files. The audit produces ignored `data/training.parquet`, `data/holdout.parquet`, and `data/schedule.parquet`, plus tracked audit/provenance reports. Compare the source checksum to `reports/data_manifest.json` before comparing results.

## Research checkpoints

`prepare-round N` records the exact candidate definitions, calibration cutpoints and code/data hashes. That manifest and the production code must be committed and pushed before `run-round N` will score them. Completed rounds cannot be overwritten. Per-signature checkpoints permit a partial round to resume with unchanged inputs; inference checkpoints preserve completed bootstrap calculations. A new study needs a new output history and a preregistered protocol, rather than edits to completed results.

```sh
python -m nqpatterns.cli prepare-round 1
# Review, commit and publish the manifest.
python -m nqpatterns.cli run-round 1
```

The published study already has these artifacts, so these commands will refuse to replace them. Raw data are deliberately separate from the repository; a clone alone cannot rerun research before acquisition. The operator applies `TournamentState` after each round and publishes its decision in `results/tournament_state.json` before preparing the next round.

`prepare-final` freezes at most 12 distinct-signature survivors with full-training baseline calibration. After that manifest is published, `validate-final` can open holdout exactly once. With no survivors, it writes an empty library and explicit no-survivor stage reports without reading heldout outcomes. A heldout access marker prevents a second evaluation after either completion or failure.

## Artifact navigation

- `results/round_*/manifest.json`: precise signatures and frozen input hashes.
- `results/round_*/summary.json` and `interpretation.json`: compact outcomes and reasons.
- `results/round_*/scores.jsonl`: one complete record per direction/magnitude/window hypothesis, including resolution distributions. These are large files; inspect them locally.
- `results/round_*/checkpoints/`: recoverable per-signature records from the screening pass; final inference statuses are in `scores.jsonl`.
- `history/candidates.jsonl`: compact append-only final statuses and references, including scope rejections.
- `patterns/library.json`: confirmed entries only, in the schema accepted by `LiveMatcher`.
- `reports/stage1.json`, `stage2.json`, `stage3.json`: terminal stage evidence.

No command connects a feed, submits orders, performs forward testing, or generates Pine Script.
