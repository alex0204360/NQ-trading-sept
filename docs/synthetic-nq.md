# NQ synthetic market lab

Open `examples/orderbook_market_simulator.html` locally in a browser. It is self-contained and defaults to **NQ calibrated scenarios**, with prices in NQ points. Playback starts at **one completed candle per second**, adjustable from 0.1 to 500. Rendering is limited to 30 frames/second and does not set simulation speed. Pausing, changing speed, or returning from a background tab clears playback debt. A slow device can run below the requested rate; the overlay reports the measured rate.

The original Exact and Corrected Unity reconstruction modes remain selectable. They retain their original internal price units and mechanics. NQ mode is a separate stochastic OHLCV generator, not a rescaled Unity order book.

## Historical source and calibration

The included profile is fitted to **706,008 audited training bars**, from 2022-12-26 23:00 UTC through 2024-12-31 21:59 UTC. The raw version-1 Kaggle CSV was downloaded and its SHA-256 verified against `main`'s `reports/data_manifest.json`:

```text
1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b
```

The existing mechanical exclusions and calendar exclusions in `reports/data_quality.json` were replayed against that exact source. End stamps were shifted back one minute and converted from America/New_York to UTC using the existing pipeline convention. The resulting training partition has the recorded 706,008 rows. The 2025 real holdout was excluded from calibration and its outcomes were not scored.

The profile records one-minute log-return dispersion, lag-one correlation, upper/lower wick fractions, and log-volume statistics. Its volatility-persistence mapping, Student-t innovations, and scenario dynamics are explicit modeling assumptions. It is not a fitted intraday seasonal model, execution simulator, or validated replica of real NQ market microstructure. Source timestamp convention and contract rolls retain the existing audit limitations.

NQ prices follow a 0.25-point grid, with a $20/point contract multiplier ([CME specifications](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html)). The example starting level of 24,000 is configurable, not a current quote. Fees default to $4.50 round trip and one tick of slippage per side; these are editable assumptions, not broker quotes.

## Scenarios

Seeded paths support noise, up/down trends, mean-reverting ranges, compression, breakout, failed breakout, volatility shock, and mixed regimes. Episode durations and directions vary. These are designed stress cases; their frequencies and predictable drifts are not claims about actual NQ patterns. Noise is a useful negative control.

Use **Generate 1,000 bars** to build a preview without accelerating playback. It pauses and yields to the browser between batches. Export bars or supervised examples as JSONL. The browser retains at most 50,000 bars. Changing a scenario, calibration, price, or cost assumption starts a new path.

## Refit and generate training datasets

First acquire the historical training partition using the existing audited pipeline in `docs/reproduce.md`. A repository clone alone does not contain raw historical bars. For a new run:

```sh
python examples/calibrate_nq.py data/training.parquet data/new_nq_profile.json
node examples/generate_nq_dataset.js --profile data/new_nq_profile.json --output data/synthetic_run_01 --sequences 100 --bars 1000 --seed 10001 --horizon 10 --commission 4.5 --slippage 1
```

Alternatively, use the committed profile without downloading historical data:

```sh
node examples/generate_nq_dataset.js --output data/synthetic_run_01 --sequences 100 --bars 1000 --seed 10001
```

Node.js 20 or newer is recommended. The generator has no external Node dependencies. Calibration uses the repository's pandas/NumPy dependencies and accepts canonical CSV/JSONL/Parquet. It rejects naive timestamps, invalid bars, tick violations, and any bar at or after the fixed real holdout cutoff. It refuses to overwrite a profile; the dataset generator refuses existing output directories.

Each path has a unique seed and `sequence_id`. Entire sequences, not rows or overlapping windows, are allocated 70/15/15 to train, validation, and synthetic test. The default 100 x 1,000 run produces 100,000 bars and 93,000 examples at a ten-bar horizon, after 60-bar warmup and future-label trimming. Paths share calibration parameters but use independent random streams. Synthetic test paths are not a replacement for real-market holdout.

Files include `train_bars.jsonl`, `train_examples.jsonl`, corresponding validation/test files, a diagnostic `scenario_audit.jsonl`, and `manifest.json` containing calibration, configuration, checksums, split counts, and assumptions. Generated prices use artificial contiguous minute timestamps per sequence; this does not simulate exchange sessions, holidays, or contract rollovers. Do not concatenate independent paths into one continuous feed.

### Model-input contract

Only the `features` object in each example is model input. It uses the completed bar and at most 60 preceding bars: candle geometry, normalized trailing returns, ATR, trailing price/volume states, and prior-range breakout distances. Features cannot cross sequence, split, or minute-gap boundaries.

`targets` is a separate future-label object. A decision follows completed bar t, enters at open[t+1], and exits at close[t+horizon]. Long/short net dollars include two-sided tick slippage and round-trip fees for one NQ contract. The hindsight action label chooses long/short only if its future fixed-horizon payoff exceeds those costs, otherwise hold. It is not a live decision rule or globally optimal policy; overlapping labels are correlated and trades are not independently realizable.

Never use `targets`, timestamps/sequence IDs, seeds, or `scenario_audit.jsonl` as input features. Scenario membership is simulator information, unavailable in live trading. Do not randomly split overlapping examples or tune against the real 2025 holdout.

## Validation and intended next step

```sh
node --test tests/test_nq_synthetic.js
python -m unittest discover -s tests -p test_nq_calibration.py
```

Tests cover elapsed-time playback at different refresh rates, pause/background behavior, all scenario seeds/prefixes/tick geometry, future-feature isolation, label entry/exit and costs, and calibration holdout rejection.

This delivery creates a training and stress-testing dataset. It does not train a model or demonstrate profitability. The completed study on `main` found no confirmed pattern library. A model trained here can learn assumptions designed into the simulator. Establishing a real edge still requires frozen model/feature choices, chronological real-data validation, realistic trade scheduling and costs, drawdown analysis, and prospective paper testing. Keep synthetic-vs-real comparisons separate from the completed preregistered tournament artifacts.
