# Phase 1: audited real NQ replay

Open `examples/orderbook_market_simulator.html`. All application code and audit
metadata are embedded in this one HTML file; there are no CDN dependencies.
Select **Real NQ · Kaggle CSV**, then choose the original CSV from dataset version 1:
https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025

The full CSV stays local and is parsed in a Web Worker. Its SHA-256 must match
`1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b`
to apply the main-branch audit exclusions. The audited replay has 706,008 training
bars, 12,051 quarantined source rows, and 330,516 excluded 2025 holdout bars.
Unknown CSVs receive basic validation, not a claim of calendar verification.
Canonical CSVs require timezone-aware interval-start `timestamp` and OHLCV fields.

The source ET timestamps are inferred end stamps, converted to UTC interval starts
by subtracting one minute. The publisher has not verified that convention or the
continuous-contract roll method. Historical calendar completeness is not certified.
These limitations from main remain unresolved, not silently repaired.

The chart has right-hand price labels, bottom time labels, colored candle volume,
crosshair OHLCV, ET/UTC display, wheel zoom, drag pan, replay-position seeking and
UTC date navigation. The playback slider controls candles per elapsed second in
all modes; CSV playback is not capped at 50,000 bars. Replay cannot infer tick-level
fills or an order book from minute OHLCV. Synthetic modes use an explicitly
artificial one-minute timeline. This is an independent TradingView-style renderer,
not TradingView's product or charting library.

Run `node --test tests/test_nq_synthetic.js tests/test_nq_chart_viewer.js`.
Set `NQ_RAW_CSV` to the original file path to also run the full-source audit test.
The distributable viewer remains one HTML; the JS source is kept separately only
for tests and maintenance, and tests check that the embedded code is identical.

## Phase 2 research gate

Proceed only after this real-data viewer is validated. Generate synthetic stress
regimes separately from real samples. Define patterns using past-only features;
record counts, outcomes, costs, baseline rates and uncertainty rather than promising
an 80% success rate. An observed 8,000/10,000 is a conditional sample statistic,
not a guaranteed next prediction or evidence of profitability by itself.

Iterate only inside pre-2025 chronological development folds with horizon purging,
day/sequence-aware uncertainty and multiple-testing controls. Compare net expectancy,
drawdown and turnover after fees/slippage, not accuracy alone. Freeze rules before
the final 2025 holdout test; do not repeatedly mine that holdout. Synthetic success
does not establish a real market edge.
