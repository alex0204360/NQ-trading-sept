# Entry-economics audit and candidate experiment (exploratory only)

This is a non-mutating research instrument, not a qualified strategy. The uploaded Campaign 2 report (whose exact JS execution code/ledgers are not published on the accessible GitHub branches) reports 12/12 families negative after both base and stressed costs. The selected quiet-trend compression family has 3,263 trades and -$29,948.50 net; the directional subset has only 62 trades and does not establish an edge.

## Question

Are losses driven primarily by signal quality relative to modeled round-trip friction, or by bracket and holding policies? These are hypotheses rather than causal conclusions. The audit computes signal-time ATR14, 5/20-bar compression and prior-20 breakout status from completed candles, then records retrospective favorable/adverse movement before an exit and preserves actual simulated net P&L.

## Run it

Use ONLY the checkout containing the checksum-pinned pre-2025 data/training.parquet and data/schedule.parquet of the existing nqscalp project. The raw Kaggle CSV must not be substituted; it includes the sealed holdout. The Python module is deliberately outside src/nqscalp so the frozen research manifest's source-code hashes do not change.

    PYTHONPATH=src:. python -m tools.entry_audit \
      --root . \
      --trades reports/scalping/r001/trades/<one-existing-ledger>.csv \
      --output reports/diagnostics/entry-audit.json

The command verifies both frozen source hashes before reading the audited training bars. It creates a summary JSON and trade-detail CSV without modifying existing simulated trades. The ledger must use the same index and timestamp scheme as the audited data. The original training parquet is not committed to GitHub, so a full real-data diagnostic cannot be run from remote source alone.

## Guardrails

- Signal-time ATR, compression and breakout flags are causal. Excursions are separate post-entry outcomes and must NEVER be used as entry-time features.
- For stop/target/gap exits, the resolving candle's high/low are excluded from the excursion window, since the trade might have exited before those extremes. For a timed end-of-bar exit, its candle can be included. These excursions are conservative incomplete diagnostics, not realizable best fills.
- All simulated trade net P&L remains included even if its bar path crosses an unexplained gap. The result flags path-incomplete rows. Unresolved positions block the audit rather than disappearing from losses.
- ATR-to-friction bins (<2, 2-5, >=5) are descriptive, not optimized or traded. An apparent best bin is not a proven filter. Re-simulate any filter prospectively with occupancy changes, realistic next-bar fills, and drawdown.
- No 2025 data may be read for this exploratory research. Previously inspected 2024 cannot become fresh confirmation by being called validation.

## Next single fixed challenger, to register separately before scoring

Working hypothesis: a **compressed-range breakout followed by a meaningful directional close** may avoid low-quality entries. Compare against the unchanged quiet-trend control.

One fixed proposed specification (not claimed to be optimal): preceding five-bar high-low range <=45% of preceding twenty-bar high-low range; completed signal bar closes at least one NQ tick beyond the **preceding** twenty-bar extreme and finishes in the outer 25% of its high-low range in the breakout direction. Only 09:35-11:30 ET signal completions; no data gaps. Next-bar market entry; veto if opening gap is more than 4 points beyond the broken level. One NQ contract, fixed 10-point stop, 20-point target, 30-minute maximum, and no simultaneous positions. Use unchanged $10 round-trip base commission with one tick adverse slippage each side, plus stressed execution assumptions from the existing engine. Record veto reasons, event counts, every attempted order, net expectancy, quarters, drawdown, and day-cluster uncertainty.

Freeze this rule before any new outcome evaluation and keep it explicitly exploratory. A fixed parameter is a falsifiable proposal, NOT evidence that it will improve profitability. Positive pre-2025 numbers cannot justify opening the still-sealed 2025 test repeatedly.

Finally, the Campaign 2 report mentions code such as run_nq_campaign.js and extensive observation/decision ledgers, which were not found on these accessible branches as of this review. Do not silently equate the Campaign 2 engine with the different nqscalp simulator. Reconcile signal/bar timestamps, fills, and source hashes before transferring these diagnostics.
