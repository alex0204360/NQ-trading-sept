# Prospective net-profit rebuild protocol

2026-09-11. Recovered from the interrupted, unpushed design. Commit this before production implementation. The user authorizes implementation; no additional plan approval is required. Follow the ECC test-first/review workflow pinned in `research.plan.md`. The new goal supersedes the original no-simulator scope and empty-library completion rule, but does not authorize a broker/feed connection or real trades.

## Architecture and corrections

Build independent package `nqscalp`. Existing `nqpatterns` and original results remain frozen references. Active commands use the new package. Reuse audited OHLCV/calendar inputs, not V1 directional-lift gates or V2 globally shrunk state tables.

| Diagnosed issue | Correction | Regression check |
| --- | --- | --- |
| Directional labels omit economic payoff | Actual entries/SL/TP/timeouts/net P&L | Exact prices, costs, balances |
| Fixed45 suppression wastes opportunities after exits | Actual order/position occupancy | Reentry only after exit, no overlap |
| Sparse control strata reject candidates | Economic support/uncertainty and unseen confirmation | Net-return gates, not hit-rate proxy |
| Broad state scores suppress every supported action | Conditional event grammar and richer price trajectories | Full event-to-trade counts, two representations |
| Gaps/unknown source construction | Preserve unresolved paths and provenance | Never delete losses or fill at pre-gap close |
| Truncated cache | Atomic writes, identity/schema/size/digest validation | Truncation and changed-code cache tests |
| Interrupted handoff | Shared research/matcher/replay logic, checkpointed workflow | Reproducible commands and ledger checks |

## Data and causal features

Standard bars: timezone-aware UTC interval-start timestamp and finite positive tick-aligned OHLC, finite nonnegative volume; unique increasing one-minute timestamps. Bars become available one minute after start. Gaps/contract changes reset warmup; never impute prices. Feature validity requires current completed bar plus60 preceding contiguous bars. No future completeness filter may decide live eligibility.

Features: normalized cumulative returns r1/r2/r3/r5/r10/r20/r40, seq1..seq12=(current close minus lagged close)/ATR14, signed body, wick fractions, range/ATR, prior5/20/60 extrema, prior5/prior20 range compression, efficiency10, position20, volume/current prior20mean, and ET session bucket. Every learned scale/centroid uses fit data only.

Round1: 48 symbolic signatures, six families times two directions times all/open/midday/late contexts. Families: momentum continuation, pullback continuation, compression breakout, breakout, failed-break rejection, exhaustion reversal. Exact conditions are saved before scoring. Second representation:32 standardized trajectory prototypes on seq1..12 plus body_signed/range_atr/compression5_20/efficiency10, fit deterministic at most50,000 initial rows, seed20260911. Fixed centroids, no test-period refitting.

## Execution and risk

One pending order or position at a time. Initial market orders activate at the next consecutive bar open. Stop/target offsets attach to reference open; actual entry is slipped adversely. Stop touches count; target requires one-tick penetration and fills at target without favorable gap improvement. Both exits are modeled market-on-trigger with adverse slippage. Opening stop gap fills at worse opening price. Without known opening ordering, a bar touching both exits loses at stop. Timeout exits at Hth bar close. H is integer1..45 and expiry uses elapsed minutes, not compressed row counts. H>45 receives `rejected_out_of_scope` without claiming longer-horizon evidence.

Record detection/activation/entry/exit/expiry, entry/exit reference and slipped prices, bracket levels, reason, holding bars/minutes, fees/slippage, reference gross points, actual fill P&L, net points/dollars, adverse excursion, balance. Detection at the resolving bar's close can submit the next order; earlier detections are suppressed.

Calendar eligibility uses only known schedule breaks/closes and period end; exiting exactly at close is allowed. Unexpected gaps leave accepted trades unresolved, not retrospectively excluded or exited at the previous close. Preserve them and fail qualification. Earlier actual exits remain priced despite later gaps. Subsequent recovery can resume diagnostics, but unresolved P&L cannot become zero.

Instrument: NQ tick0.25, $20/point, one contract, hypothetical initial$50,000. [CME specifications](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html) verified2026-09-11. Primary fee allowance$5/side plus1tick adverse slippage per market fill, equivalent1point roundtrip. Stress2ticks/side=1.5points; severe3ticks=2points. [Broker pricing](https://ninjatrader.com/pricing/) confirms commissions are per side and exclude additional exchange/clearing fees; our allowance is an explicit assumption, not a user quote. Slippage includes spread, not double-charged. Subscriptions/taxes are separate. No MNQ fill claim follows from NQ data.

Planned loss including primary costs <=$500; no sizing optimization/pyramiding. Gaps may exceed planned loss and cannot be capped in P&L. Require closed-equity and conservative intratrade drawdown each <=$5,000. Compute adverse excursion from actual entry excluding fees; add fees to the conservative account bound. No account margin or future-profit guarantee is implied.

## Search and selection

Pre2025 history is extensively examined development data. Initial fit ends before2024-01-01T05:00Z with45minute purge. Validation is2024, four ET calendar quarters. No fit selection uses validation outcomes; subsequent use of validation to guide rounds is explicitly exploratory.

Initial grid: TP{4,8,12,20,32}, SL{4,8,12,20}, H{1,3,5,10,15,20,30,45}, directions both unless event fixes direction.320 unrestricted choices/signature. Simulate actual occupancy for each choice on fit data. Require100 closed trades and20dates for calibration. Select at most2 actions/signature by mean net minus1.645 day-cluster SE; tie shorterH/smallerSL/smallerTP/ID. Log all attempts and support failures. Negative supported ranks stay diagnostic, not actionable recommendations.

Validate selected actions. Development finalist requires150 closed trades,40dates,12 event-bearing weeks,20trades per quarter,3/4 quarters net-positive, zero unresolved executions/accounting/overlap violations, mean>=0.5points, positive two-tick stress net P&L, both drawdowns<=$5,000, and positive lower95% day-block expectancy intervals. Rank by smaller interval lower bound across5/20day blocks, then mean,dates,ID. Freeze at most2 exact signature/action definitions for confirmation. Reused-development intervals are ranking aids, never confirmatory significance.

Round2 must use two approaches and be separately registered based on Round1 diagnosis: refine events/prototypes or investigate structural swing stops, volatility-scaled exits, and contingent stop/pullback entry. Expand selected definitions to every H1..45. Contingent entries require one-tick limit penetration, pending expiry<=3bars, no favorable gap improvement, and no entry-bar target credit when post-entry ordering is unknown. Dedicated tests must precede their use. Further representations require prospective manifests, reasons and full search counts. Do not relax fees/risk retrospectively to qualify failures.

## Unseen evidence and acceptance

2025 has had mechanical audit/split access only, no pattern outcomes. Reserve H1=[2025-01-01T05:00Z,2025-07-01T04:00Z), H2=[2025-07-01T04:00Z,2025-12-12T05:00Z). Filtered reads must expose only the selected block. Before access commit and verify publication of candidates, source/config/model/input hashes and ranking. Append access record before loading outcomes. No replacements after viewing results. Failed H1 becomes development; H2 remains untouched until a new freeze. Once both are used, new unseen data is required for another claim.

Each exact submitted definition independently must pass:150 closed trades,40dates,12 event weeks; mean>=0.5points and positive totalnet; positive2tick stress; zero unresolved/accounting/overlap violations; bothdrawdowns<=$5,000; positive three predefined subwindow totals with20trades each. H1 subwindows Jan-Feb/Mar-Apr/May-Jun; H2 Jul-Aug/Sep-Oct/Nov-endpoint. n<30 explicitly low-confidence.

Jointly resample daily net and trade counts including scheduled zero-event dates, with5/20day moving blocks,19,999replicates,seed20260911. Require positive lower95% intervals for both, and use the larger centered-null one-sided p. Holm across every submitted definition, assigning failures p1, alpha0.025 per block, at most0.05 overall. Ordinary95% intervals alone do not establish family-wise significance. Candidate portfolio mixing is not independently validated; default matcher activates one confirmed definition based on frozen development priority. Historical qualification remains conditional on unresolved data-source construction, not verified live fills. No measured forward performance exists.

## Delivery and workflow

Files: immutable `research/` manifests; `reports/scalping/` outcomes, ledgers, risk and profit; `history/scalping.jsonl` stages/agents/failures; `patterns/scalping-library.json` qualifying exact definitions. Include pattern ID/signature/direction/entry/SL/TP/H/sample/win rate/interval/net expectancy/time-to-resolution histogram and quantiles. Actionable matcher response includes supporting cutoff and counts, quoted offsets, activation-resolved levels, expiry and expectednet. Otherwise provide no-trade reason.

One reproducible workflow command and one standard-OHLCV matcher command; replay uses the same matching and execution logic. Historical artifacts are never relabeled forward performance. Commit RED tests before GREEN production, independent review, >=80% new-package line/branch coverage, lint/build and actual real-data replay. Root owns commits; each delegated artifact is attributed. Save after each stage and verify pushes.

Successful stop requires unseen profitability acceptance and delivery verification. Failed rounds trigger prospectively registered alternatives. A session interruption or60minute compute checkpoint records objective `unfinished`, completed experiments and the next concrete experiment; it is not a conclusion that no edge exists or fulfillment by an empty library.
