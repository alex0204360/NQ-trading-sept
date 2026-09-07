# MNQ/NQ Pattern-Mining & Strategy Engine: Phase 0 plan

Version: 0.1.0. Prepared September 6, 2026, before implementation or data download.

Status update, September 7: the original plan commit `152f83bad72e9af30260be467eb55aa210b7d1f0` is published to `alex0204360/NQ-trading-sept` and was verified remotely before implementation. User-authorized Git authentication resolved the earlier publication block. Implementation is now in progress. The preregistered protocol below is unchanged; dated execution status belongs in the README and stage reports. See `phase0-status.md` for the access history.

## 1. Objective and acceptance

Discover whether NQ one-minute OHLCV contains repeatable directional first-passage patterns within 1 through 45 consecutive minutes. Deliver a reproducible Python research package, complete candidate ledger, structured pattern library, and a bar-input matching module. A rigorously reported empty library is a valid scientific outcome. NQ evidence does not automatically validate MNQ prices, volume, fills, or profitability.

The user has authorized the complete project, incremental GitHub publication, and discrete sub-agents. Before any source code, tests, executable configuration, or downloader is written, the Phase 0 plan must be committed and published to the new project repository. Do not use an unrelated existing repository as the project destination.

No live or delayed feed, broker connection, order execution, Pine Script, or forward test will be built. Phase 3 Stage 3 is explicitly out of scope. The later feed approach is dependency injection: callers supply completed standard OHLCV bars to the matcher.

## 2. ECC workflow and provenance

Source: [affaan-m/ECC](https://github.com/affaan-m/ECC), inspected at commit `e04ea0b9cc8248686edf5ac751cadff550e162b8`, whose commit date is September 3, 2026.

Read before implementation: root `AGENTS.md`, `agents/planner.md`, `commands/plan.md`, `rules/common/development-workflow.md`, `rules/common/git-workflow.md`, and skills `search-first`, `tdd-workflow`, `verification-loop`, and `plan-orchestrate`. The last is a prompt-generation workflow and is not being invoked as an execution system. No ECC runtime hooks or installed plugin are claimed.

Apply requirements analysis, research before implementation, an explicit architecture and dependency order, test-first work, independent review, verification, and descriptive conventional commits. Use the available sub-agent facility for delegated roles. Preserve RED and GREEN evidence in separate reachable commits, with at least 80% line and branch coverage for the Python package before delivery. Adapt end-to-end testing to a Python CLI and streamed bars; no browser application is part of this project. The user's existing authorization covers the requested implementation and ordinary verification; no additional plan-approval checkpoint is being introduced.

Team knowledge belongs in `docs/`; result provenance belongs in `history/`. No credentials belong in source, documents, logs, commit messages, or result artifacts. Kaggle authentication, if needed, uses a runtime secret through `KAGGLE_API_TOKEN`; no literal token is preserved in this project.

## 3. Data contract and audit

Source: [Kaggle dataset](https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025), downloaded with the user's requested KaggleHub dataset handle after the plan is published. Pin the resolved dataset version, filenames, byte sizes, SHA-256 hashes, download time, and package versions in a manifest. Keep raw market data outside Git; commit the manifest, audit, and derived results. Check source licensing before redistributing source rows.

Publisher-stated metadata, not yet verified against downloaded bytes: December 26, 2022 at 18:01 through December 11, 2025 at 20:52, US Eastern time; CSV; 1,048,575 rows. That count equals an Excel worksheet's row limit minus one and raises a possible export-truncation question. It does not prove truncation. A statement of no nulls does not establish complete sessions or correct prices. Exact columns, provenance, bar timestamp convention, and contract-roll construction require verification.

Canonical completed bar fields: timezone-aware `timestamp`, `open`, `high`, `low`, `close`, `volume`. Timestamp denotes the one-minute interval's start; a bar is available only at interval end. Instrument, tick size, source timezone, and source timestamp convention are explicit dataset/matcher metadata. Optional contract identifiers are preserved if present. NQ tick size is 0.25 index points, per [CME contract specifications](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html).

Audit requirements and handling:

- Preserve an immutable raw copy; record source row numbers and every exclusion or transformation.
- Parse source ET timestamps using `America/New_York`, then store UTC. Do not silently interpret naive times as UTC. Quarantine unresolved daylight-saving ambiguities and invalid dates; no guessed offsets. If timezone or timestamp convention remains unknown, label the run blocked for confirmatory inference until a documented interpretation is frozen.
- Require finite positive OHLC, high at least max(open, close, low), low at most min(open, close, high), and finite nonnegative volume. Quarantine invalid bars. Do not invent volume or impute missing prices.
- Report tick-grid violations. Investigate whether they reflect adjustment methodology before rejecting a whole series.
- Sort by timestamp only after preserving source order information. Collapse exactly identical duplicates with an audit entry; quarantine all conflicting bars at a duplicate timestamp. Never select whichever price is favorable.
- Enumerate gaps and contiguous segments. A holding window may not cross a missing minute, contract change, session break, or unresolved data problem. Do not compress elapsed time by treating the next observed bar as the next minute.
- Compare observed sessions to a documented CME calendar. Distinguish scheduled closures from unexplained gaps and unavailable sessions; an unresolved calendar comparison remains a stated limitation.
- Flag unusual ranges or jumps above 20 times the trailing 60-bar median true range; retain plausible market extremes in the primary data and report sensitivity separately. No return winsorization or outcome-based row deletion.
- Preserve contract boundaries when known. Unknown continuous-contract adjustment or roll rules remain a provenance limitation; do not claim transportability to raw live contracts without resolving it.

All primary features require at most 60 prior completed bars in the same contiguous segment. Mark warmup bars unavailable, with counts. Audit decisions affecting results must be frozen before candidate scoring; any later correction is an explicit amendment with invalidated runs identified.

## 4. Chronological split and leakage controls

Exact holdout cutoff: `2025-01-01T00:00:00-05:00`, equivalent to `2025-01-01T05:00:00Z`.

Training is every valid earlier bar in the requested dataset. Held-out history is every valid bar at or after the cutoff, through the dataset's actual endpoint. Neither prices, outcomes, candidate prevalence, nor performance in that period may influence discovery. Source manifest and split/audit operations may parse all rows mechanically; do not expose held-out summary statistics to candidate agents. Record this infrastructure access separately from the one-time model evaluation. If source coverage cannot support this split, log a data-coverage block and amend the plan before scoring, without selecting a cutoff based on returns.

Four expanding walk-forward folds, with boundaries in America/New_York:

| Fold | Fit/calibration data ends before | Validation interval, start inclusive and end exclusive |
| --- | --- | --- |
| 1 | 2023-04-01 | 2023-04-01 to 2023-07-01 |
| 2 | 2023-07-01 | 2023-07-01 to 2024-01-01 |
| 3 | 2024-01-01 | 2024-01-01 to 2024-07-01 |
| 4 | 2024-07-01 | 2024-07-01 to 2025-01-01 |

Each fit starts at the actual beginning of the training data. Fit scalers, bins, clusters, and any learned candidate parameters only on that fold's fit portion. Exclude fit events whose future label window reaches the validation boundary; apply a 45-minute purge before each validation interval. Validation features may use preceding completed bars for warmup, but cannot use future validation bars. Use explicit date intervals rather than assuming equally spaced rows, consistent with the limitations in [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Reusable walk-forward folds are exploratory because tournament rounds may adapt to their results. Multiple-testing adjustment alone does not erase this adaptation. Do not call a training-selected pattern confirmed. For per-pattern cluster scores in this delivery, fit the signature scaler and reference centroids using only the initial fitting period before April 1, 2023, and keep those artifacts unchanged across all four folds and the holdout. Symbolic rule definitions also remain identical across folds. Calibration of the baseline expands with each fold. Later rounds may propose different specifications using training results, but they must refit reference artifacts on that initial fitting period and label all reused-fold scores exploratory. Never pool unrelated cluster IDs. An optional family-level procedure diagnostic may refit within each fold, but its pooled support cannot qualify an individual pattern for the library.

## 5. Outcome labeling, before scoring

Parameter grid: every integer horizon H from 1 through 45 bars, crossed with thresholds of 1, 2, 4, 8, 16, and 32 NQ points, or 4, 8, 16, 32, 64, and 128 ticks. Both up and down directions are eligible. These are research thresholds, not asserted achievable or profitable trade targets.

Detection occurs after completed bar t. Reference price is the next consecutive bar's open, E. Inspect bars t+1 through t+H inclusive, with symmetric barriers E+M and E-M. H=1 includes the next bar's intrabar range. Do not enter at the detection bar's close using its subsequent movement. This is an idealized reference-price convention and does not establish a fill at E.

Label the first barrier reached: `up`, `down`, `no_clear_outcome`, `ambiguous_both`, or `censored_data`. Exact touches count as reached. On each future bar inspect its open first: an opening price beyond one barrier establishes that first direction. Otherwise, if high and low cross both barriers on the first touching bar, order is unknown and the label is `ambiguous_both`. Never guess intrabar order. No touch in a complete window is `no_clear_outcome`. An incomplete window is `censored_data`, not no-clear or a loss. For primary comparability, use a common population with all 45 consecutive future bars available; separately report how many shorter-horizon events this excludes. Availability exclusions are retrospective data-completeness filters and cannot be used as live signals.

For a fixed signature, process otherwise valid completed-bar detections chronologically and keep the first, then suppress further detections for a full 45-minute interval regardless of actual resolution. Apply this causal schedule before retrospective future-completeness filtering; an event whose outcome is censored still consumes its scheduled interval. Use the same event schedule for every magnitude/horizon pair and direction. Record raw, suppressed, retained, and outcome-eligible counts separately. Event thinning is per signature; different patterns can share events and are not independent strategies. An accepted detection at t permits another detection at t+45 minutes, whose reference entry starts at t+46; their outcome bars do not overlap.

Primary `hit_rate` = confirmed favorable first-hit count divided by all eligible retained signals. Keep no-clear and ambiguous labels and their counts; their inclusion in this probability denominator does not classify them as losing trades. Also report conditional `resolved_hit_rate` = favorable divided by favorable plus unfavorable, when the denominator is nonzero. Report coverage, no-clear, ambiguity, censorship, and suppression rates explicitly.

For every scored hypothesis report mean, standard deviation, median, p25, p75, p90, minimum, maximum, and exact 1..H histogram of resolution bars, both for all unambiguous resolutions and for favorable resolutions. No-clear cases are right-censored at H and excluded from resolution-time means. Report cumulative resolved fraction of all eligible events by bar, so a fast conditional mean cannot hide a large unresolved population. Empty distributions and rates are null with counts, never fabricated zeros.

Reject any requested H below 1 or above 45 before scoring. H>45 receives status `rejected_out_of_scope` and reason `rejected — out of scope: holding window exceeds 45 bars`. A boundary test at H=46 is a scope test, not a historical experiment. Do not assert a long-horizon pattern would have been significant unless it was actually tested in a separately authorized study.

## 6. Baselines, support, uncertainty, and statistical gates

Primary baseline is a signal-independent 45-minute UTC grid of eligible reference events within each validation period, using the identical barriers, time windows, and data-completeness rules. Match calendar year-quarter, four ET detection-time buckets (00:00-06:00, 06:00-12:00, 12:00-18:00, 18:00-24:00), and trailing 14-bar mean true-range/close tertiles, giving 12 strata per quarter. Fit tertile cutpoints on past fit data only; freeze full-training cutpoints before holdout. Weight each stratum's baseline probability by the candidate's event distribution in that period. Require at least 30 control events in each occupied stratum in each fold; otherwise that hypothesis fails baseline support. Shared candidate/control events are permitted and their dependence is preserved during resampling. These strata reduce time/regime mismatch but do not establish causal attribution to the pattern itself.

Report both ordinary lift and conservative lift. Conservative lift = candidate confirmed-favorable probability minus matched baseline probability counting both confirmed-favorable and ambiguous-both as favorable for the baseline. Candidate ambiguities thus count against the claimed edge while baseline ambiguities count in its favor. This conservative lift is the sole primary hypothesis test. The 50% symmetric-direction probability is descriptive only, since unresolved outcomes make 50% an inappropriate automatic unconditional baseline.

Use joint moving-block resampling of candidate and controls, retaining within-day order and all observations. A trading date is the CME session date under the frozen calendar; complete no-event trading dates remain in the resampling frame. Recompute stratum weights and both probabilities for each resample. Use 5-consecutive-trading-day blocks and a prespecified 20-day-block sensitivity, sampled from admissible consecutive blocks without wrapping across period boundaries; truncate the concatenated blocks to the period's original day count. Resample separately within each fold for pooled walk-forward statistics and within the holdout for its result. Use 9,999 replicates per block length for exploratory intervals and 99,999 per block length for frozen holdout tests/intervals. The fixed root seed is 20260906; all hypotheses in a given period/block length share the same resampled day indices.

Report the envelope of the two 95% percentile bootstrap intervals for primary probabilities and conservative lift. P-values use the centered bootstrap null distribution and a plus-one correction, with one-sided alternative conservative lift >0. For each hypothesis take the larger p-value from the two block lengths before Holm correction. Require positive lower lift bounds for both block lengths. Undefined stratum support in any replicate fails that hypothesis's inference gate; do not discard unfavorable or undefined replicates and silently reduce the bootstrap sample. Degenerate probability distributions are also an explicit inference failure.

These are approximate dependence-aware inferences requiring sufficiently stable, weakly dependent blocks, not distribution-free guarantees. Report their assumptions and both block-length results. Binomial/Wilson intervals may be displayed as descriptive diagnostics, but cannot replace clustered uncertainty.

Every Phase 2 survivor must satisfy all of these exploratory gates:

| Gate | Fixed requirement |
| --- | --- |
| Eligible sample | At least 1,000 nonoverlapping events across the four validation folds |
| Fold support | At least 100 events and 20 event-bearing dates in each fold |
| Matched baseline | At least 30 controls in each occupied fold/stratum |
| Minimum edge | Pooled conservative lift at least 0.05, or 5 percentage points |
| Stability | Positive conservative lift in at least 3 of 4 folds; no fold below -0.02 |
| Interval exclusion | Pooled 95% clustered lower bound for conservative lift above zero |
| Probability precision | Primary hit-rate 95% interval full width at most 0.10 |
| Lift precision | Conservative-lift 95% interval full width at most 0.15 |

These deliberately stringent support requirements can reject real but rare patterns; failure means insufficient evidence under this study, not proof of no possible effect.

Holdout gates apply to each of at most 12 hypotheses frozen before opening it:

| Gate | Fixed requirement |
| --- | --- |
| Eligible sample | At least 1,000 events on at least 100 event-bearing dates and 40 ISO weeks |
| Matched baseline | At least 30 controls in every occupied holdout stratum |
| Minimum edge | Conservative lift at least 0.05 |
| Precision | Primary hit-rate interval width at most 0.10; lift interval width at most 0.15 |
| Statistical evidence | One-sided Holm-adjusted p-value at most 0.05 and 95% clustered lower lift bound above zero |
| Dependence sensitivity | Positive lower lift bounds for both 5-day and 20-day blocks; maximum p-value used before Holm |

Holm adjustment covers every frozen submitted hypothesis, including failures and low-support hypotheses assigned p=1. All pattern/direction/threshold/window variations count separately; no substitution after evaluation. Holm is available in [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html). Selected-model intervals remain marginal unless explicitly labeled simultaneous; adjusted tests are the basis for the family-level significance decision. n<30 always receives an explicit `low_confidence` flag even though the confirmation support gate is much higher.

## 7. Tournament: finite search and stop conditions

Maximum four rounds. Each round enters exactly two distinct candidate approaches and records their definitions before evaluation, satisfying the requested minimum of two. Individual candidate failures do not stop the other candidate. A family may produce up to 32 signatures per round, yielding at most 34,560 direction/magnitude/horizon hypotheses per round across two families. Every generated hypothesis, insufficient-support screen, exception, and scope rejection goes in the append-only ledger. Unattempted configurations are not labeled statistical failures.

First calculate all candidates' support, outcome counts, point-estimate lifts, and fold stability. Among those passing these inexpensive gates, rank by descending pooled conservative lift, descending support, ascending mean favorable resolution time, then canonical ID. Select the first hypothesis for each distinct signature, then at most 12 signatures per approach, 24 per round, for bootstrap scoring. Adjacent threshold/window variants cannot consume the entire shortlist. Entries excluded by this budget or signature-selection rule are `not_evaluated_inference_budget`, not statistical rejects. This finite inference budget is part of the search design. Bootstrap only this shortlist; do not attempt expensive resampling for every grid configuration. Every retained library hypothesis must still pass the full interval gates.

| Round | Approach A | Approach B |
| --- | --- | --- |
| 1 | Trailing candle-shape clusters with fixed initial-period scaling/centroids | Price indicator-state combinations, including trailing standardized price and trend state |
| 2 | Causal swing/breakout geometry from past extrema | Bar-volume and range signatures; do not call OHLCV volume-at-price data |
| 3 | Deepen the best prior eligible family | Deepen the next distinct eligible family, or a predeclared alternate causal sequence family if none passed |
| 4 | Refine short-horizon structure in one prior family | Refine a different prior family under the same scoring gates |

For rounds 3-4, ranking and any new search definition use training results only and must be committed before running that round. Each generator records an exact finite signature/configuration manifest before evaluation. Do not change statistical gates or expand H to produce winners. Previously evaluated identical hypothesis/data/model hashes reuse their recorded results and do not masquerade as new independent evidence.

A library-quality score is the largest positive 95% lower conservative-lift bound among eligible cumulative survivors, with empty library score zero. Define Q0=0 and Qr as the maximum of Q(r-1) and the qualifying scores discovered in round r. Improvement is Qr minus Q(r-1), so it is never measured against a weaker replacement library. Preserve all provisional survivors in round artifacts; a later cumulative summary may flag unstable or superseded entries without erasing history.

Stop at the first applicable condition:

1. Fixed limit: finish round 4, then enter the common finalization path below.
2. Diminishing returns: assess improvement beginning with round 2. Stop when two successive statistically evaluable rounds each improve library-quality score by less than 0.01, or 1 percentage point. The earliest such stop is after round 3. An improvement of at least 0.01 resets the counter. A round containing only candidate errors is logged as an execution failure and leaves this counter unchanged, but still consumes its fixed round slot.
3. Early quality attempt: after at least two completed rounds, if at least three distinct signatures clear the training gates with conservative lift at least 0.08, freeze finalists and open the holdout once. The quality stop succeeds only if at least three distinct frozen signatures pass all holdout gates, each with conservative lift at least 0.07. If it fails, stop as `holdout_spent_quality_gate_not_reached`; never resume tuning against that holdout.
4. Operational block: log and checkpoint a required input/access/publication failure. Retry a transient external failure at most twice. Do not continue multiple unpublished rounds when incremental publishing is unavailable. Resume a blocked stage only after the missing capability is restored; it is not evidence of a losing pattern.

These rules deliberately prevent repeated Stage 2 peeking. A held-out evaluation is terminal whether it passes or fails. Every statistical stop uses one common finalization path: if no survivor exists, produce the empty library and leave holdout unopened; otherwise freeze finalists and evaluate once. Retain any individually confirmed patterns even if the early three-pattern quality target is missed. Record all simultaneously satisfied stop conditions and one primary reason, prioritizing a completed holdout decision, then fixed round limit, then diminishing returns.

Before any holdout access, select at most 12 surviving hypotheses deterministically by descending lower conservative-lift bound, descending support, ascending mean favorable resolution time, then canonical ID. Select at most one direction/magnitude/horizon combination per exact signature. Preserve the qualified signature artifacts unchanged; do not refit a cluster into a different pattern after selection. Freeze exact definitions, pipeline version, artifact hashes, full-training baseline cutpoints, and IDs in a committed manifest. Full-training fit statistics on these exact frozen signatures are descriptive and separate from walk-forward statistics.

## 8. Validation and outputs

Stage 1: verify causal feature computation with prefix-invariance and future-perturbation tests; validate schema and the 45-bar ceiling; report the exact frozen signatures' descriptive training fit and exploratory walk-forward evidence separately.

Stage 2: expose heldout data only to the evaluator after verifying the frozen manifest commit. Record the first evaluation access time and code/data hashes. Evaluate all submitted hypotheses, retain all results, and produce one table with event counts, primary and resolved hit rates, baselines, lifts, confidence intervals, adjusted p-values, outcome frequencies, and resolution-time distributions. Failed holdout patterns remain in history and are excluded from the confirmed library. If no training pattern survives, produce an empty library and `not_run_no_survivors` Stage 2 artifact; the heldout stays unopened.

Stage 3: produce an explicit `out_of_scope_not_run` status artifact. No feed is connected and no forward evidence is claimed.

The matcher accepts a sequence of completed canonical bars or one new bar at a time. It validates timestamps, OHLCV, instrument metadata, ordering, one-minute spacing, and warmup, and resets feature state at gaps/contract changes. It shares the exact causal feature/signature code, frozen artifacts, and per-signature 45-minute detection suppression policy with historical evaluation. A gap does not retroactively erase an already accepted event's elapsed-time suppression. It returns matched pattern IDs, detection timestamp, historical statistics with training/holdout provenance, and confirmation status. No network or broker library is required. Default matching uses only confirmed patterns; an empty confirmed library returns an explicit empty match list. An optional diagnostic mode can expose provisional or suppressed raw matches with their status clearly identified; such raw matches are not the event population represented by historical statistics.

Each library entry contains schema version, pattern ID, family, machine-readable signature and learned artifact references, direction, magnitude in points and ticks, horizon, nonoverlap policy, feature requirements, training/WF/holdout sample sizes, primary and conditional hit rates, confidence method and bounds, baseline/lift, adjusted test result, outcome counts, resolution distributions, data/model hashes, and status. Historical sample size is never presented as a count of live matches.

## 9. Implementation sequence and artifacts

At preregistration, no files in the following implementation rows existed. This table records the planned dependency order; the README reports current execution status.

| Dependency | Work and locations | Delegation/review | Risk and verification |
| --- | --- | --- | --- |
| New GitHub repository available | Publish this plan, initial `CHANGELOG.md`, `docs/phase0-status.md`, `history/events.jsonl` | Root publisher; planning review | Verify remote plan commit before any code |
| Plan published | `pyproject.toml`, dependency lock, `src/nqpatterns/data.py`, `schema.py`, `splits.py`, `reports/data_quality.json` | Data pipeline agent; independent Python review | High: timezone, gaps, rolls; test audit accounting and immutable raw data |
| Pipeline audited | `outcomes.py`, `tests/test_outcomes.py`, scope-rejection ledger | Label/TDD agent; validation reviewer | High: H=1/H=45/H=46, double touches, opening gaps, no-clear, censorship |
| Labels verified | `features.py`, `candidates/`, `statistics.py`, `validation.py`, `tournament.py`, finite round manifests | Separate candidate and validation agents | High: leakage, dependence, multiple testing; future-perturbation and synthetic known-outcome tests |
| Each round complete | `results/round_01/` through attempted rounds, `history/candidates.jsonl`, provisional libraries, changelog | Root loop operator/publisher; reviewer | Publish candidates before run and results afterward; persist exceptions |
| Stop/freeze reached | `results/frozen_manifest.json`, Stage 1/2/3 reports, `patterns/library.json` | Independent heldout evaluator | High: immutable finalists and one-time holdout; include every reject |
| Frozen schema/features available | `matcher.py`, standard-bar example fixture, matcher tests | Matcher agent; independent reviewer | High: batch/stream equivalence, gaps, missing bars, empty library |
| All attempted stages logged | README with actual numbers and limits; verification and authorship history | Documentation/review agent; root | No fabricated results, stale changelog, secrets, or unpushed commits |

Use established packages for data frames/arrays, clustering, statistical primitives, and testing, with exact versions locked after environment verification. Custom code owns market-data audit rules, chronological orchestration, first-passage semantics, hypothesis provenance, and the matcher. Read primary package documentation before using version-dependent APIs.

Tests must establish behavior rather than mirror implementation: all outcome edge cases; one-minute elapsed-time enforcement; timezone/DST failure handling; duplicate/gap audit accounting; split embargo and withheld-data access; identical features under streaming and batch calculation; fixed-seed reproducibility; no future influence; known synthetic directional and null data; multiple-comparison correction including failures; empty-library behavior; and persisted exception/stop artifacts. A small synthetic end-to-end workflow verifies mechanics only and must never be presented as NQ market evidence.

Every work unit follows test-first RED, minimal GREEN, review, and a changelog update. Conventional commit examples are `docs: preregister NQ research protocol`, `test: specify gap-aware outcome labels`, `feat: label first barrier outcomes within 45 minutes`, `feat: register round 1 candidate families`, `docs: publish round 1 results and rejection reasons`, and `docs: publish frozen holdout validation`. Keep genuine chronological commits; do not backdate or manufacture iteration.

## 10. Completion standard and current limitation

Completion requires a new accessible GitHub project; plan published before code; audited real dataset; at least two approaches in every attempted tournament round; labels and full hypothesis history; stop reason; frozen historical validation or explicit no-survivor skip; feed-agnostic matcher; tests and review evidence; all deliverables pushed with remote SHAs verified. The README must distinguish demonstrated historical numbers, model-selection effects, provenance assumptions, untested transaction costs/slippage, unknown forward behavior, and unverified NQ-to-MNQ transfer.

The original Phase 0 publication block was resolved on September 7. The original plan was verified on GitHub before implementation and data download began. Statistical gates remain unchanged. Consult current stage artifacts for achieved scope and measured results.
