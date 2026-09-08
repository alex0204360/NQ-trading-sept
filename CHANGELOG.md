# Changelog

## Round 3 results and tournament stop - 2026-09-08

- Scored all 17,280 wick/swing-reversal hypotheses; all failed at least one screening gate, with no bootstrap finalists or survivors.
- Total: 45,360 hypotheses across 84 signatures and three completed rounds; no candidate execution failures.
- Quality remained 0. Rounds 2 and 3 each improved it by less than 0.01, triggering the original diminishing-returns stop. Round 4 is not run.
- Finalization will publish an empty library and keep heldout outcomes unopened. No statistical significance or profitability is claimed.

## Round 3 definitions - 2026-09-08

- Preregistered sixteen price/wick rejection and sixteen prior-swing reversal signatures: 17,280 hypotheses.
- Used fallback refinements already defined before market scoring; no winning approach exists to deepen. The choice is documented in docs/round3-decision.md.
- All statistical gates and stop conditions remain unchanged.

## Round 2 results - 2026-09-08

- Scored all 15,120 swing and volume/range hypotheses; all failed the preregistered screening gates.
- Zero bootstrap finalists and zero survivors; quality remains 0.
- First diminishing-return assessment failed to improve quality by 0.01. One further non-improving evaluable round will stop the tournament.
- Published baseline support and data interpretation limitations; heldout outcomes remain unopened.

## Round 2 definitions - 2026-09-08

- Preregistered sixteen causal swing/breakout signatures and twelve bar-volume/range signatures, for 15,120 hypotheses.
- Published a control-stratum support diagnostic after Round 1. Sparse strata remain unsupported under the original 30-control gate; no thresholds or strata were changed.

## Round 1 results - 2026-09-08

- Scored all 12,960 preregistered hypotheses across four training walk-forward folds.
- All failed at least one point-estimate/support gate; zero bootstrap finalists and zero survivors.
- Preserved full resolution distributions, per-fold statistics, rejection reasons, and resumable signature checkpoints.
- Quality score remains 0; no stop condition yet. Heldout outcomes remain unopened.

## Round 1 definitions - 2026-09-08

- Preregistered eight frozen candle-shape clusters and sixteen price-state signatures.
- Each signature tests 540 direction/magnitude/window combinations; 12,960 total hypotheses.
- Baseline cutpoints and code/data hashes frozen before scoring.

## 0.3.0 - 2026-09-08 - Audited pipeline and tested research engine

- Implemented chronological normalization, calendar reconciliation, first-passage outcomes, causal features, four finite candidate rounds, joint block inference, and OHLCV matcher.
- Audited 1,048,575 raw rows; retained 706,008 training and 330,516 heldout bars after 12,051 calendar exclusions. Every exclusion is recorded.
- Froze inferred end-stamp interpretation and recorded unresolved source/roll/calendar metadata, which prevents unsupported confirmation claims.
- Independent component verification: 244 tests pass; root validation and synthetic round integration: 28 tests pass.
- Added round checkpoints, fixed dependency lock, and command-line entry points. No market round results claimed yet.

## Integration review - 2026-09-08

- Added regression specifications for timestamp storage units, positive OHLC, exact model reproducibility, and partial session dates shared by adjacent folds.
- Added a synthetic end-to-end round contract that must log every parameter variant, reject weak support, and avoid holdout access.

## Orchestration and matcher RED checkpoint - 2026-09-07

- Specified protocol gates, exact tournament stops, immutable finalist manifests, calendar audit accounting, and stream/batch matcher equivalence before implementation.

## Statistical harness RED checkpoint - 2026-09-07

- Specified joint block resampling, matched baseline weighting, ambiguity handling, undefined inference, and Holm failure accounting before production.

## 0.2.0 - 2026-09-07 - Plan published; implementation checkpoint

- Published and remotely verified the original plan commit before implementation; preserved the user's initial README commit through a merge.
- Downloaded Kaggle dataset version 1 successfully. After the interrupted session, restored the Python environment and verified the pinned version and raw-file checksum.
- Added package/dependency configuration and explicit component contracts.
- Data, outcome-labeling, and feature/candidate agents supplied tests before production implementation; RED evidence is preserved in this checkpoint.
- No tournament results or statistically significant patterns are claimed at this point.

## 0.1.1 - 2026-09-07 - Repository supplied; publish attempt denied

- User created `alex0204360/NQ-trading-sept` and authorized publishing the existing work.
- Verified repository metadata and attempted to publish the plan as the first file.
- GitHub returned HTTP 403, `Resource not accessible by integration`. No remote commit or file was created by this attempt.
- Recorded the changed blocker: the repository exists, but the connected integration cannot write it. Preserve the plan-first boundary while repository access is corrected.
- Updated the plan's status notice without changing any statistical or implementation rule, and packaged the documentation plus genuine local Git history for review.
- Tried the subsequently user-authorized personal access token via HTTPS Git push. GitHub identified the account but returned HTTP 403 permission denied. No credentials were saved in project files, Git configuration, or logs; remote publication remains blocked.

## 0.1.0 - 2026-09-06 - Phase 0 documentation

- Verified GitHub account connectivity for `alex0204360`.
- Read ECC planning, research-first, test-first, review, and commit workflows.
- Prepared the research plan before any implementation or dataset download.
- Defined the January 1, 2025 ET holdout cutoff; all 1..45 bar horizons; six magnitude thresholds; conservative outcome and baseline semantics; exact support, precision, testing, and stopping gates.
- Delegated statistical planning and source/dependency research; recorded contributions and access findings.
- Recorded the repository-creation blocker. No GitHub project was created or published and no tournament round was attempted.
- Locally committed the plan alone as `152f83bad72e9af30260be467eb55aa210b7d1f0`, then recorded setup status, source research, and review evidence in a separate documentation checkpoint. Remote publication remains pending.

## Tournament rounds

Not started. Each attempted round will receive its own dated entry with family definitions, results, rejection counts, changes, and stop decision.
