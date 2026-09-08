# Independent component review

Reviewer: `review_components` sub-agent. Date: September 8, 2026.
Scope: data normalization, chronological splits, outcome labels, causal features,
candidate signatures, statistical estimates, and the bar-input matcher. No market
data was accessed for this review. Synthetic fixtures establish implementation
behavior only.

## Findings and fixes

1. Feature gap detection interpreted a timestamp array's integer storage as
   nanoseconds even when pandas stored microseconds. The existing regression
   failed. Normalize the validated timestamp index to nanoseconds before elapsed
   comparisons, making equivalent physical timestamps produce identical features.
2. Fixed random seeds alone did not make exact KMeans JSON artifact hashes stable
   under parallel floating-point reductions. The initial-fit future-perturbation
   test failed despite identical fitting rows. Limit BLAS/OpenMP fitting to one
   thread so fixed inputs produce exact, reproducible artifacts.
3. Outcome input validation accepted internally consistent nonpositive OHLC.
   A new negative-price fixture failed with `DID NOT RAISE`. Require positive
   prices consistently with the canonical bar contract.
4. The statistical compiler required disjoint calendar date labels across folds.
   A midnight ET fold boundary can split one CME session between two otherwise
   disjoint event intervals. A new fixture established that valid period frames
   must be allowed to share this session date. Keep unique period IDs and ordered,
   unique dates within each period, with pooled event-date support deduplicated.
   Resampling remains separate within each fold. Dependence across that partial
   boundary session is an approximation, not a claim of independent observations.

## Checks

The preexisting targeted component suite passed 242 tests after fixes 1 and 2.
The added negative-price and shared-session regressions were separately observed
failing before implementation and checkpointed by root at `edb14ec`. The final
targeted run passed all 244 tests in 53.02 seconds. Isolated coverage output is
`reports/component-coverage.json`; it is a component measurement, not full-package
coverage for the newly integrated orchestration.

| Component | Line coverage | Branch coverage |
| --- | ---: | ---: |
| Data | 100% | 100% |
| Splits | 100% | 100% |
| Features | 100% | 100% |
| Candidates | 100% | 100% |
| Outcomes | 94.74% | 88.89% |
| Statistics | 95.17% | 90.00% |
| Matcher | 91.45% | 86.90% |

The review verified the centered, one-sided plus-one bootstrap test; joint
candidate/control daily sufficient counts; shared random day draws; candidate
stratum reweighting in every resample; preservation of explicit zero-event dates;
failure on undefined resamples rather than their deletion; both block-length
interval envelopes; and Holm adjustment including failed hypotheses.

Existing causal tests cover prefix invariance, future perturbation, gap/contract
warmup resets, and frozen initial-period model fitting. Outcome tests cover H=1,
H=45, scope rejection at H=46, opening-price priority, ambiguous double touches,
incomplete future windows, and causal suppression before censorship. Matcher
tests cover batch/stream equivalence, bounded history, empty libraries, atomic
invalid-input rejection, and explicit diagnostic populations.

## Remaining integration responsibilities

Root owns calendar provenance, actual data audit, immutable manifests, heldout
access, statistical gates, tournament publication, and final coverage. Candidate
round 3 currently uses deterministic price-state and swing refinements; any
winner-dependent family selection must be resolved against the preregistered
plan before publishing that round's manifest. No component review establishes
historical significance, profitability, live fills, or MNQ transferability.
