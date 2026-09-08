# Changelog

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
