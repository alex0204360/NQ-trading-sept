# Phase 0 review and agent contributions

No agent wrote implementation or tests, downloaded the dataset, or made GitHub mutations.

| Agent | Work produced | Destination | Status |
| --- | --- | --- | --- |
| `/root` | GitHub access checks, ECC workflow reading, protocol integration, architecture, commit plan, setup log and recovery steps | Plan, README, changelog, status and history | Documentation prepared and locally committed; remote publication blocked |
| `/root/statistical_plan` | Exact statistical support and precision recommendations; conservative baseline, one-time holdout, temporal dependence controls; independent plan review | `research.plan.md` sections 4-8; these review notes | Completed |
| `/root/data_research` | Dataset metadata and reusable dependency/source investigation | `source-research.md` and plan data/source sections | Findings recorded in source research document |

The statistical reviewer identified five material issues in the earlier draft. The root agent integrated these corrections before the first local plan commit:

1. Freeze signature-defining scalers and cluster centers from the initial pre-April 2023 fit period, and preserve their identities across folds and holdout. Do not pool unrelated clusters or change a qualified pattern through final refitting.
2. Apply 45-minute detection suppression before retrospective outcome-completeness checks. A censored event consumes its causal suppression interval.
3. Jointly resample candidate and baseline data with common five-day and twenty-day blocks. Use the interval envelope and larger p-value; apply Holm only to the complete frozen holdout submission set.
4. Deduplicate by signature before allocating the 24-hypothesis exploratory resampling budget, preserving diversity across candidate patterns.
5. Use cumulative incumbent improvement, explicit earliest diminishing-return timing, and a common finalization path. Empty libraries leave holdout untouched; an overall quality-target miss does not erase individually confirmed patterns.

The reviewer then reported no additional material defects found, accounting for the final edits listed. This is a review of the research protocol, not validation of software, data, or a market effect.

The root agent chose a common 45-minute suppression and completeness population across all horizons for comparability, with its coverage limitation stated explicitly. The final protocol, not earlier sub-agent suggestions, is the authoritative preregistration after publication.
