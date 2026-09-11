# Failure analysis recovered after interruption

2026-09-11. Starting commit `731607099729bdc1e3151b767a695e5c86aea503`; baseline checkpoint `9324f4e`. Existing remote is the supplied GitHub repository. No Bitbucket remote or URL is available, so that relationship is unverified. No clone or migration occurred. Unrelated untracked work is preserved.

The previous turn completed the following real-data checks before interruption. Only the initial baseline push survived; detailed diagnostic artifacts are being regenerated. This recovery distinction does not change the findings or invent a new experiment. Original source/results remain available as evidence.

## Reproduced observations

V1's 275 tests pass. Its documented round command refuses to overwrite completed evidence, intentionally. All three manifests match source/training/calendar hashes. A fresh recomputation, without checkpoint reuse, matched all 45,360 complete archived score rows in 222.968 seconds.

| Round | Signatures | Scores | Raw detections | Retained | WF eligible | Survivors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 12,960 | 1,788,923 | 214,130 | 178,779 | 0 |
| 2 | 28 | 15,120 | 758,733 | 203,675 | 169,929 | 0 |
| 3 | 32 | 17,280 | 566,166 | 126,499 | 105,660 | 0 |

Detection counts overlap across signatures; raw/retained span training while eligible counts cover validation folds. Minimum-edge failures are 12,960/15,120/16,950. Baseline-support failures are 12,960/15,120/15,120; reasons overlap. No hypothesis reaches bootstrap inference. The confirmed library is empty. V1 orders, fills, closed trades and net P&L are unimplemented, not a measured zero-profit trading strategy.

V2 (`bef84d0be184d558f52435aacc9f59b5eed04527` in the separately existing follow-up checkout) has 706,008 training bars, 654,103 valid feature rows, 130,817 five-minute opportunities, 130,297 with positive calendar horizon, and 125,518 common45 fitting rows. Each of four families replays 113,688 validation decisions. All frozen family/fold and baseline ledgers reproduce zero accepted signals/orders/closed trades. Activity ends in `policies.select_action`, before execution.

Every supported V2 action rank is negative. Best ranks are -0.068753245 (regimes), -0.646675334 (prototype8), -0.469385404 (prototype16). The only positive ranks belong to unsupported regime16 state13: n173/25dates and +0.104405818 in fold1; n183/29dates and +0.002455856 in fold2. Support requires n200. Saying every score is negative would be inaccurate.

Fresh V2 vector calculations covered 108,000 outcomes on 100 seeded real training rows. All 1,200 scalar comparisons agreed: 635 stops, 461 targets, 104 timeouts. All 78 execution/replay fixtures passed. Probe total -1,325.25 points is a mechanical diagnostic collection, not a strategy backtest.

## Inventory, consequences, and uncertainty

| Component | Observation | Consequence/correction |
| --- | --- | --- |
| Data | Audited Kaggle v1, calendar, chronological split and hashes exist | Data is available; zero input is not the failure |
| V1 discovery | Causal signatures, directional first-passage labels, fixed45 suppression | Hit-rate lift does not measure executable net profit |
| V1 screening | n1000, lift0.05, occupied-stratum controls30, fold gates | Sparse controls and inadequate edge reject every hypothesis |
| V1 execution | No trade/position/account simulator | Cannot establish profitability, even if a pattern passed |
| V2 discovery | 8/16 broad states with global shrinkage and mean-minus-SE scores | No supported action clears selection |
| V2 execution | Next-open, TP/SL/timeout, fees, conservative ambiguity, replay exist | Mechanics operate on real bars; not the historical zero-action cause |
| V2 handoff | Frozen empty survivors; terminal library/Stage2 absent | Prior workflow was interrupted and is not a completed new-goal delivery |
| Cache recovery | V2 payoff file has102,866,944bytes; header requires542,237,888; digest mismatch | Current refitting fails; atomic/hash-checked caches needed |
| Environment | Interrupted venv executables broken, packages retained | Recover runtime without altering research conclusions |

No software defect has been demonstrated to cause historical zero survivors. Tests/reproduction establish internal consistency, not advantage. Broad states may conceal conditional price-action effects; this is a hypothesis requiring new research. Fixed45 suppression and five-minute decision clocks reduce opportunity independently of actual exit time; changing them is a prospective design change. The old budgets explain why searches stopped, but do not satisfy the expanded profitability objective.

The cache truncation is a current recovery defect. It does not prove that the cache was corrupt when old models were fit. Missing data must never be converted into a favorable exit at the last observed close.

## Data and reproducibility

Kaggle `tgtanalytics/nq-futures-1min-bar-2022-2025` v1, raw SHA256 `1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b`. Training SHA256 `1722334651520716155d7d14aca334342319e4431c084b5d6be8d24bd31b6f08`. Calendar SHA256 `fccb7ac0bc7b5c709f76ca5528b303b861b42ef8d78c6d36703252771c9c24a3`. Training has1,137segments and122 inherited missing intervals/1,347minutes. Whole-source audit has12,051calendar-excluded rows and4,146missing expected minutes; scopes differ. No imputation or outcome-based exclusions are authorized.

Eastern source timestamps were inferred to denote interval ends and converted to UTC starts. Vendor/contract/roll construction and publisher-confirmed timestamp convention remain unavailable. Historical inference is conditional on these assumptions, not verification of live individual-contract fills.

2025 outcomes remain unopened; prior access was mechanical audit/split only. All pre2025 history is examined development data and cannot be relabeled unseen.

From the existing checkout:

```sh
PYTHONPATH=src:.venv/lib/python3.12/site-packages python -m pytest -q
PYTHONPATH=src:.venv/lib/python3.12/site-packages python -m nqpatterns.cli run-round 1
```

The second command intentionally produces the archived overwrite guard. Recomputed diagnostics under `tools/diagnostics/` invoke the same frozen engine without altering original artifacts. The recovery agent regenerates missing logs; any differences must be investigated before production evaluation.

Agents `/root/reproduce_v1` and `/root/reproduce_v2` produced the original fresh checks; `/root/design_review` reviewed the replacement protocol. Usage interruption lost their unpushed files. `/root/recover_diagnostics` regenerates evidence. Root owns integration/commits. Profitability remains unfinished.
