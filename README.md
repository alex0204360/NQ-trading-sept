# NQ-trading-sept

MNQ/NQ Pattern-Mining & Strategy Engine. This research delivery is complete: three tournament rounds tested 84 signatures and 45,360 direction/magnitude/window hypotheses. No hypothesis passed every preregistered screening gate. The [confirmed pattern library](patterns/library.json) is therefore empty.

The Python infrastructure is ready to test. This result does not prove that short-horizon NQ patterns cannot exist: sparse matched-control groups, strict support requirements, and unresolved source metadata limit what this study can establish.

## Measured results

| Round | Approaches | Signatures | Hypotheses | Survivors |
| --- | --- | ---: | ---: | ---: |
| [1](results/round_01/interpretation.json) | Frozen candle clusters; price states | 24 | 12,960 | 0 |
| [2](results/round_02/interpretation.json) | Prior swing breakouts; bar volume/range | 28 | 15,120 | 0 |
| [3](results/round_03/interpretation.json) | Price/wick rejection; swing reversal | 32 | 17,280 | 0 |

Every signature tested both directions, thresholds of 1, 2, 4, 8, 16 and 32 points, and every integer holding window from 1 through 45 bars. Each round used four chronological training walk-forward folds. The required conservative lift was at least 5 percentage points, with at least 1,000 eligible events and the additional support, stability and precision rules in the [original plan](docs/research.plan.md).

No candidate reached the bootstrap shortlist. Confidence intervals are consequently null for screened rejects; none is presented as a statistically significant winner. Full scores include outcome counts, matched baselines, rejection reasons and time-to-resolution distributions. Hypothesis counts are not independent trade counts.

The tournament stopped for `diminishing_returns`: library quality stayed at zero, and Rounds 2 and 3 each improved it by less than 0.01. Round 4 was not run. The [frozen finalist manifest](results/frozen_manifest.json) contains no entries. [Stage 2](reports/stage2.json) is `not_run_no_survivors`; heldout outcomes remain unopened. [Stage 3](reports/stage3.json) is explicitly out of scope.

## Data and remaining assumptions

The audited Kaggle version contains 1,048,575 raw bars. Calendar reconciliation excluded 12,051 rows and retained 706,008 training bars and 330,516 heldout bars. The fixed cutoff is January 1, 2025 at 00:00 Eastern. Raw files, exclusions, gaps, flags and checksums are documented in the [data audit](reports/data_quality.json).

Timestamp end stamps were inferred from session boundaries before scoring; the publisher did not verify that convention. Contract-roll construction and complete historical calendar accuracy remain unresolved. Some baseline strata have fewer than the required 30 controls. These limitations are detailed in [data interpretation](docs/data-interpretation.md). No live profitability, transaction-cost performance, or NQ-to-MNQ transfer has been demonstrated.

## Test it

Start with the [installation and testing guide](docs/reproduce.md), using Python 3.12. After installation:

```sh
python -m pytest -q
python examples/match_bars.py examples/demo_library.json examples/demo_bars.jsonl --diagnostic
```

The invented 160-bar fixture produces three accepted diagnostic detections and 97 explicitly suppressed matches. It is a mechanics demo, not a discovered market pattern. The delivered confirmed library returns no matches, as expected for an empty library.

The [live-matching module](src/nqpatterns/matcher.py) accepts completed standard OHLCV bars through `LiveMatcher.update(bar)` or `match_bars(iterable)`. Timestamps are aware interval starts; 60 prior contiguous bars provide warmup. It returns matched IDs and historical statistics, applies the same 45-minute suppression as research, and handles gaps and contract changes. A future feed can supply these bars without modifying the matcher. No feed, broker connection, order execution, forward test, or Pine Script is included.

Verification: 275 tests passed; 85.7% line coverage and 84.1% branch coverage; source lint and package builds passed. See [verification](reports/verification.json) and [artifact integrity](reports/artifact_integrity.json).

## Project history

The original plan was published before code. Tests, implementation, candidate definitions, each round's results, and finalization were committed in separate checkpoints. [CHANGELOG.md](CHANGELOG.md), [agent/stage history](history/events.jsonl), and the [candidate ledger](history/candidates.jsonl) preserve failures and decisions. Large per-hypothesis JSONL files are intended for local inspection; the linked round summaries are compact.
