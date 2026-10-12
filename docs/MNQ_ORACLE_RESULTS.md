# MNQ oracle imitation: completed preliminary experiment

## Decision

**No verified causal edge; no deployment.** The frozen development selector chose **no trade**, not the forest. The user-provided future-informed labels were used to train three causal classifiers; future prices, oracle exits and excursions were not input features. Labels can teach statistical associations, but cannot supply future information at execution time.

| Final-period policy | Trades at base cost | Net, $4 round trip | Net, $6 | Net, $10 | Base bar-close max DD |
|---|---:|---:|---:|---:|---:|
| Selected no-trade | 0 | $0 | $0 | $0 | $0 |
| Prior-return momentum | 409 | -$1,612 | -$2,517.50 | -$3,890 | $1,773 |
| Causal ALMA surrogate | 132 | -$624.50 | -$885 | -$1,401 | $675 |
| Logistic student | 208 | -$1,002.50 | -$1,418.50 | -$2,250.50 | $1,076.50 |
| Forest student | 199 | +$101 | -$297 | -$1,093 | $551.50 |
| Histogram student | 99 | -$700 | -$943 | -$1,281.50 | $880.50 |

All 18 development threshold/exit configurations are retained in `reports/mnq_oracle_2026/development.json`, including their losses. The forest's small final gain is not permission to switch the selected policy after observing final outcomes. Its stressed losses and short sample do not establish robustness. Stress scenarios rerun execution and can change trade counts.

## Reproduction and privacy

Preparation code from PR #4 was imported to main via PR #6 **without merging its unrelated rebuild ancestry**. Original NQ experiments, PR #3 and the sealed 2025 holdout were not modified. Public result work is draft PR #7.

Pinned code-preparation source: `e813003dc50fc5e891447fff9fb7ea83a9e16031`. Pinned private input: `alex0204360/NQ-profitability-research` commit `2ebbb3129dee265a562515ed2cf7117f0965c73d`, folder `data/mnq_oracle_2026`. Git command-line access to the private repository failed authentication; the authenticated GitHub connector retrieved the exact seven required files at that commit. No daily 2025/2026 file was needed or opened. All six minute-part hashes and teacher hash are checked before either stage.

Use an isolated Python 3.12 environment, not the older repository-wide research dependencies:

```sh
python -m pip install -r tools/mnq_oracle_requirements.txt
python -m pytest -q tests/test_mnq_oracle_prepare.py tests/test_mnq_oracle_research.py tests/test_mnq_oracle_artifacts.py
python -m tools.mnq_oracle_research development --data /private/data/mnq_oracle_2026 --out /private/results/run01
python -m tools.mnq_oracle_research final --data /private/data/mnq_oracle_2026 --out /private/results/run01
python -m tools.verify_mnq_oracle /private/results/run01
python -m tools.mnq_oracle_report /private/results/run01 reports/mnq_oracle_2026
```

Set `MNQ_PRIVATE_RUN` to the completed private run directory to enable the artifact integration test. Running final evaluation twice in one directory is rejected. An unchanged independent reproducibility rerun in a separate directory produced the identical `results.json` SHA256: `67735ca10d581f1c7abb4e50d616bbf9761fcbfa771d16a9672cb2b5811b6deb`. This repeated execution was a reproducibility check, not another selection or tuning experiment.

The plan was committed before training (`ab92df3818c63a971adc4625ce0d2f8ab5e48351`). Source, development results, failed candidate configurations and selected thresholds were committed before opening final performance (`1b5d808f0a45172ef7ef7c409370ad2afe28151d`). Exact versions, seeds, source hashes, input hashes, model hashes and private artifact hashes are retained. Serialized models must only be loaded from this trusted, hash-verified private run; joblib is not a safe format for untrusted files.

Public artifacts contain aggregate statistics and hashes only. Row-level predictions, all 3,907 trade evaluations across 24 policy/cost scenarios, gap/cancellation events and serialized models stay in private artifacts. A local deterministic archive, `mnq_causal_run01.zip` (663,512 bytes; SHA256 `0abf9d87f1121f0b55d1dda5845ef6247e6fc3a5a6b8f60af04f4e59405cbac8`), contains the complete run. Upload of that sensitive archive was blocked by a safety review pending explicit user approval and confirmation of repository privacy; it is not claimed as committed. A private destination branch was created, but no artifact commit or PR was created there. No raw Yahoo CSV or proprietary Pine code is published publicly. Licensing and retention review for the private input PR remains unresolved; that draft is not merged merely because file access works.

## Audit and evaluation design

- Reproduced 28,312 source rows, 28,311 complete minute-aligned bars, one partial last candle excluded, 26 zero-volume bars, 75 interruptions, 76 segments.
- Reproduced 866 matched teacher entries, 813 feature-ready examples, 542 teacher entry prices equal to the Yahoo minute close, and missing IDs 256, 403, 507, 508, 593, 671, 672, 754, 821.
- Of the gaps, four exceed one day; 71 are shorter. Duration is not proof that a gap is scheduled. No forward fill or synthetic exit candles.
- Training: September 27 through October 5 session boundary, with 60-minute and event-end purging. Development: October 5 through October 8 boundary, similarly purged. Final: October 8 22:00 UTC through October 12 14:30 UTC, approximately two sessions including partial Monday.
- Fit 7,675 training rows; evaluated 3,782 development rows and 2,221 feature-ready final rows. Final class counts: 67 short, 2,088 wait, 66 long. Wait-only accuracy would be about 94%, demonstrating why accuracy is a poor target here.
- Features use completed minute OHLCV only. Signals execute at the next contiguous open, one MNQ contract ($2/point, $0.50/tick). An oracle timestamp is aligned to the feature minute start, so imitation is delayed by one minute relative to that assumed teacher timestamp.
- Training rows are limited to the exported teacher coverage; earlier candles are warm-up only. Completeness of the exported trade list and no-trade negatives is assumed, not independently verified.
- Stop-first ambiguous OHLC bars, adverse stop gaps, capped favorable target gaps, no overlapping positions, zero-volume entry cancellations, and explicit unresolved exposure across gaps. Drawdown is liquidatable equity at bar closes, not intrabar or tick drawdown.

## What the labels taught—and did not teach

Final directional event recall: logistic 63.2%, forest 48.1%, histogram 3.8%. This measures agreement on oracle events, not profitable-trade accuracy. Directional average precision ranges approximately 0.056–0.083 versus directional prevalence around 0.030; the observed association does not establish executable profitability. Class-balanced training probabilities are not assumed calibrated; confusion matrices, Brier scores, calibration bins and sampled precision–recall curves are recorded.

The matched lookahead-on ALMA surrogate made +$618 base; the teacher-timing/fixed-exit noncausal benchmark made +$700.50 base. Both are deliberately ineligible oracle diagnostics, not trading strategies. Their $10-cost results were negative. The teacher's reported entry/exit prices and a matched Yahoo-close hindsight reference are separately labeled; neither is a certified executable or mathematical upper-bound result.

Exact private Pine offset, sigma, HTF anchoring and full semantics were not provided in the linked repositories. The ALMA benchmark therefore uses explicitly frozen surrogate assumptions (length 2, offset 0.85, sigma 6, complete 8-minute UTC buckets). Its lookahead-on/off variants otherwise share settings. It must not be described as an exact reproduction of ChartChico 6.9.

## Failed checks and remaining limitations

An initial new ALMA test incorrectly checked minute-of-hour modulo eight rather than epoch-aligned UTC buckets; that test assertion was corrected before final evaluation. The implementation did not change. The original six preparation tests passed. Logistic fitting emitted a SciPy `Unknown solver options: iprint` compatibility warning; fitting completed, no convergence failure was reported, and identical results were reproduced with pinned versions. No model fit exceptions were hidden.

Independent timestamp semantics, continuous-contract mapping, exchange-calendar reconciliation and actual brokerage fees remain unverified. No edge claim, real-money execution or agent marketplace is justified. Dashboard validation is structural; no screenshot-based visual inspection is claimed.

**Next smallest experiment:** obtain exact non-lookahead indicator settings and independently verified contract-specific timestamps, then collect a new untouched period for a preregistered forward-paper test. Keep these final outcomes frozen; do not tune until the small forest profit grows.
