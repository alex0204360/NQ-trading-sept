# MNQ oracle imitation: Codex handoff (2026-10-12)

Status: DATA INGESTION PREPARED; no causal profitability has been established.

## Goal and boundaries
Test whether future-informed ChartChico x Scalper 6.9 entry direction/timing has detectable predecessors in actual **MNQ** minute candles. Treat the 89.6% / 875-trade historical teacher report as an **oracle**, never an executable strategy result. MNQ: 0.25 points/tick, $0.50/tick, $2/point. Do not convert the teacher P&L to NQ by multiplying results without a separately modeled NQ execution study.

Preserve all existing NQ Kaggle experiments and the sealed 2025 holdout. This is a distinct September–October **2026 MNQ** project. Do not alter or claim results for Campaign 2, the existing research hashes, PR #3, or the 2025 sealed holdout.

## Data location

Public CODE: alex0204360/NQ-trading-sept, branch research/mnq-oracle-prep-20261012
Private DATA: alex0204360/NQ-profitability-research, branch research/mnq-oracle-dataset-20261012
Private folder: data/mnq_oracle_2026/

The private branch holds:
- MNQ_Yahoo_1m_20260913_20261012_part01.csv through part06.csv, each having its own CSV header, 28,312 combined 1-minute-source records.
- mnq_oracle_teacher_labels.csv, 875 completed hindsight-oracle trades.
- MNQ_Yahoo_daily_20251013_20261012.csv, 251 daily bars. Daily bars are **not** joined in the initial pipeline. If used, only prior completed sessions are allowed.

The public repository intentionally does NOT contain raw Yahoo market data or the proprietary Pine script. Do not publish private files into the public PR.

Original user-upload Yahoo 1-minute CSV: SHA256 32cb3e076a236948810559734b6d63d6f3e79489ee020ee0c260feac78b4ae63, size 2,587,905 bytes. The CSV parts use normalized LF endings and repeated headers. Their SHA256 hashes, in order:
1. 3430116d80a7e4e2b9e2b2ed34315dc5255d2433620b737c96055556312176f2
2. a7a64421e57fa8142a7c07f4415c719981c7d6020d50c068749806a0aff3eb34
3. 514249caa15cc110b289c5c48b2c8afa44c1d602d2174e5b7c393bf67a4c4a4b
4. a218f1228416cd191d38d1841e66a75b2eb677f87448f42b8f9a120388816933
5. b2777b42a4ef96037c868e4664e9142aac8ecb010abf4e4e65d2b59b6a28bff6
6. bcf54e4b620996c1f4c28cda92f442f473573e042319c9eb9e1e810dc15a1114

## Verified preliminary audit

The full uploaded CSV was locally inspected on 2026-10-12, and the preparation script was run successfully against these exact parts and the original 875-label CSV.

- 28,312 input rows; 28,311 minute-aligned retained; 1 non-minute ending at 2026-10-12T15:40:43Z removed as partial/inconsistent.
- Coverage: 2026-09-13T22:10:00Z to 2026-10-12T15:40:00Z (not a complete 30 calendar days).
- 26 zero-volume retained bars; 75 >1-minute interruptions, including expected weekends/maintenance and possible missing exchange candles. 76 uninterrupted segments.
- All prices follow quarter-point ticks and OHLC inequalities.
- 866 / 875 oracle trade entries match available minute timestamps; 9 entries missing bars (IDs 256, 403, 507, 508, 593, 671, 672, 754, 821).
- 813 / 875 teacher labels correspond to fully warmed-up causal feature rows (some gaps reset lookbacks).
- At matched entries, 542 trade prices exactly equal Yahoo minute close. Timestamp convention and 2026 continuous-contract roll mapping still need independent verification.
- No baseline AI model has yet been trained and no simulated student net profitability has been demonstrated.

## Reproduce

Check out BOTH repositories. From the public repo root using Python with pandas/numpy:

~~~bash
PYTHONPATH=. python -m pytest -q tests/test_mnq_oracle_prepare.py
python tools/mnq_oracle_prepare.py \
 --candles /path/to/private-repo/data/mnq_oracle_2026/MNQ_Yahoo_1m_20260913_20261012_part*.csv \
 --teacher /path/to/private-repo/data/mnq_oracle_2026/mnq_oracle_teacher_labels.csv \
 --out results/mnq_oracle_prep
~~~

Outputs: mnq_clean_bars.csv; mnq_causal_features_teacher_labels.csv; mnq_audit.json. Results are local and should be checkpointed separately, not committed wholesale to existing frozen NQ research.

## Non-negotiable information timing

Teacher Pine has request.security(..., lookahead=barmerge.lookahead_on), with an HTF multiplier of 8 and a 2-period ALMA. The teacher is NOT an oracle that was available at entry time. Never feed any of the following to the student: oracle-aligned exit prices, the trade's favorable/adverse excursion, future highs/lows, future bar/HTF closes, next-bar return, or hindsight strategy equity.

Yahoo timestamps denote the start of each interval, so a feature based on that minute's OHLCV is known only at its *end*. Model decisions can use it no earlier than the close; simulate execution at the NEXT bar open, with a price gap if appropriate. The 2026 oracle entry timestamps are currently interpreted as Eastern interval STARTS, a hypothesis that needs a timestamp/provenance test before modeling fills.

Split BEFORE fitting/threshold tuning; group by contiguous session/day; purge any training labels whose event/horizon touches the evaluation window. Never random-split overlapping candles.

## Next Codex execution, in order

1. Pull both branch heads and run the six tests; reproduce the audit counts above. Verify each part's hash; fail on discrepant data or overlaps. Review 9 missing teacher timestamps, quarterly-roll mapping, zero-volume bars, timestamp conventions, and unexplained intraday gaps. Never forward-fill OHLCV or manufacture an exit candle.
2. Build a separate, pre-registered causal signal benchmark for this 8x HTF/ALMA crossover. Ensure Pine lookahead_on vs lookahead_off uses otherwise identical settings. Compare to a trivial no-trade baseline and simple prior-return/momentum classifiers.
3. Train at most three lightweight, reproducible teacher-imitation baselines (e.g. regularized logistic regression, balanced random forest, histogram boosting) on normalizations/past-only features. Count no-trade negatives, and report class prevalence, long/short confusion matrix, average precision/precision-recall, directional event recall and calibration. Do NOT optimize solely for accuracy, since 'wait' dominates.
4. Select threshold and candidate exit rules on development data ONLY; keep a chronological final evaluation segment untouched. Since the sample has about 2 weeks of oracle trades, label any model conclusions *preliminary*. Use a 60-minute minimum purge, increased when exit/horizon demands. In particular, do not mistake longer oracle durations for permission to train on future.
5. Implement a deterministic, single-position MNQ replay, next-open fills, fees and slippage, realistic session pauses, stops/targets/time-exits, conservative stop-before-target on ambiguous bars. Report every trade including unresolved/market-closure events, gap exposure, maximum drawdown, profit factor, net expectancy, turnover and trade count. Distinguish gross, friction, and net. Example sensitivity: $1.50 commission per contract per side + one 0.25-point tick slippage per side ($4.00 round trip), then stress $6/$10 round trip; configurable and independently confirm actual brokerage fees. No claim of profitability on teacher-trade replay.
6. Compare teacher/oracle upper bound, causal ALMA, no-trade, and student on the SAME dates and price feed. Distinguish exact fills from lower-resolution ambiguous fills. Include day/session results. No cherry-picked cohorts or rewrites after opening final holdout.
7. Commit code/tests/manifests with seeded reproducibility, explain exactly which steps ran and which remain open, and open a DRAFT PR. Propose the smallest next actionable experiment only after initial costed results exist.

STOP conditions: if the private dataset cannot be accessed in Codex, report it explicitly, do not fabricate candle rows. If the timestamp conventions or roll mapping invalidate the comparison, do not report an 'edge'. No automated real-money execution and no claim the teacher's hindsight win rate is live-tradable.
