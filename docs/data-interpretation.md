# Frozen data interpretation and limitations

The requested [Kaggle data card](https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025/data) describes Eastern-time one-minute OHLCV and publishes a CC0 license. It does not explain the bar timestamp convention, upstream vendor, contract identifiers, or continuous-contract adjustment process. The raw CSV hash and resolved version are in `reports/data_manifest.json`; source prices are not redistributed.

Before any candidate scoring, a mechanical timestamp audit found bars at 18:01 and 17:00 ET, but none at 18:00 or 17:01. We inferred end stamps and subtracted one minute to produce canonical interval-start timestamps. This is an explicit inference, not publisher verification. It is frozen in `reports/pipeline_freeze.json` and all round manifests.

The dataset contains 1,048,575 records, matching an Excel worksheet's maximum data rows when a header is present. Its final session is partial. These observations raise an export-truncation question but do not prove why the publisher's file ends there.

[CME's published equity-futures hours](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html) include the afternoon halt; the [2022 NQ options clearing notice](https://www.cmegroup.com/notices/clearing/2022/09/Chadv22-345.pdf) also records it. Calendar reconciliation uses the version-pinned `CME_Equity` schedule from [pandas-market-calendars](https://github.com/rsheftel/pandas_market_calendars/blob/master/pandas_market_calendars/calendars/cme.py), including holidays and breaks. The source contains rows during some calendar-closed minutes. Those rows are quarantined by timestamp, with source-row IDs retained in the audit. No prices are filled, smoothed or winsorized. Historical holiday completeness and every exceptional exchange closure have not been independently certified against exchange notices.

Missing expected minutes and entirely missing sessions are listed separately from bars that fall outside the frozen calendar. Boundary-session incompleteness is identified explicitly; a scheduled missing-minute count is not automatically a count of bad ticks. All features and outcome windows stop at resulting gaps or contract boundaries. There is no contract column in this file, so unknown rolls cannot be reconstructed from metadata alone.

The metadata gate remains false. Even a statistical holdout pass would be marked unconfirmed for data provenance until a separately documented study resolves those assumptions. Current market evidence, software correctness, and execution profitability are separate questions.

Baseline support is another limitation: some quarter/time-of-day/volatility strata contain fewer than the preregistered 30 controls. Their outcomes are not pooled into neighboring strata after observing results. A failed support gate means insufficient evidence under this study, not proof that price patterns cannot exist.
