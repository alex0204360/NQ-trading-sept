# Phase 0 source and dependency research

Researcher: `/root/data_research`. Integrated by `/root`. No data was downloaded or audited; no dependency was installed for this project.

The dates, ET timezone, row count and zero-null claim in the plan are publisher-stated, based on indexed snippets from the [dataset card](https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025). Direct card extraction returned no readable text. Partial [publisher notebook](https://www.kaggle.com/code/tgtanalytics/nq-week-seasonality/notebook) snippets expose OHLCV and VWAP-related columns, but the complete raw header, dtypes and semantics remain unverified. Extra indicators will not be trusted as causal inputs. Vendor, contract identifiers, rolls and adjustment rules remain unknown. Dataset licensing is unresolved; a notebook's license does not establish the dataset's license.

The reported row count matches the Excel worksheet limit minus one. This suggests an export-truncation question; it is not proof of missing data. The audit must reconcile actual endpoints, sessions and missing minutes. Likewise, no nulls is not proof of correct or complete price history.

## Reuse decisions

| Dependency | Reuse | Project-specific work | Primary reference |
| --- | --- | --- | --- |
| KaggleHub | Dataset retrieval, explicit versions and cache | Manifest, checksums, frozen input version; runtime-only authentication | [Official repository](https://github.com/Kaggle/kagglehub) |
| pandas and NumPy; PyArrow for Parquet | Parsing, arrays, storage and timezone primitives | Canonical OHLCV validation, raw-row audit trail, contract/gap policy | [pandas time-series documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html) |
| SciPy | Numerical/statistical primitives | Joint day-block resampling, first-passage outcomes and failure handling | [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |
| statsmodels | Holm multiple-testing adjustment | Freeze the complete hypothesis family and include every failed submission | [Multiple-testing documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) |
| scikit-learn | Clustering primitives and established preprocessing where needed | Fixed reference-artifact identity, explicit calendar folds and outcome purging | [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) |
| pytest and coverage tooling | Development verification | User-visible behavior, causal invariants, regression and end-to-end tests | Version/API verification deferred to implementation setup |

Ordinary row bootstrapping does not preserve one-minute dependence. Generic time-series split tools do not know exchange sessions, missing minutes, or a label's future interval. The project will own those rules explicitly. No existing library or documentation establishes that a tradable NQ pattern exists.

Package names above are planned dependencies, not verified installed versions. Exact compatible versions and licenses will be checked and locked when implementation can begin. No large trading framework is needed for a bar-input research library.

## Research limitations

The root agent read the pinned ECC workflow via the GitHub connector. Raw web fetching of several ECC files failed, and guessed canonical plan/adaptation file paths returned 404; the actual planner, command, and applicable skill files were subsequently read. GitHub CLI was unavailable, so its repository/code-search workflow was not executed. Public official repository and package documentation research was used instead; no exhaustive registry coverage is claimed.

The data researcher was asked to stop additional browsing and finalize the evidence already gathered, leaving unresolved metadata explicit. This was a bounded documentation handoff, not a failed market experiment. Historical data access remains unverified, and implementation remains blocked before code because the new repository is unavailable.
