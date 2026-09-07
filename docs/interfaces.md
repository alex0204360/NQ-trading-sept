# Component contract

The first plan commit `152f83bad72e9af30260be467eb55aa210b7d1f0` was verified on GitHub before these implementation interfaces were created. Repository main was `3cb0f52` at verification.

All historical components accept a pandas DataFrame with a RangeIndex and columns `timestamp` (timezone-aware UTC interval start), `open`, `high`, `low`, `close`, `volume`. Sorting/auditing belongs to the data pipeline; downstream components reject unsupported ordering. An optional `contract` is retained. A gap is any timestamp difference other than exactly one minute or a changed contract.

Detection index t refers to a completed bar. Entry is open[t+1]. Horizons include bars t+1 through t+H, maximum45. Public modules: `data.py` normalize/audit source rows; `splits.py` fixed chronological partitions; `outcomes.py` outcome arrays and causal event thinning; `features.py` past-only features; `candidates.py` finite signature definitions; `statistics.py` audit-friendly estimates; `matcher.py` stream/batch detection.

JSON records must use Python-native finite scalars and null for unavailable values. No NaN/Infinity output, secrets, source market data, or pickled executable models. Agent changes are confined to assigned files. Root owns commits and pushes, history/changelog, dependency setup, actual download, scoring orchestration, and integration. Agents must not commit, publish, change thresholds, access heldout, or download data.

All implementation work follows tests-first RED then minimal GREEN and independently reviewed fixes. Root commits verified RED evidence before accepting production files, preserving the required separate reachable checkpoints. Tests of missing implementation should fail for that absence, not missing third-party dependencies. Communicate readiness after tests are written, then wait for root's RED checkpoint before implementing.
