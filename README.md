# NQ-trading-sept

NQ trading repository for September: a Python research system for discovering repeatable NQ one-minute patterns with holding windows of 1 through 45 bars.

Implementation is in progress. The [preregistered plan](docs/research.plan.md) was published before code. Kaggle dataset version 1 was downloaded; data auditing, outcome labels, and candidate modules are the current work. No tournament has completed and no statistically significant pattern has been established yet.

Follow [commits](https://github.com/alex0204360/NQ-trading-sept/commits/main/), [CHANGELOG.md](CHANGELOG.md), and the [history](history/events.jsonl) for actual checkpoints and agent contributions. Earlier GitHub access failures were resolved; their original records remain in the history.

The matcher will accept completed standard one-minute OHLCV bars from any caller. A future feed can supply those bars without changing the module. Live/delayed connections, order execution, forward validation, and Pine Script are outside this delivery.

An empty validated library is a valid research result. Historical NQ evidence does not establish live profitability or MNQ execution performance. Actual measured results and the tournament stop reason will be published here when execution finishes.
