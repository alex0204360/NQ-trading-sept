# MNQ/NQ Pattern-Mining & Strategy Engine

Phase 0 documentation only. This project has not yet downloaded market data, implemented code, run candidates, or established any statistically significant pattern.

Read `docs/research.plan.md` for the complete protocol, fixed statistical gates, outcome definitions, chronological split, bounded tournament, artifact layout, and tests. Read `docs/phase0-status.md` for the exact setup status and next dependency. The candidate and agent history starts in `history/events.jsonl`; no historical trading results are implied by setup events.

The user created [NQ-trading-sept](https://github.com/alex0204360/NQ-trading-sept) on September 7, 2026. The connected integration can read its metadata, but the first plan-file publication attempt returned HTTP 403, `Resource not accessible by integration`. No remote commit was created. Implementation remains blocked because the user requires the plan committed first; the next dependency is integration write access to this repository.

A subsequent user-authorized personal-access-token push also returned HTTP 403 permission denied. Its repository selection and Contents write permission require review. No token is present in this project and no remote push has succeeded.

Planned output is a Python pattern library and matching module accepting completed standard one-minute OHLCV bars from any caller. A future data feed can supply these bars without changing the matching module. No specific feed, execution connection, Pine Script, or forward test is part of this delivery.

This is research infrastructure, and an empty validated library is an acceptable result. NQ historical evidence will not establish MNQ execution performance or live profitability. Actual statistical results and the tournament stop reason will replace this setup-only status after the authorized research can run.
