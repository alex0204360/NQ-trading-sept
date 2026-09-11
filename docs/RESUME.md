# Resumable profitability rebuild

Status: unfinished; no profitable strategy is demonstrated.

Branch: `rebuild/profitable-price-action` in the existing reference checkout. Start: `731607099729bdc1e3151b767a695e5c86aea503`. Do not clone, migrate, delete historical results, or touch held-out prices for development.

Current step: finish actual V1/V2 baseline recomputation. V1's 275 tests pass and all frozen provenance hashes match. Original round overwrite protection works. Diagnostic agents own `tools/diagnostics/reproduce_v1.py`, `reproduce_v2.py`, and corresponding `reports/rebuild` results. Root owns commits.

Next concrete work: commit evidence and failure analysis, then the prospective rebuild design before any production edit. Build a fresh price-action and net-payoff engine with explicit executable trades, conservative ambiguous fills, a shared matcher/replay decision path, and causal chronological research. Run real-data experiments; an empty library is not completion of the revised goal.

Runtime: `../nq-rebuild-runtime/bin/python` with `PYTHONPATH=src`. Restore from `requirements.lock` if necessary. Credentials stay in transient process environments only, never source or documentation.

Held-out outcomes: unopened. Pre-2025 data is repeatedly examined development data; it can never be relabeled unseen. Later confirmation requires a frozen experiment, source/model/config hashes, and an access ledger written before reading outcomes.
