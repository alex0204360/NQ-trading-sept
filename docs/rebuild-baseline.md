# Rebuild baseline, 2026-09-11

The active request is the user's profitability rebuild. The earlier V1 discovery delivery and V2 two-round budget do not fulfill this request. Profitability remains unfinished.

Starting commit: `731607099729bdc1e3151b767a695e5c86aea503`.
Existing checkout: `mnq-nq-pattern-engine`. Dedicated branch: `rebuild/profitable-price-action`.
The only configured remote is `origin`, `https://github.com/alex0204360/NQ-trading-sept.git`. GitHub connector metadata confirms account `alex0204360`, repository identity, and push access. No Bitbucket remote or Bitbucket URL was supplied in the visible task. A Bitbucket relationship is therefore unverified; no repository was cloned, migrated, or redirected. Preexisting untracked work is preserved.

The separately existing `nq-profitability-research` checkout is a prior follow-up study, not this branch's destination. Its source, frozen models, and training reports are read as historical evidence; new active implementation belongs here.

## Environment and commands

The interrupted session left broken virtual-environment executable links. Retained packages initially allowed the following commands from this checkout:

```sh
PYTHONPATH=src:.venv/lib/python3.12/site-packages python -m pytest -q
PYTHONPATH=src:.venv/lib/python3.12/site-packages python -m nqpatterns.cli run-round 1
```

Tests: 275 passed in 31.09 seconds. The documented round command reached its existing-results guard and raised `FileExistsError: A completed/partial round cannot be overwritten`. This is intentional preservation of published evidence, not the cause of zero survivors. Command output is preserved in `reports/rebuild/baseline-tests.log` and `baseline-cli-round1.log`.

An isolated runtime was restored using the existing lock file:

```sh
python -m venv --copies --system-site-packages ../nq-rebuild-runtime
../nq-rebuild-runtime/bin/python -m pip install -r requirements.lock
```

Subsequent diagnostic commands use that Python with `PYTHONPATH=src`, or the prior study's source path for V2. Three frozen round manifests were checked against `ResearchRunner._provenance()`: all source, training, calendar, and pipeline hashes match exactly. A separate diagnostic recomputes frozen scores without overwriting original results or using their checkpoints. Its final command and comparison belong in the failure analysis.

## Available data

Kaggle `tgtanalytics/nq-futures-1min-bar-2022-2025`, version 1; raw SHA-256 `1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b`. Audited pre-2025 training contains 706,008 one-minute bars. Training SHA-256 `1722334651520716155d7d14aca334342319e4431c084b5d6be8d24bd31b6f08`. Calendar SHA-256 `fccb7ac0bc7b5c709f76ca5528b303b861b42ef8d78c6d36703252771c9c24a3`.

The 330,516 rows at or after `2025-01-01T05:00:00Z` were previously mechanically audited and split but never scored for pattern outcomes. Do not open these prices for development. Source timezone is Eastern; interval-end source timestamps were inferred, not publisher verified, and converted to UTC interval starts. Contract-roll construction and vendor remain unknown. Existing audit and provenance artifacts are retained unchanged.

## Workflow

The ECC workflow already read and pinned in `docs/research.plan.md` remains applicable: plan, test-first implementation, independent review, reproducible verification, incremental commits. The user's new request supersedes V1's prohibition on a simulated execution engine and its definition of an empty library as a completed objective. It does not authorize real broker orders, a feed connection, or claims from horizons over 45 minutes.

Before production changes: finish baseline recomputation, commit `docs/failure-analysis.md`, then commit `docs/rebuild-design.md`. Each subsequent experiment must be registered before evaluating its outcomes. Preserve unsuccessful experiments and resumable state.
