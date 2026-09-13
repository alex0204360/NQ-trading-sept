"""End-to-end chronological fixtures for experiment and checkpoint guards."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nqscalp import cli, research
from nqscalp.features import compute_features


def test_registered_workflow_retains_gaps_and_never_qualifies_losing_fixture(tmp_path, monkeypatch):
    bars, sessions = [], []
    for year in (2023, 2024):
        for date in pd.date_range(f"{year}-02-01", periods=25, freq="B", tz="UTC"):
            opening = date + pd.Timedelta(hours=14)
            close = 100 + np.arange(80) * 0.25
            bars.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(opening, periods=80, freq="min"),
                        "open": close - 0.25,
                        "high": close + 0.5,
                        "low": close - 0.5,
                        "close": close,
                        "volume": 100,
                    }
                )
            )
            sessions.append(
                {"market_open": opening, "market_close": opening + pd.Timedelta(hours=2)}
            )
    bars = pd.concat(bars, ignore_index=True)
    schedule = pd.DataFrame(sessions)
    features = compute_features(bars)
    signature = {
        "id": "fixture",
        "family": "synthetic",
        "direction": 1,
        "session_bucket": "all",
        "conditions": [{"feature": "r5", "op": ">=", "value": 1}],
    }
    manifest = {
        "source_hashes": {"code": "fixture"},
        "input_hashes": research.INPUT_HASHES,
        "signatures": [signature],
        "actions": [{"direction": 1, "target_points": 4, "stop_points": 4, "horizon": 1}],
    }
    path = tmp_path / "research/r001.json"
    research.write_json(path, manifest)
    monkeypatch.setattr(research, "source_hashes", lambda root: {"code": "fixture"})
    monkeypatch.setattr(research, "training_context", lambda root: (bars, schedule, features))
    monkeypatch.setattr(
        research.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=path.read_bytes()),
    )
    result = research.run_experiment(tmp_path)
    assert result["status"] == "completed" and result["fit_actions"] == 1
    assert not result["holdout_accessed"] and result["finalists"] == []
    candidate = result["candidates"][0]
    assert candidate["summary"]["net_points"] < 0
    assert candidate["summary"]["n_unresolved"] == 25
    assert candidate["summary"]["accounting_violations"] == 0
    assert "n_unresolved" in candidate["reasons"]
    ledger = next((tmp_path / "reports/scalping/r001/trades").glob("*.csv"))
    trades = pd.read_csv(ledger)
    assert trades.status.eq("unresolved").sum() == 25
    assert trades.balance_after_dollars.isna().any()
    monkeypatch.setattr(
        research, "simulate_batch", lambda *a, **k: pytest.fail("recomputed completed checkpoint")
    )
    assert research.run_experiment(tmp_path) == result
    checkpoint = tmp_path / "reports/scalping/r001/checkpoints/fixture.json"
    damaged = json.loads(checkpoint.read_text())
    damaged["manifest_sha256"] = "wrong"
    research.write_json(checkpoint, damaged)
    with pytest.raises(ValueError, match="checkpoint identity"):
        research.run_experiment(tmp_path)


def test_uncommitted_manifest_prevents_data_access(tmp_path, monkeypatch):
    path = tmp_path / "research/r001.json"
    research.write_json(path, {"source_hashes": {}})
    monkeypatch.setattr(
        research.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"changed")
    )
    monkeypatch.setattr(
        research, "training_context", lambda *a: pytest.fail("read data before manifest guard")
    )
    with pytest.raises(ValueError, match="commit registered"):
        research.run_experiment(tmp_path)


def test_match_cli_and_failure_are_auditable(tmp_path, monkeypatch, capsys):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({"schema_version": 1, "patterns": [], "active_pattern_id": None}))
    bars = tmp_path / "bars.csv"
    bars.write_text("timestamp,open,high,low,close,volume\n2024-01-03T15:00Z,100,101,99,100,10\n")
    argv = [
        "nqscalp",
        "--root",
        str(tmp_path),
        "match",
        "--library",
        str(library),
        "--bars",
        str(bars),
    ]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert json.loads(capsys.readouterr().out)["decision"] == "no_trade"
    monkeypatch.setattr("sys.argv", argv + ["--expected-sha256", "incorrect"])
    with pytest.raises(ValueError, match="identity"):
        cli.main()
    events = [
        json.loads(line) for line in (tmp_path / "history/scalping.jsonl").read_text().splitlines()
    ]
    assert events[-1]["status"] == "failed"
