"""Independent private artifact accounting checks; no model fitting or tuning."""
import argparse
import json
from pathlib import Path
import pandas as pd
from tools.mnq_oracle_prepare import sha256


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha256(directory/name) != expected:
            raise ValueError("Checksum mismatch: "+name)
    report = json.loads((directory/"results.json").read_text())
    count = 0
    for name, scenarios in report["backtests"].items():
        for scenario, metrics in scenarios.items():
            file = directory/(name+"_"+scenario+"_trades_PRIVATE.csv")
            if metrics["trades"] == 0:
                assert file.read_text().strip() == ""
                assert metrics["net"] == 0
                continue
            trades = pd.read_csv(file)
            assert len(trades) == metrics["trades"]
            assert (trades.entry_index == trades.signal_index+1).all()
            assert (pd.to_datetime(trades.entry_time)-pd.to_datetime(trades.signal_time) == pd.Timedelta(minutes=1)).all()
            assert (trades.exit_index >= trades.entry_index).all()
            assert (trades.entry_index.iloc[1:].to_numpy() > trades.exit_index.iloc[:-1].to_numpy()).all()
            costs = report["plan"]["costs"][scenario]
            assert (trades.commissions == 2*costs["commission_per_side"]).all()
            assert (trades.slippage == costs["slippage_ticks_per_side"]).all()
            assert ((trades.exit-trades.entry)*trades.side*2-trades.commissions == trades.net).all()
            assert (trades.gross_price_move-trades.slippage-trades.commissions == trades.net).all()
            assert trades.net.sum() == metrics["net"]
            assert trades.gross_price_move.sum() == metrics["gross_price_move"]
            assert sum(value["net"] for key, value in metrics["day_session"].items() if key.startswith("day:")) == metrics["net"]
            assert pd.to_datetime(trades.entry_time, utc=True).min() >= pd.Timestamp(report["plan"]["final_start"])
            assert pd.to_datetime(trades.exit_time, utc=True).max() < pd.Timestamp(report["plan"]["coverage_end_exclusive"])
            count += len(trades)
    assert report["selected_policy"] == report["selection"]["selected"]["model"]
    assert report["audit"]["sealed_nq_2025_files_opened"] == 0
    return {"verified": True, "scenario_count": sum(len(value) for value in report["backtests"].values()), "trade_evaluations": count, "limitations": "Accounting, input hashes and single-position chronology checked; does not certify fills, source provenance or statistical significance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
