"""Reproducible commands with an append-only stage and failure ledger."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .research import digest, prepare_experiment, run_experiment, write_trades


def main():
    parser = argparse.ArgumentParser(description="NQ causal scalping research")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "research"):
        command = sub.add_parser(name)
        command.add_argument("--experiment", default="r001")
        if name == "research":
            command.add_argument("--max-signatures", type=int)
    for name in ("match", "replay"):
        command = sub.add_parser(name)
        command.add_argument("--library", type=Path, required=True)
        command.add_argument("--bars", type=Path, required=True)
        command.add_argument("--expected-sha256")
        command.add_argument("--diagnostic", action="store_true")
        command.add_argument("--known-close")
        if name == "replay":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = args.root / "history/scalping.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)

    def log(status, **detail):
        with history.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        time=datetime.now(UTC).isoformat(),
                        actor="root",
                        stage=args.command,
                        status=status,
                        **detail,
                    )
                )
                + "\n"
            )

    experiment = getattr(args, "experiment", None)
    log("started", experiment=experiment)
    try:
        if args.command == "prepare":
            value = prepare_experiment(args.root, args.experiment)
            print(
                json.dumps(
                    {
                        "experiment": args.experiment,
                        "signatures": len(value["signatures"]),
                        "actions": len(value["actions"]),
                        "status": value["status"],
                    }
                )
            )
        elif args.command == "research":
            value = run_experiment(args.root, args.experiment, args.max_signatures)
            print(
                json.dumps({k: v for k, v in value.items() if k not in ("candidates", "finalists")})
            )
        else:
            from .matcher import Matcher, replay

            if args.expected_sha256 and digest(args.library) != args.expected_sha256:
                raise ValueError("library identity differs")
            library = json.loads(args.library.read_text())
            bars = (
                pd.read_parquet(args.bars)
                if args.bars.suffix == ".parquet"
                else pd.read_csv(args.bars)
            )
            bars["timestamp"] = pd.to_datetime(bars.timestamp)
            schedule_path = args.root / "data/schedule.parquet"
            schedule = pd.read_parquet(schedule_path) if schedule_path.exists() else None
            if args.command == "match":
                matcher = Matcher(
                    library,
                    diagnostic=args.diagnostic,
                    schedule=schedule if args.known_close is None else None,
                )
                print(json.dumps(matcher.match(bars, known_close=args.known_close)))
            else:
                if schedule is None:
                    raise ValueError("replay requires a known schedule")
                pattern = next(
                    p
                    for p in library["patterns"]
                    if p["pattern_id"] == library["active_pattern_id"]
                )
                trades = replay(bars, pattern, schedule, diagnostic=args.diagnostic)
                write_trades(args.output, trades)
                print(
                    json.dumps(
                        {
                            "trades": len(trades),
                            "output": str(args.output),
                            "measured_forward_performance": False,
                        }
                    )
                )
        log("completed", experiment=experiment, objective="unfinished")
    except Exception as exc:
        log("failed", experiment=experiment, error=repr(exc), objective="unfinished")
        raise
