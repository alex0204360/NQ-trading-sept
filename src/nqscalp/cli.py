"""Reproducible commands with an append-only stage and failure ledger."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .research import prepare_experiment, run_experiment


def main():
    parser = argparse.ArgumentParser(description="NQ causal scalping research")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "research"):
        command = sub.add_parser(name)
        command.add_argument("--experiment", default="r001")
        if name == "research":
            command.add_argument("--max-signatures", type=int)
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

    log("started", experiment=args.experiment)
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
        else:
            value = run_experiment(args.root, args.experiment, args.max_signatures)
            print(
                json.dumps({k: v for k, v in value.items() if k not in ("candidates", "finalists")})
            )
        log("completed", experiment=args.experiment, objective="unfinished")
    except Exception as exc:
        log("failed", experiment=args.experiment, error=repr(exc), objective="unfinished")
        raise
