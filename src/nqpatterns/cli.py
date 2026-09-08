"""Command-line entry points. No credentials, feed connections, or orders."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="NQ short-horizon pattern research")
    parser.add_argument("--root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--source", type=Path)
    prepare = commands.add_parser("prepare-round")
    prepare.add_argument("number", type=int, choices=range(1, 5))
    run = commands.add_parser("run-round")
    run.add_argument("number", type=int, choices=range(1, 5))
    commands.add_parser("prepare-final")
    commands.add_parser("validate-final")
    match_parser = commands.add_parser("match")
    match_parser.add_argument("--bars", type=Path, required=True)
    match_parser.add_argument("--library", type=Path, default=Path("patterns/library.json"))
    args = parser.parse_args(argv)
    if args.command == "audit":
        from nqpatterns.acquisition import acquire

        result = acquire(args.root, args.source)
    elif args.command in ("prepare-round", "run-round"):
        from nqpatterns.research import ResearchRunner

        runner = ResearchRunner(args.root)
        result = (
            runner.prepare_round(args.number)
            if args.command == "prepare-round"
            else runner.evaluate_round(args.number)
        )
    elif args.command == "prepare-final":
        from nqpatterns.finalize import prepare_finalization

        state = json.loads((args.root / "results/tournament_state.json").read_text())
        result = prepare_finalization(args.root, state)
    elif args.command == "validate-final":
        from nqpatterns.finalize import evaluate_finalization

        result = evaluate_finalization(args.root)
    else:
        from nqpatterns.matcher import LiveMatcher

        matcher = LiveMatcher(args.library)
        result = matcher.match_bars(
            [json.loads(line) for line in args.bars.read_text().splitlines() if line.strip()]
        )
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
