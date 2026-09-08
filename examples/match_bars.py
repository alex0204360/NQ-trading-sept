"""Replay local standard OHLCV JSONL through the feed-independent matcher.

Usage: python examples/match_bars.py patterns/library.json completed-bars.jsonl

Each input line is one completed bar, for example:
{"timestamp":"2026-09-01T14:30:00Z","open":24000,"high":24002,
 "low":23999.75,"close":24001,"volume":250,"completed":true}

Prices in this example are invented format examples, not NQ observations. Bars
must be ordered and minute-aligned. Supply 60 prior contiguous bars for warmup.
No feed is connected, and the matcher does not create or execute trade orders.
"""

import argparse
import json

from nqpatterns.matcher import LiveMatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="Frozen JSON pattern library")
    parser.add_argument("bars", help="Local JSONL of completed standard OHLCV bars")
    parser.add_argument("--diagnostic", action="store_true", help="Include marked raw matches")
    args = parser.parse_args()
    matcher = LiveMatcher(args.library, diagnostic=args.diagnostic)
    with open(args.bars, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                matches = matcher.update(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid completed bar at input line {line_number}: {exc}") from exc
            print(json.dumps({"line": line_number, "matches": matches}, allow_nan=False))


if __name__ == "__main__":
    main()
