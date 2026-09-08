# ruff: noqa: TRY004 -- schema errors consistently use ValueError
"""Match frozen research patterns against caller-supplied completed OHLCV bars.

There is no data connection or order execution here. ``timestamp`` is the UTC-aware
minute interval start; passing a bar asserts it is completed. Callers may include
``completed=True`` explicitly. The reported ``available_at`` is its interval end.
"""

import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import match_candidate, validate_signature
from .features import compute_features

_METADATA = {
    "schema_version": 1,
    "instrument": "NQ",
    "bar_interval_minutes": 1,
    "timestamp_convention": "interval_start",
    "tick_size": 0.25,
}
_PRICE_FIELDS = ("open", "high", "low", "close")
_COOLDOWN = pd.Timedelta(minutes=45)
_ONE_MINUTE = pd.Timedelta(minutes=1)


def _load_library(source: dict[str, Any] | str | PathLike) -> dict[str, Any]:
    if isinstance(source, (str, PathLike)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Pattern library must be a JSON object or a local JSON path")
    # Round-trip makes the snapshot independent of caller mutation and rejects NaN,
    # Infinity, executable objects, and unsupported non-JSON model artifacts.
    library = json.loads(json.dumps(source, allow_nan=False))
    for key, expected in _METADATA.items():
        value = library.get(key)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"Unsupported library {key}; expected {expected!r}")
    patterns = library.get("patterns")
    if not isinstance(patterns, list):
        raise ValueError("Library patterns must be a list")
    seen = set()
    for pattern in patterns:
        if not isinstance(pattern, dict):
            raise ValueError("Every pattern must be an object")
        identifier = pattern.get("pattern_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every pattern requires a nonempty pattern_id")
        if identifier in seen:
            raise ValueError(f"Duplicate pattern_id: {identifier}")
        seen.add(identifier)
        if not isinstance(pattern.get("family"), str) or not pattern["family"].strip():
            raise ValueError("Every pattern requires a family")
        if pattern.get("direction") not in ("up", "down"):
            raise ValueError("Pattern direction must be up or down")
        if pattern.get("status") not in ("confirmed", "provisional"):
            raise ValueError("Pattern status must be confirmed or provisional")
        horizon = pattern.get("horizon_bars")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 45:
            raise ValueError("Pattern holding window must be an integer from 1 through 45 bars")
        magnitude = pattern.get("magnitude_points")
        if isinstance(magnitude, bool) or not isinstance(magnitude, Real) or magnitude <= 0:
            raise ValueError("Pattern magnitude_points must be positive")
        if not _on_tick_grid(magnitude):
            raise ValueError("Pattern magnitude_points must respect the 0.25-point NQ tick")
        if not isinstance(pattern.get("historical_stats"), dict):
            raise ValueError("Every pattern requires a historical_stats object")
        validate_signature(pattern.get("signature"))
    return library


def _on_tick_grid(value: float) -> bool:
    units = value / 0.25
    return math.isfinite(units) and math.isclose(units, round(units), rel_tol=0, abs_tol=1e-7)


def _canonical_bar(bar: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(bar, dict):
        raise TypeError("A completed OHLCV bar must be a dictionary")
    if "completed" in bar and bar["completed"] is not True:
        raise ValueError("Only completed one-minute bars are accepted")
    for key in ("instrument", "bar_interval_minutes", "timestamp_convention", "tick_size"):
        if key in bar and (isinstance(bar[key], bool) or bar[key] != _METADATA[key]):
            raise ValueError(f"Bar {key} does not match the frozen library metadata")
    if "timestamp" not in bar:
        raise ValueError("Bar requires timestamp")
    try:
        stamp = pd.Timestamp(bar["timestamp"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Bar timestamp cannot be parsed") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Bar timestamp must be timezone-aware")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor("min"):
        raise ValueError("Bar timestamp must identify a one-minute interval start")
    result = {"timestamp": stamp}
    for key in (*_PRICE_FIELDS, "volume"):
        value = bar.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Bar {key} must be a finite number")
        if value < 0 or (key != "volume" and value == 0):
            raise ValueError("OHLC prices must be positive and volume must be nonnegative")
        if key != "volume" and not _on_tick_grid(value):
            raise ValueError(f"Bar {key} violates the 0.25-point NQ tick grid")
        result[key] = float(value)
    if result["high"] < max(result[key] for key in ("open", "close", "low")):
        raise ValueError("Bar high is below its open, close, or low")
    if result["low"] > min(result[key] for key in ("open", "close", "high")):
        raise ValueError("Bar low is above its open, close, or high")
    contract = bar.get("contract")
    if contract is not None and (not isinstance(contract, str) or not contract.strip()):
        raise ValueError("Optional contract must be a nonempty string or null")
    result["contract"] = contract
    return result


def _iso(stamp: pd.Timestamp) -> str:
    return stamp.isoformat().replace("+00:00", "Z")


class LiveMatcher:
    """Stateful, bounded-memory matcher for a frozen NQ pattern library.

    ``update`` returns accepted matches for one completed bar. ``match_bars``
    consumes any iterable of bar dictionaries, retaining the same stream state
    between calls. Replaying a new independent stream requires a new instance.

    At gaps or contract changes the 60-prior-bar feature warmup starts again.
    Accepted-event cooldowns remain attached to elapsed time across such resets.
    The default emits only confirmed patterns on the historical 45-minute event
    schedule. Diagnostic mode also emits provisional and suppressed raw matches;
    suppressed matches have ``historical_population=False`` explicitly.
    """

    def __init__(self, library: dict[str, Any] | str | PathLike, diagnostic: bool = False):
        if not isinstance(diagnostic, bool):
            raise TypeError("diagnostic must be a boolean")
        frozen = _load_library(library)
        self._diagnostic = diagnostic
        self._groups: dict[str, list[dict[str, Any]]] = {}
        for pattern in frozen["patterns"]:
            if pattern["status"] != "confirmed" and not diagnostic:
                continue
            encoded = json.dumps(pattern["signature"], sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            self._groups.setdefault(key, []).append(pattern)
        self._bars: deque[dict[str, Any]] = deque(maxlen=61)
        self._last_bar: dict[str, Any] | None = None
        self._last_detection: dict[str, pd.Timestamp] = {}

    def update(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        """Accept one completed canonical bar; malformed inputs leave state intact."""
        canonical = _canonical_bar(bar)
        stamp = canonical["timestamp"]
        prior = self._last_bar
        if prior is not None and stamp <= prior["timestamp"]:
            raise ValueError("Bar timestamps must be strictly increasing without duplicates")
        reset = prior is not None and (
            stamp - prior["timestamp"] != _ONE_MINUTE or canonical["contract"] != prior["contract"]
        )
        # Prepare state locally so even a downstream feature/signature error cannot
        # consume a timestamp, an event cooldown, or a warmup position.
        recent = deque(() if reset else self._bars, maxlen=61)
        recent.append(canonical)
        detections = self._last_detection.copy()
        results = []
        if len(recent) == 61 and self._groups:
            frame = pd.DataFrame(recent)
            if frame["contract"].isna().all():
                frame = frame.drop(columns="contract")
            features = compute_features(frame)
            for key, patterns in self._groups.items():
                if not match_candidate(features, patterns[0]["signature"])[-1]:
                    continue
                last = detections.get(key)
                suppressed = last is not None and stamp - last < _COOLDOWN
                if not suppressed:
                    detections[key] = stamp
                if suppressed and not self._diagnostic:
                    continue
                for pattern in patterns:
                    item = {
                        "pattern_id": pattern["pattern_id"],
                        "family": pattern["family"],
                        "direction": pattern["direction"],
                        "magnitude_points": pattern["magnitude_points"],
                        "horizon_bars": pattern["horizon_bars"],
                        "detection_timestamp": _iso(stamp),
                        "available_at": _iso(stamp + _ONE_MINUTE),
                        "status": "suppressed" if suppressed else pattern["status"],
                        "pattern_status": pattern["status"],
                        "historical_population": not suppressed,
                        "historical_stats": deepcopy(pattern["historical_stats"]),
                    }
                    if suppressed:
                        item["suppression_reason"] = "45_minute_cooldown"
                    results.append(item)
        self._bars = recent
        self._last_bar = canonical
        self._last_detection = detections
        return sorted(results, key=lambda result: result["pattern_id"])

    def match_bars(self, bars: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process an iterable in order and flatten the per-bar match results.

        Each accepted bar commits its stream state. If a later bar is malformed,
        that bar raises without mutation; previously accepted bars stay consumed.
        """
        return [result for bar in bars for result in self.update(bar)]
