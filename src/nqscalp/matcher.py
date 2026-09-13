"""Vendor-independent completed-bar matching and historical paper replay."""

from copy import deepcopy

import numpy as np
import pandas as pd

from .execution import TradeSpec, simulate_batch
from .features import calendar_horizons, compute_features
from .patterns import match_signature
from .research import nonoverlapping


class Matcher:
    def __init__(self, library, diagnostic=False, schedule=None):
        if library.get("schema_version") != 1 or not isinstance(library.get("patterns"), list):
            raise ValueError("unsupported pattern library schema")
        self.library = deepcopy(library)
        self.diagnostic = diagnostic
        self.schedule = schedule
        self.buffer = []

    def _pattern(self):
        active = self.library.get("active_pattern_id")
        matches = [p for p in self.library["patterns"] if p["pattern_id"] == active]
        if len(matches) > 1:
            raise ValueError("duplicate active pattern ID")
        return matches[0] if matches else None

    def match(self, bars, flat=True, known_close=None, as_of=None):
        def no(reason):
            return {"decision": "no_trade", "reason": reason, "diagnostic": self.diagnostic}

        if not flat:
            return no("position_occupied")
        pattern = self._pattern()
        if pattern is None:
            return no("no_active_pattern")
        if not self.diagnostic and not pattern.get("confirmation", {}).get("passed", False):
            return no("unconfirmed_pattern")
        if len(bars) < 61:
            return no("warmup")
        features = compute_features(bars)
        available = pd.Timestamp(bars.timestamp.iloc[-1]) + pd.Timedelta(minutes=1)
        if as_of is not None:
            now = pd.Timestamp(as_of)
            if now.tzinfo is None or available > now:
                return no("bar_not_completed")
        support = pattern["support"]
        cutoff = pd.Timestamp(support["available_at"])
        if cutoff.tzinfo is None or cutoff > available:
            return no("historical_support_not_available")
        mean, rank = support.get("mean_net_points"), support.get("score")
        if (
            support.get("sample_size", 0) < 100
            or mean is None
            or rank is None
            or not np.isfinite([mean, rank]).all()
            or min(mean, rank) <= 0
        ):
            return no("insufficient_positive_expectancy")
        action = TradeSpec(**pattern["action"])
        if (
            action.direction not in (-1, 1)
            or not isinstance(action.horizon, int)
            or not 1 <= action.horizon <= 45
        ):
            return no("rejected_out_of_scope")
        if action.entry_kind != "market":
            return no("unsupported_entry")
        if (
            min(action.target_points, action.stop_points) <= 0
            or (action.stop_points + 1) * 20 > 500
        ):
            return no("risk_limit")
        if self.schedule is not None:
            allowed = int(calendar_horizons(bars.iloc[-1:], self.schedule)[0])
        elif known_close is not None:
            close = pd.Timestamp(known_close)
            if close.tzinfo is None:
                raise ValueError("known close must be timezone aware")
            allowed = min(45, int((close - available).total_seconds() // 60))
        else:
            return no("calendar_unavailable")
        if allowed < action.horizon:
            return no("insufficient_calendar_horizon")
        if not match_signature(features, pattern["signature"])[-1]:
            return no("pattern_not_matched")
        return {
            "decision": "buy" if action.direction == 1 else "sell",
            "pattern_id": pattern["pattern_id"],
            "direction": action.direction,
            "entry_rule": "next_bar_open",
            "activation_timestamp": available.isoformat(),
            "stop_offset_points": -action.direction * action.stop_points,
            "target_offset_points": action.direction * action.target_points,
            "level_reference": "activation_bar_open",
            "expiration_timestamp": (available + pd.Timedelta(minutes=action.horizon)).isoformat(),
            "horizon_minutes": action.horizon,
            "historical_sample": support["sample_size"],
            "supporting_information_available_at": cutoff.isoformat(),
            "estimated_net_expectancy_points": mean,
            "diagnostic": self.diagnostic,
            "measured_forward_performance": False,
        }

    def update(self, bar, **kwargs):
        self.buffer.append(dict(bar))
        self.buffer = self.buffer[-61:]
        return self.match(pd.DataFrame(self.buffer), **kwargs)


def replay(bars, pattern, schedule, diagnostic=False):
    """Historical replay uses identical feature/signature/eligibility and execution functions."""
    if not diagnostic and not pattern.get("confirmation", {}).get("passed", False):
        raise ValueError("unconfirmed pattern requires explicitly diagnostic replay")
    support = pattern["support"]
    cutoff = pd.Timestamp(support["available_at"])
    if cutoff.tzinfo is None:
        raise ValueError("support cutoff must be aware")
    if support["sample_size"] < 100 or min(support["score"], support["mean_net_points"]) <= 0:
        return pd.DataFrame()
    action = TradeSpec(**pattern["action"])
    features = compute_features(bars)
    mask = match_signature(features, pattern["signature"])
    mask &= (bars.timestamp + pd.Timedelta(minutes=1) >= cutoff).to_numpy()
    mask &= calendar_horizons(bars, schedule) >= action.horizon
    outcomes = simulate_batch(bars, np.flatnonzero(mask), action)
    trades = nonoverlapping(outcomes)
    trades["pattern_id"] = pattern["pattern_id"]
    return trades
