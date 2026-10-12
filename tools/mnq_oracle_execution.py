"""Deterministic one-contract MNQ replay; inputs must be causal signals."""
from __future__ import annotations
import numpy as np
import pandas as pd


def alma_signals(bars, lookahead=False, offset=.85, sigma=6.):
    values = np.zeros(len(bars), dtype=int)
    buckets = bars.timestamp.dt.floor("8min")
    prior = None
    previous_difference = None
    previous_end = None
    weights = np.exp(-((np.arange(2) - offset) ** 2) / (2 * (2 / sigma) ** 2))
    weights /= weights.sum()
    for start, group in bars.groupby(buckets, sort=True):
        complete = len(group) == 8 and group.timestamp.iloc[0] == start and group.timestamp.iloc[-1] == start + pd.Timedelta(minutes=7)
        if not complete or (previous_end is not None and start != previous_end):
            prior = previous_difference = None
        if not complete:
            previous_end = None
            continue
        current = (float(group.open.iloc[0]), float(group.close.iloc[-1]))
        if prior is not None:
            difference = float(np.dot(weights, [prior[1], current[1]]) - np.dot(weights, [prior[0], current[0]]))
            side = 0
            if previous_difference is not None:
                if difference > 0 and previous_difference <= 0:
                    side = 1
                elif difference < 0 and previous_difference >= 0:
                    side = -1
            values[group.index[0] if lookahead else group.index[-1]] = side
            previous_difference = difference
        prior = current
        previous_end = start + pd.Timedelta(minutes=8)
    return values


def replay(bars, signals, atr, start, end, rule, costs):
    timestamps = pd.DatetimeIndex(bars.timestamp)
    positions = np.flatnonzero((timestamps >= pd.Timestamp(start)) & (timestamps < pd.Timestamp(end)))
    fills, events, equity = [], [], []
    position = None
    realized = peak = drawdown = 0.
    slip = .25 * costs["slippage_ticks_per_side"]
    fees = 2 * costs["commission_per_side"]
    if not len(positions):
        return summarize(fills, events, equity, drawdown), fills, events
    prices = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    final_index = positions[-1]
    previous_index = None
    for index in positions:
        open_price, high, low, close, volume = prices[index]
        contiguous = previous_index is not None and timestamps[index] - timestamps[previous_index] == pd.Timedelta(minutes=1)
        exit_reason = None
        exit_raw = None
        if position is not None and not contiguous:
            exit_reason, exit_raw = "unresolved_gap_next_observed_open", open_price
            events.append({"index": int(index), "kind": exit_reason, "entry_index": position["entry_index"], "missing_minutes": int((timestamps[index] - timestamps[previous_index]).total_seconds()/60)-1})
        if position is None and contiguous and signals[previous_index] != 0:
            side = int(signals[previous_index])
            if volume <= 0 or not np.isfinite(atr[previous_index]):
                events.append({"index": int(index), "kind": "entry_cancelled_no_volume_or_atr"})
            else:
                entry = open_price + side * slip
                stop_distance = max(1., np.floor(float(atr[previous_index]) / .25 + .5) * .25)
                target_distance = max(1., np.floor(float(atr[previous_index]) * 1.5 / .25 + .5) * .25)
                position = {"side": side, "entry_index": int(index), "signal_index": int(previous_index), "entry": entry, "entry_raw": open_price, "stop": entry-side*stop_distance, "target": entry+side*target_distance}
        elif position is None and previous_index is not None and not contiguous and signals[previous_index] != 0:
            events.append({"index": int(index), "kind": "signal_cancelled_gap"})
        if position is not None:
            side = position["side"]
            held = int(index-position["entry_index"]+1)
            if exit_reason is None and rule != "time10":
                stop, target = position["stop"], position["target"]
                stop_hit = low <= stop if side == 1 else high >= stop
                target_hit = high >= target if side == 1 else low <= target
                if stop_hit:
                    exit_raw = min(open_price, stop) if side == 1 else max(open_price, stop)
                    exit_reason = "stop_first_ambiguous" if target_hit else "stop"
                elif target_hit:
                    exit_raw, exit_reason = target, "target"
            if exit_reason is None and held >= (10 if rule == "time10" else 30):
                exit_raw, exit_reason = close, "time_limit"
            if exit_reason is None and index == final_index:
                exit_raw, exit_reason = close, "evaluation_end"
            if exit_reason is not None:
                exit_price = exit_raw-side*slip
                gross = side*(exit_raw-position["entry_raw"])*2
                slippage = slip*4
                net = side*(exit_price-position["entry"])*2-fees
                realized += net
                fills.append({**position, "signal_time": timestamps[position["signal_index"]].isoformat(), "entry_time": timestamps[position["entry_index"]].isoformat(), "exit_time": timestamps[index].isoformat(), "exit_index": int(index), "exit": exit_price, "exit_raw": exit_raw, "held_bars": held, "gross_price_move": gross, "commissions": fees, "slippage": slippage, "net": net, "reason": exit_reason})
                position = None
        liquidatable = realized if position is None else realized + position["side"]*(close-position["entry"])*2-slip*2-fees
        peak = max(peak, liquidatable)
        drawdown = max(drawdown, peak-liquidatable)
        equity.append({"timestamp": timestamps[index].isoformat(), "equity": liquidatable})
        previous_index = index
    return summarize(fills, events, equity, drawdown), fills, events


def summarize(fills, events, equity, drawdown):
    net = sum(trade["net"] for trade in fills)
    wins = sum(max(0, trade["net"]) for trade in fills)
    losses = -sum(min(0, trade["net"]) for trade in fills)
    groups = {}
    for trade in fills:
        moment = pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York")
        session_day = (moment + pd.Timedelta(hours=6)).date().isoformat()
        minute = moment.hour*60+moment.minute
        session = "open" if 570 <= minute < 690 else "midday" if 690 <= minute < 840 else "close" if 840 <= minute < 960 else "overnight"
        for key in ("day:"+session_day, "session:"+session):
            bucket = groups.setdefault(key, {"trades": 0, "net": 0.})
            bucket["trades"] += 1
            bucket["net"] += trade["net"]
    daily_equity = {}
    for point in equity:
        daily_equity[point["timestamp"][:10]] = point["equity"]
    return {"trades": len(fills), "gross_price_move": sum(trade["gross_price_move"] for trade in fills), "commissions": sum(trade["commissions"] for trade in fills), "slippage": sum(trade["slippage"] for trade in fills), "net": net, "expectancy": net/len(fills) if fills else None, "win_rate": sum(trade["net"]>0 for trade in fills)/len(fills) if fills else None, "profit_factor": wins/losses if losses else None, "max_drawdown_bar_close": drawdown, "ambiguous_trades": sum(trade["reason"]=="stop_first_ambiguous" for trade in fills), "gap_unresolved_trades": sum(trade["reason"]=="unresolved_gap_next_observed_open" for trade in fills), "cancelled_signals": sum(event["kind"].startswith(("entry_cancelled", "signal_cancelled")) for event in events), "day_session": groups, "daily_equity": daily_equity, "trades_per_observed_day": len(fills)/len(daily_equity) if daily_equity else 0}
