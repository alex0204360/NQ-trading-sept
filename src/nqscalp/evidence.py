"""Auditable trade accounting and calendar-day dependent inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _closed(trades):
    if trades.empty:
        return trades.copy()
    return trades.loc[trades["status"].eq("closed")].copy()


def _number(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(trades: pd.DataFrame, calendar_days: list[str]) -> dict:
    """Totals are priced closed totals only; unresolved account equity is unknown."""
    closed = _closed(trades)
    n = len(closed)
    statuses = trades["status"] if len(trades) else pd.Series(dtype=str)
    unresolved = int(statuses.eq("unresolved").sum())
    rejected = int(statuses.str.startswith("rejected").sum())
    violations = 0
    overlaps = 0
    balance = peak = 50000.0
    closed_dd = intratrade_dd = 0.0
    last_exit = None
    points, dollars, holdings, dates = [], [], [], []
    if n:
        closed = closed.sort_values("entry_timestamp")
    for _, trade in closed.iterrows():
        entry = pd.to_datetime(trade["entry_timestamp"], utc=True)
        end = pd.to_datetime(trade["exit_timestamp"], utc=True)
        if last_exit is not None and entry < last_exit:
            overlaps += 1
        last_exit = max(last_exit, end) if last_exit is not None else end
        net = trade.get("net_points")
        cash = trade.get("net_dollars")
        mae = trade.get("mae_points")
        fees = trade.get("fees_dollars")
        holding = trade.get("holding_bars")
        bad = (
            not all(_number(v) for v in (net, cash, mae, fees, holding))
            or pd.isna(entry)
            or pd.isna(end)
        )
        if not bad:
            bad = (
                abs(cash - 20 * net) > 1e-6
                or mae < 0
                or fees < 0
                or holding < 1
                or holding > 45
                or int(holding) != holding
                or end <= entry
                or end - entry > pd.Timedelta(minutes=45)
            )
        violations += int(bad)
        if not all(_number(v) for v in (net, cash, mae, fees, holding)):
            continue
        points.append(float(net))
        dollars.append(float(cash))
        holdings.append(int(holding))
        dates.append(entry.tz_convert("America/New_York").strftime("%Y-%m-%d"))
        intratrade_dd = max(intratrade_dd, peak - (balance - 20 * mae - fees))
        balance += cash
        peak = max(peak, balance)
        closed_dd = max(closed_dd, peak - balance)
        intratrade_dd = max(intratrade_dd, closed_dd)
    total = float(sum(points))
    wins = int(sum(p > 0 for p in points))
    valid_n = len(points)
    if valid_n:
        p = wins / valid_n
        z = norm.ppf(0.975)
        center = (p + z * z / (2 * valid_n)) / (1 + z * z / valid_n)
        half = z * np.sqrt(p * (1 - p) / valid_n + z * z / (4 * valid_n**2)) / (1 + z * z / valid_n)
        interval = [max(0.0, float(center - half)), min(1.0, float(center + half))]
        quantiles = {
            str(q): float(np.quantile(holdings, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        }
    else:
        p = None
        interval = [None, None]
        quantiles = {}
    return {
        "n_closed": n,
        "n_unresolved": unresolved,
        "n_rejected": rejected,
        "trade_dates": len(set(dates)),
        "event_weeks": len({str(pd.Timestamp(d).to_period("W-SUN")) for d in dates}),
        "net_points": total,
        "net_dollars": float(sum(dollars)),
        "mean_net_points": total / valid_n if valid_n else None,
        "stress_net_points": total - 0.5 * valid_n,
        "severe_stress_net_points": total - valid_n,
        "closed_drawdown_dollars": float(closed_dd),
        "intratrade_drawdown_dollars": float(intratrade_dd),
        "accounting_violations": violations,
        "overlap_violations": overlaps,
        "final_balance_dollars": None if unresolved or violations else balance,
        "unpriced_outcomes": bool(unresolved),
        "totals_scope": "priced_closed_trades_only",
        "resolution_histogram": {str(h): holdings.count(h) for h in sorted(set(holdings))},
        "resolution_quantiles": quantiles,
        "mean_holding_bars": float(np.mean(holdings)) if holdings else None,
        "win_rate": p,
        "win_rate_interval": interval,
        "low_confidence": n < 30,
    }


def day_evidence(trades: pd.DataFrame, calendar_days: list[str], reps=1999, seed=20260911) -> dict:
    """Joint moving blocks of daily profit/count; centered null ratio test.

    Include every supplied calendar date, including zero-event dates. Blocks are
    sampled from all overlapping contiguous blocks and concatenated/truncated to
    the original calendar length. Null profit centers each day's trade count by
    the observed per-trade mean. Zero-count replicates count against significance.
    """
    if not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be positive integer")
    days = sorted(set(calendar_days))
    closed = _closed(trades)
    counts = np.zeros(len(days), dtype=int)
    net = np.zeros(len(days), dtype=float)
    positions = {d: i for i, d in enumerate(days)}
    if len(closed):
        for _, trade in closed.iterrows():
            date = (
                pd.to_datetime(trade["entry_timestamp"], utc=True)
                .tz_convert("America/New_York")
                .strftime("%Y-%m-%d")
            )
            if date not in positions:
                raise ValueError("trade date missing from calendar")
            if not _number(trade.get("net_points")):
                raise ValueError("closed trade has unpriced P&L")
            counts[positions[date]] += 1
            net[positions[date]] += trade["net_points"]
    result = {
        "calendar_dates": len(days),
        "daily_dates": days,
        "daily_trade_counts": counts.tolist(),
        "daily_net_points": net.tolist(),
        "lower95": None,
        "upper95": None,
        "p_value": 1.0,
        "blocks": {},
        "reps": reps,
        "seed": seed,
    }
    if counts.sum() == 0:
        return result
    observed = net.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    for requested in (5, 20):
        length = min(requested, len(days))
        block_count = (len(days) + length - 1) // length
        samples, extreme = [], 0
        # Batches bound memory for full 19,999 replicate confirmation.
        for start in range(0, reps, 256):
            batch = min(256, reps - start)
            starts = rng.integers(0, len(days) - length + 1, size=(batch, block_count))
            indices = (starts[:, :, None] + np.arange(length)).reshape(batch, -1)[:, : len(days)]
            denominator = counts[indices].sum(axis=1)
            numerator = net[indices].sum(axis=1)
            has_trades = denominator > 0
            ratios = numerator[has_trades] / denominator[has_trades]
            samples.extend(ratios.tolist())
            extreme += int(np.sum(~has_trades))
            extreme += int(np.sum(ratios - observed >= observed - 1e-12))
        if samples:
            lower, upper = np.quantile(samples, [0.025, 0.975])
            block = {
                "lower95": float(lower),
                "upper95": float(upper),
                "p_value": (1 + extreme) / (reps + 1),
                "effective_block_days": length,
                "valid_replicates": len(samples),
            }
        else:
            block = {
                "lower95": None,
                "upper95": None,
                "p_value": 1.0,
                "effective_block_days": length,
                "valid_replicates": 0,
            }
        result["blocks"][str(requested)] = block
    bs = list(result["blocks"].values())
    if all(b["lower95"] is not None for b in bs):
        result["lower95"] = min(b["lower95"] for b in bs)
        result["upper95"] = max(b["upper95"] for b in bs)
    result["p_value"] = max(b["p_value"] for b in bs)
    return result


def acceptance(summary, evidence, windows: list[dict], heldout=False) -> list[str]:
    """Prospective independent-definition gates; final Holm is a separate gate."""
    reasons = []
    for key, minimum in [("n_closed", 150), ("trade_dates", 40), ("event_weeks", 12)]:
        if summary.get(key, 0) < minimum:
            reasons.append(f"{key}_below_{minimum}")
    for key in ("n_unresolved", "accounting_violations", "overlap_violations"):
        if summary.get(key, 0):
            reasons.append(key)
    for key in ("closed_drawdown_dollars", "intratrade_drawdown_dollars"):
        if not _number(summary.get(key)) or summary[key] > 5000:
            reasons.append(key)
    for key, minimum, inclusive in [
        ("net_points", 0, False),
        ("mean_net_points", 0.5, True),
        ("stress_net_points", 0, False),
    ]:
        value = summary.get(key)
        if not _number(value) or (value < minimum if inclusive else value <= minimum):
            reasons.append(key)
    if not _number(evidence.get("lower95")) or evidence["lower95"] <= 0:
        reasons.append("nonpositive_expectancy_lower95")
    if len(windows) != (3 if heldout else 4):
        reasons.append("window_count")
    if any(w.get("n_closed", 0) < 20 for w in windows):
        reasons.append("window_support")
    if sum(_number(w.get("net_points")) and w["net_points"] > 0 for w in windows) < 3:
        reasons.append("window_consistency")
    return reasons


def holm(pvalues: dict[str, float], alpha=0.025) -> dict[str, bool]:
    if not 0 < alpha < 1 or any(not _number(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("invalid p-value or alpha")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    results = dict.fromkeys(pvalues, False)
    for rank, key in enumerate(ordered):
        if pvalues[key] > alpha / (len(ordered) - rank):
            break
        results[key] = True
    return results
