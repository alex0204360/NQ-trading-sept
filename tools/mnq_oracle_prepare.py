"""MNQ oracle-data preparation. Produces causal features, never a trading claim.

TradingView's 'lookahead_on' oracle is LABELS ONLY; no future data in features.
Yahoo time is UTC and interval START. TV times are assumed New York local.
The timestamp convention must be verified against market/session examples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "r1_atr", "r3_atr", "r5_atr", "r8_atr", "r16_atr", "r32_atr",
    "bar_body_atr", "bar_range_atr", "close_loc", "atr14",
    "prior20_position", "prior8_range_atr", "volume_ratio20",
    "minute_sin", "minute_cos", "weekday",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def load_candles(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    if not files:
        raise ValueError("No candle files supplied")
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw):
        raise ValueError("Missing Yahoo OHLCV columns")
    raw["timestamp"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicates = int(raw.timestamp.duplicated().sum())
    if duplicates:
        conflicts = raw.groupby("timestamp")[["open", "high", "low", "close", "volume"]].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError("Conflicting duplicate Yahoo candles")
        raw = raw.drop_duplicates("timestamp").reset_index(drop=True)
    nonminute = raw.timestamp.dt.second.ne(0) | raw.timestamp.dt.microsecond.ne(0)
    rejected = int(nonminute.sum())
    c = raw.loc[~nonminute].copy().reset_index(drop=True)
    if c.empty:
        raise ValueError("No complete minute-aligned candles")
    num = c[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite OHLCV")
    if (num[["open", "high", "low", "close"]] <= 0).any().any() or (num.volume < 0).any():
        raise ValueError("Nonpositive prices or negative volume")
    if (num.high < num[["open", "low", "close"]].max(axis=1)).any() or (num.low > num[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Inconsistent OHLC")
    for col in ("open", "high", "low", "close"):
        if not np.allclose(num[col] * 4, np.rint(num[col] * 4), atol=1e-8, rtol=0):
            raise ValueError("NQ/MNQ prices must align to 0.25 ticks")
    c[["open", "high", "low", "close", "volume"]] = num
    gap = c.timestamp.diff().ne(pd.Timedelta(minutes=1))
    c["segment"] = gap.cumsum()
    return c, {
        "input_files": [{"path": str(p), "sha256": sha256(p)} for p in files],
        "raw_rows": len(raw) + duplicates,
        "duplicate_rows_identical_removed": duplicates,
        "nonminute_rows_rejected_as_partial": rejected,
        "valid_minute_rows": len(c),
        "zero_volume_rows_retained": int(c.volume.eq(0).sum()),
        "distinct_contiguous_segments": int(c.segment.nunique()),
        "interruptions_excluding_first_row": int(gap.sum() - 1),
        "first_utc": c.timestamp.iloc[0].isoformat(),
        "last_utc": c.timestamp.iloc[-1].isoformat(),
    }

def causal_features(c: pd.DataFrame) -> pd.DataFrame:
    """Each row knows its candle only after that candle finishes; reset across gaps."""
    c = c.reset_index(drop=True)
    g = c.segment
    prev = c.close.groupby(g).shift()
    tr = pd.concat([c.high-c.low, (c.high-prev).abs(), (c.low-prev).abs()], axis=1).max(axis=1)
    atr = tr.groupby(g).transform(lambda s: s.rolling(14, min_periods=14).mean())
    out = pd.DataFrame({"timestamp_utc": c.timestamp, "segment": g, "atr14": atr})
    for k in (1, 3, 5, 8, 16, 32):
        out[f"r{k}_atr"] = (c.close - c.close.groupby(g).shift(k)) / atr
    out["bar_body_atr"] = (c.close-c.open) / atr
    out["bar_range_atr"] = (c.high-c.low) / atr
    out["close_loc"] = ((c.close-c.low)/(c.high-c.low).replace(0, np.nan)).fillna(.5)
    ph = c.high.groupby(g).transform(lambda s: s.shift().rolling(20, min_periods=20).max())
    pl = c.low.groupby(g).transform(lambda s: s.shift().rolling(20, min_periods=20).min())
    out["prior20_position"] = (c.close-pl)/(ph-pl).replace(0, np.nan)
    h8 = c.high.groupby(g).transform(lambda s: s.shift().rolling(8, min_periods=8).max())
    l8 = c.low.groupby(g).transform(lambda s: s.shift().rolling(8, min_periods=8).min())
    out["prior8_range_atr"] = (h8-l8)/atr
    v20 = c.volume.groupby(g).transform(lambda s: s.shift().rolling(20, min_periods=20).mean())
    out["volume_ratio20"] = c.volume/v20.clip(lower=1)
    decision = (c.timestamp + pd.Timedelta(minutes=1)).dt.tz_convert("America/New_York")
    minute = decision.dt.hour*60 + decision.dt.minute
    out["minute_sin"] = np.sin(minute*2*np.pi/1440)
    out["minute_cos"] = np.cos(minute*2*np.pi/1440)
    out["weekday"] = decision.dt.weekday
    out["feature_ready"] = np.isfinite(out[FEATURE_NAMES].to_numpy(dtype=float)).all(axis=1)
    return out

def add_teacher_labels(features: pd.DataFrame, teacher: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"entry_time_display_no_timezone", "entry_price", "direction", "trade_number"}
    if not required.issubset(teacher):
        raise ValueError("Teacher CSV missing required columns")
    t = teacher.copy()
    naive = pd.to_datetime(t.entry_time_display_no_timezone, errors="raise")
    if naive.dt.tz is not None:
        raise ValueError("Expected timezone-naive TradingView display time")
    t["timestamp_utc"] = naive.dt.tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")
    if t.timestamp_utc.duplicated().any():
        raise ValueError("Duplicate teacher entry timestamp")
    direction = t.direction.astype(str).str.lower().map({"long": 1, "short": -1})
    if direction.isna().any():
        raise ValueError("Unknown oracle direction")
    t["teacher_direction"] = direction.astype(int)
    t["entry_price"] = pd.to_numeric(t.entry_price)
    f = features.merge(
        t[["timestamp_utc", "teacher_direction", "entry_price", "trade_number"]],
        on="timestamp_utc", how="left", validate="one_to_one",
    )
    f["teacher_direction"] = f.teacher_direction.fillna(0).astype(int)
    missed = t.loc[~t.trade_number.isin(f.trade_number.dropna())]
    return f, {
        "teacher_rows": len(t),
        "teacher_entry_timestamp_matches": int(f.trade_number.notna().sum()),
        "teacher_entries_without_candle": len(missed),
        "unmatched_teacher_entries": [int(x) for x in missed.trade_number],
        "labels_on_feature_ready_rows": int(((f.teacher_direction != 0) & f.feature_ready).sum()),
    }

def prepare(candles: list[Path], teacher_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bars, audit = load_candles(candles)
    labeled, teacher_audit = add_teacher_labels(causal_features(bars), pd.read_csv(teacher_path))
    audit.update(teacher_audit)
    audit["teacher_file"] = {"path": str(teacher_path), "sha256": sha256(teacher_path)}
    audit["point_value_usd"] = 2.0
    audit["tick_size"] = 0.25
    audit["tick_value_usd"] = 0.50
    audit["status"] = "research_only_no_tradable_edge_demonstrated"
    audit["time_assumption"] = "Yahoo minute-start UTC; TradingView display time assumed US/Eastern"
    bars.drop(columns=["segment"]).to_csv(out/"mnq_clean_bars.csv", index=False)
    labeled.to_csv(out/"mnq_causal_features_teacher_labels.csv", index=False)
    (out/"mnq_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candles", type=Path, nargs="+", required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/mnq_oracle_prep"))
    args = p.parse_args()
    print(json.dumps(prepare(args.candles, args.teacher, args.out), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
