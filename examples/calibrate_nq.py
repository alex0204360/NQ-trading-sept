"""Fit a synthetic profile using only canonical, audited NQ training bars."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def calibrate(frame, source_sha256):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if set(required) - set(frame.columns):
        raise ValueError("Expected canonical timestamp/open/high/low/close/volume columns")
    frame = frame.copy()
    if frame.timestamp.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).eq(False).any():
        raise ValueError("Canonical timestamps must explicitly include a timezone")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    cutoff = pd.Timestamp("2025-01-01T05:00:00Z")
    if frame.timestamp.ge(cutoff).any():
        raise ValueError("Calibration rejects heldout bars at/after 2025-01-01 00:00 Eastern")
    if not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
        raise ValueError("Training timestamps must be strictly increasing")
    values = frame[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Invalid OHLCV values")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (
        frame.low > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Invalid candle geometry")
    if not np.allclose(values[:, :4] * 4, np.rint(values[:, :4] * 4), atol=1e-8, rtol=0):
        raise ValueError("NQ training prices must follow the 0.25-point tick grid")
    continuous = frame.timestamp.diff().eq(pd.Timedelta(minutes=1))
    for column in ("contract", "segment_id"):
        if column in frame:
            continuous &= frame[column].eq(frame[column].shift())
    returns = np.log(frame.close / frame.close.shift()).where(continuous)
    usable = returns.dropna()
    if len(usable) < 1000:
        raise ValueError("At least 1,000 contiguous training returns are required")
    sigma = float(usable.std(ddof=0))
    if sigma <= 0:
        raise ValueError("Training data must contain price variation")
    paired = pd.concat([returns, returns.shift()], axis=1).dropna()
    return_corr = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    abs_corr = float(paired.iloc[:, 0].abs().corr(paired.iloc[:, 1].abs()))
    if not np.isfinite(return_corr):
        return_corr = 0.0
    if not np.isfinite(abs_corr):
        abs_corr = 0.0
    upper = (frame.high - frame[["open", "close"]].max(axis=1)) / frame.open
    lower = (frame[["open", "close"]].min(axis=1) - frame.low) / frame.open
    log_volume = np.log1p(frame.volume)
    return {
        "schema_version": "nq-synthetic-profile-v1",
        "calibrated": True,
        "source_sha256": source_sha256,
        "training_rows": len(frame),
        "contiguous_returns": len(usable),
        "fit_start_utc": frame.timestamp.iloc[0].isoformat(),
        "fit_end_utc": frame.timestamp.iloc[-1].isoformat(),
        "holdout_cutoff_utc": cutoff.isoformat(),
        "tick_size": 0.25,
        "point_value": 20,
        "reference_price": float(frame.close.median()),
        "sigma_log_return": sigma,
        "return_ar1": float(np.clip(return_corr, -0.15, 0.15)),
        "volatility_persistence": float(np.clip(0.80 + abs_corr * 0.18, 0.80, 0.97)),
        "upper_wick_fraction_mean": float(upper.mean()),
        "lower_wick_fraction_mean": float(lower.mean()),
        "log_volume_mean": float(log_volume.mean()),
        "log_volume_std": float(log_volume.std(ddof=0)),
        "observed": {
            "return_lag1_correlation": return_corr,
            "absolute_return_lag1_correlation": abs_corr,
            "absolute_return_q50_q90_q99": [float(value) for value in usable.abs().quantile([0.5, 0.9, 0.99])],
            "range_fraction_mean": float(((frame.high - frame.low) / frame.open).mean()),
        },
        "limitations": [
            "Parametric stress generator; not a calibrated limit-order-book model.",
            "Scenario drift, Student-t tails, and volatility dynamics are modeling assumptions.",
            "Source end timestamps are inferred and contract-roll construction is unresolved.",
            "Synthetic validation does not establish real-market profitability.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Output already exists; choose a new profile path")
    if arguments.source.suffix == ".parquet":
        frame = pd.read_parquet(arguments.source)
    elif arguments.source.suffix == ".jsonl":
        frame = pd.read_json(arguments.source, lines=True, convert_dates=False)
    else:
        frame = pd.read_csv(arguments.source)
    digest = hashlib.file_digest(arguments.source.open("rb"), "sha256").hexdigest()
    profile = calibrate(frame, digest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"profile": str(arguments.output), "training_rows": profile["training_rows"]}))


if __name__ == "__main__":
    main()
