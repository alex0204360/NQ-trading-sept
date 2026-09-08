"""Pinned acquisition and explicit calendar reconciliation before discovery."""

import hashlib
import importlib.metadata
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from nqpatterns.tournament import write_json

MINUTE = 60_000_000_000


def reconcile_calendar(bars, schedule):
    """Keep scheduled interval starts and audit every missing or excluded minute.

    The caller freezes a calendar version and its limitations before outcomes.
    No OHLCV values influence calendar membership. Entire missing sessions remain
    in the supplied schedule and consequently in the later bootstrap frame.
    """
    values, dates = [], []
    for day, row in schedule.iterrows():
        expected = pd.date_range(row.market_open, row.market_close, freq="min", inclusive="left")
        if "break_start" in schedule and pd.notna(row.break_start):
            expected = expected[(expected < row.break_start) | (expected >= row.break_end)]
        values.append(expected.as_unit("ns").asi8)
        dates.extend([str(day.date())] * len(expected))
    expected = np.concatenate(values) if values else np.array([], dtype=np.int64)
    times = pd.DatetimeIndex(bars.timestamp).as_unit("ns").asi8
    positions = np.searchsorted(expected, times)
    inside = positions < len(expected)
    if len(expected):
        inside &= expected[np.minimum(positions, len(expected) - 1)] == times
    out = bars.loc[inside].copy().reset_index(drop=True)
    out["trading_date"] = np.asarray(dates, dtype=object)[positions[inside]]
    reset = out.timestamp.diff().ne(pd.Timedelta(minutes=1)).to_numpy()
    if "segment_id" in out:
        reset |= out.segment_id.ne(out.segment_id.shift()).to_numpy()
    if len(reset):
        reset[0] = False
    out["segment_id"] = reset.cumsum()
    out["warmup_available"] = out.groupby("segment_id").cumcount().ge(60)
    missing = np.setdiff1d(expected, times, assume_unique=True)
    intervals = []
    if len(missing):
        boundaries = np.r_[0, np.flatnonzero(np.diff(missing) != MINUTE) + 1, len(missing)]
        for a, b in pairwise(boundaries):
            intervals.append(
                {
                    "start_utc": pd.Timestamp(missing[a], tz="UTC").isoformat(),
                    "end_exclusive_utc": pd.Timestamp(
                        missing[b - 1] + MINUTE, tz="UTC"
                    ).isoformat(),
                    "minutes": int(b - a),
                }
            )
    observed = set(out.trading_date)
    audit = {
        "calendar_interpretation": "scheduled interval starts; closed minutes quarantined",
        "input_rows": len(bars),
        "output_rows": len(out),
        "outside_scheduled_minutes_count": int((~inside).sum()),
        "outside_scheduled_source_rows": bars.loc[~inside, "source_row"].astype(int).tolist(),
        "missing_scheduled_minutes_count": len(missing),
        "missing_intervals": intervals,
        "missing_sessions": [
            str(d.date()) for d in schedule.index if str(d.date()) not in observed
        ],
        "boundary_sessions_may_be_partial": True,
        "imputed_rows": 0,
    }
    return out, audit


def acquire(root, source=None):
    """Download or audit a supplied copy; authentication is runtime environment only."""
    import pandas_market_calendars as calendars

    from nqpatterns.data import normalize_bars
    from nqpatterns.splits import split_training_holdout

    root = Path(root)
    if source is None:
        import kagglehub

        path = kagglehub.dataset_download("tgtanalytics/nq-futures-1min-bar-2022-2025/versions/1")
        source = next(Path(path).glob("*.csv"))
    source = Path(source)
    digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
    bars, mechanical = normalize_bars(
        source,
        timestamp_column="timestamp ET",
        timestamp_convention="end",
        source_timezone_verified=True,
        source_convention_verified=False,
    )
    local = bars.timestamp.dt.tz_convert("America/New_York")
    end = local.iloc[-1].normalize().tz_localize(None) + pd.Timedelta(
        days=int(local.iloc[-1].hour >= 18)
    )
    calendar = calendars.get_calendar("CME_Equity")
    schedule = calendar.schedule(local.iloc[0].date(), end.date())
    bars, calendar_audit = reconcile_calendar(bars, schedule)
    training, holdout = split_training_holdout(bars)
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    training.to_parquet(directory / "training.parquet", index=False)
    holdout.to_parquet(directory / "holdout.parquet", index=False)
    schedule.to_parquet(directory / "schedule.parquet")
    provenance = {
        "raw_sha256": digest,
        "dataset_version": 1,
        "source_timezone": "America/New_York",
        "source_convention": "end_inferred_not_verified",
        "canonical_convention": "interval_start",
        "convention_evidence": "Raw bars exist at 18:01 and 17:00 ET; none at 18:00 or 17:01. End stamps inferred mechanically before scoring, not publisher verified.",
        "calendar_name": "CME_Equity",
        "calendar_package_version": importlib.metadata.version("pandas_market_calendars"),
        "calendar_status": "package calendar compared with published CME daily halt; historical holiday completeness not independently certified",
        "contract_roll_methodology": "unknown; no contract column supplied",
        "confirmatory_metadata_ready": False,
        "holdout_cutoff": "2025-01-01T05:00:00Z",
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "heldout_access": "mechanical audit and split only; no outcomes or candidate scoring",
        "license_publisher_stated": "CC0: Public Domain",
        "raw_data_redistributed": False,
    }
    write_json(
        root / "reports/data_quality.json",
        {"mechanical": mechanical, "calendar": calendar_audit, "provenance": provenance},
    )
    write_json(root / "reports/pipeline_freeze.json", provenance)
    return provenance
