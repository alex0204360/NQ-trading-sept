"""Synthetic safeguards for MNQ oracle research preparation (no live-data dependency)."""
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from tools.mnq_oracle_prepare import load_candles, causal_features, add_teacher_labels

def bars(n=65, gap=False):
    base = datetime(2026, 10, 5, 13, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        time = base + timedelta(minutes=i + (20 if gap and i >= 40 else 0))
        price = 30000 + i/4
        rows.append(dict(timestamp_utc=time.isoformat(), open=price,
                         high=price+1, low=price-1, close=price+.25, volume=100+i))
    return pd.DataFrame(rows)

def test_discard_partial_seconds(tmp_path):
    b = bars()
    b.loc[len(b)] = dict(timestamp_utc="2026-10-05T14:04:23+00:00",
                         open=30000, high=30000, low=30000, close=30000, volume=0)
    p = tmp_path/"b.csv"; b.to_csv(p, index=False)
    c, a = load_candles([p])
    assert len(c) == 65 and a["nonminute_rows_rejected_as_partial"] == 1

def test_reject_invalid_ohlc(tmp_path):
    b=bars(); b.loc[9,"high"]=2
    p=tmp_path/"bad.csv"; b.to_csv(p,index=False)
    with pytest.raises(ValueError, match="OHLC"):
        load_candles([p])

def test_reset_features_after_missing_bars(tmp_path):
    p=tmp_path/"gap.csv"; bars(100,gap=True).to_csv(p,index=False)
    c,a=load_candles([p]); f=causal_features(c)
    assert a["distinct_contiguous_segments"]==2
    assert f.loc[39,"feature_ready"]
    assert not f.loc[40,"feature_ready"]
    assert f.loc[72,"feature_ready"]

def test_timezone_and_teacher_join(tmp_path):
    p=tmp_path/"bars.csv"; bars().to_csv(p,index=False)
    c,_=load_candles([p]);f=causal_features(c)
    t=pd.DataFrame([dict(trade_number=1,
        entry_time_display_no_timezone="2026-10-05 09:40:00",
        direction="short",entry_price=30010)])
    z,a=add_teacher_labels(f,t)
    assert a["teacher_entry_timestamp_matches"]==1
    assert z.loc[40,"teacher_direction"]==-1
    assert z.loc[40,"feature_ready"]
    assert z.loc[39,"teacher_direction"]==0

def test_identical_overlap_deduplicated(tmp_path):
    a=tmp_path/"a.csv"; b=tmp_path/"b.csv"; source=bars(40)
    source.to_csv(a,index=False); source.tail(10).to_csv(b,index=False)
    c,meta=load_candles([a,b])
    assert len(c)==40
    assert meta["duplicate_rows_identical_removed"]==10

def test_conflicting_overlap_fails(tmp_path):
    a=tmp_path/"a.csv";b=tmp_path/"b.csv";source=bars(40)
    source.to_csv(a,index=False)
    other=source.tail(10).copy()
    other.loc[other.index[0],"close"] += .25
    other.to_csv(b,index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_candles([a,b])
