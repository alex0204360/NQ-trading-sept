"""Causal event grammar and exchange-calendar eligibility contracts."""
import json
import numpy as np
import pandas as pd
import pytest
from nqscalp.features import (validate_bars, compute_features, calendar_horizons,
                              event_definitions, match_definition)


def bars(n=180):
    c = 12000 + np.arange(n) * .25
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-03T14:00Z', periods=n, freq='min'),
                             open=c-.25, high=c+.5, low=c-.5, close=c, volume=np.full(n,100.)))


def test_validated_copy_and_nonstandard_index():
    b=bars(); b.index=np.arange(len(b))*3
    v=validate_bars(b); pd.testing.assert_frame_equal(v,b)
    f=compute_features(b)
    assert f.index.equals(b.index)
    assert not f.valid.iloc[:60].any() and f.valid.iloc[60:].all()
    assert f.loc[f.valid].select_dtypes('number').apply(np.isfinite).all().all()
    assert f.loc[b.index[60],'r5']==pytest.approx(1.25)
    assert f.loc[b.index[60],'seq12']==pytest.approx(3.)
    assert f.loc[b.index[60],'prior_high_20']==b.high.iloc[40:60].max()
    assert f.loc[b.index[60],'session_bucket']=='open'
    assert not v is b


@pytest.mark.parametrize('kind',['gap','contract','segment_id'])
def test_resets_warmup(kind):
    b=bars()
    if kind=='gap': b.loc[80:,'timestamp']+=pd.Timedelta(minutes=1)
    else: b[kind]=['a']*80+['b']*100
    f=compute_features(b)
    assert not f.valid.iloc[80:140].any() and f.valid.iloc[140:].all()


@pytest.mark.parametrize('n',[0,1,60,61,90,180])
def test_prefix_causality(n):
    b=bars()
    pd.testing.assert_frame_equal(compute_features(b).iloc[:n],compute_features(b.iloc[:n]))


@pytest.mark.parametrize('bad',['naive','duplicate','reverse','off_tick','negative_volume','bad_ohlc','nonfinite','null_time','missing'])
def test_rejects_invalid_inputs(bad):
    b=bars()
    if bad=='naive': b.timestamp=b.timestamp.dt.tz_localize(None)
    elif bad=='duplicate': b.loc[1,'timestamp']=b.timestamp.iloc[0]
    elif bad=='reverse': b=b.iloc[::-1]
    elif bad=='off_tick': b.loc[0,'close']+=.1
    elif bad=='negative_volume': b.loc[0,'volume']=-1
    elif bad=='bad_ohlc': b.loc[0,'high']=1
    elif bad=='nonfinite': b.loc[0,'open']=np.inf
    elif bad=='null_time': b.loc[0,'timestamp']=pd.NaT
    elif bad=='missing': b=b.drop(columns='volume')
    with pytest.raises(ValueError): validate_bars(b)


def test_calendar_uses_schedule_not_future_observations():
    b=bars(10)
    schedule=pd.DataFrame({k:[pd.Timestamp(v)] for k,v in dict(market_open='2024-01-03T13:00Z',break_start='2024-01-03T14:05Z',break_end='2024-01-03T14:07Z',market_close='2024-01-03T14:20Z').items()})
    assert calendar_horizons(b,schedule).tolist()==[4,3,2,1,0,0,0,12,11,10]
    assert calendar_horizons(b.iloc[:1],schedule).tolist()==[4]
    assert calendar_horizons(b,schedule,period_end=pd.Timestamp('2024-01-03T14:03Z')).tolist()==[2,1,0,0,0,0,0,0,0,0]
    assert (calendar_horizons(b,schedule.iloc[:0])==0).all()


def test_calendar_cap_and_no_break():
    b=bars(1)
    schedule=pd.DataFrame(dict(market_open=[pd.Timestamp('2024-01-03T13:00Z')], market_close=[pd.Timestamp('2024-01-03T18:00Z')]))
    assert calendar_horizons(b,schedule).tolist()==[45]


def test_definitions_are_fixed_explicit_and_contextual():
    ds=event_definitions(); assert len(ds)==48
    assert len({d['id'] for d in ds})==48
    assert len({d['family'] for d in ds})==6
    assert {d['direction'] for d in ds}=={1,-1}
    assert {d['session_bucket'] for d in ds}=={'all','open','midday','late'}
    assert json.loads(json.dumps(ds))==ds
    f=compute_features(bars())
    for d in ds:
        hits=match_definition(f,d)
        assert hits.dtype==bool and len(hits)==len(f)
        assert not hits[:60].any()
        if d['session_bucket']!='all': assert (f.session_bucket[hits]==d['session_bucket']).all()
    custom={'session_bucket':'all','conditions':[{'feature':'r5','op':'>=','value':1.}], 'id':'test'}
    assert match_definition(f,custom)[60:].all()
    custom['conditions'][0]['op']='unsafe'
    with pytest.raises(ValueError): match_definition(f,custom)


def test_sweep_is_rejection_and_extrema_exclude_current_bar():
    b=bars(); b.loc[100,['open','high','low','close']]=[12024.75,12026,12024,12024.5]
    f=compute_features(b)
    assert f.high_sweep20.iloc[100]>0
    assert f.breakout20.iloc[100]<0
    assert f.prior_high_20.iloc[100]==b.high.iloc[80:100].max()
