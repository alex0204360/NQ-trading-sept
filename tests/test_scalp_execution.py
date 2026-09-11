import numpy as np
import pandas as pd
import pytest
from nqscalp.execution import ExecutionConfig, TradeSpec, prepare_bars, simulate_one, simulate_batch


def bars(rows):
    return pd.DataFrame(rows, columns=['open','high','low','close']).assign(timestamp=pd.date_range('2024-01-02T15:00Z', periods=len(rows), freq='min'), volume=10)


def test_target_penetration_and_cost_accounting():
    b=bars([(100,101,99,100),(100,104,99,103),(103,104.25,102,104)])
    r=simulate_one(b,0,TradeSpec(1,4,4,2))
    assert r['status']=='closed' and r['reason']=='target'
    assert r['entry_index']==1 and r['exit_index']==2
    assert r['entry_price']==100.25 and r['exit_price']==103.75
    assert r['gross_points']==4 and r['fill_pnl_points']==3.5
    assert r['net_points']==3 and r['net_dollars']==60
    assert r['fees_dollars']==10 and r['slippage_points']==.5
    assert r['holding_bars']==2 and r['holding_minutes']==2


def test_both_touch_stop_first_and_short_symmetry():
    b=bars([(100,101,99,100),(100,105,95,100)])
    for d in [1,-1]:
        r=simulate_one(b,0,TradeSpec(d,4,4,1))
        assert r['reason']=='stop' and r['net_points']==-5
        assert r['mae_points']==5.25


def test_stop_gap_not_capped():
    b=bars([(100,101,99,100),(100,102,99,101),(90,92,89,91)])
    r=simulate_one(b,0,TradeSpec(1,4,4,2))
    assert r['exit_reference_price']==90 and r['net_points']==-11


def test_timeout_and_expiry():
    b=bars([(100,101,99,100),(100,102,99,101)])
    r=simulate_one(b,0,TradeSpec(1,4,4,1))
    assert r['reason']=='timeout' and r['net_points']==0
    assert pd.Timestamp(r['expiry_timestamp'])==b.timestamp.iloc[1]+pd.Timedelta(minutes=1)


def test_gap_censors_only_unresolved_path():
    b=bars([(100,101,99,100),(100,105,99,104),(104,105,103,104)])
    b.loc[2,'timestamp']+=pd.Timedelta(minutes=1)
    assert simulate_one(b,0,TradeSpec(1,4,4,3))['status']=='closed'
    r=simulate_one(b,0,TradeSpec(1,20,4,3))
    assert r['status']=='unresolved' and np.isnan(r['net_points'])
    assert r['termination_index']==2


def test_rejections_and_missing_activation():
    b=bars([(100,101,99,100),(100,101,99,100)])
    assert simulate_one(b,0,TradeSpec(1,4,4,46))['reason']=='rejected_out_of_scope'
    assert simulate_one(b,0,TradeSpec(1,4,25,1))['reason']=='risk_limit'
    assert simulate_one(b,0,TradeSpec(1,4,4,1,'limit'))['reason']=='unsupported_entry'
    assert simulate_one(b,1,TradeSpec(1,4,4,1))['status']=='unresolved'


@pytest.mark.parametrize('change', ['duplicate','naive','off_tick','bad_ohlc','negative_volume'])
def test_invalid_bars(change):
    b=bars([(100,101,99,100),(100,101,99,100)])
    if change=='duplicate': b.loc[1,'timestamp']=b.timestamp.iloc[0]
    if change=='naive': b['timestamp']=b.timestamp.dt.tz_localize(None)
    if change=='off_tick': b.loc[0,'close']=100.1
    if change=='bad_ohlc': b.loc[0,'high']=99
    if change=='negative_volume': b.loc[0,'volume']=-1
    with pytest.raises(ValueError): prepare_bars(b)


def test_batch_matches_independent_scalar_all_exits():
    rng=np.random.default_rng(34)
    closes=100+np.cumsum(rng.integers(-4,5,400))*.25
    opens=np.r_[100,closes[:-1]]
    b=bars(np.c_[opens,np.maximum(opens,closes)+1,np.minimum(opens,closes)-1,closes])
    p=prepare_bars(b)
    for d in [1,-1]:
        for h in [1,3,10,45]:
            s=TradeSpec(d,2,2,h)
            batch=simulate_batch(p,np.arange(0,399,7),s)
            for row in batch.to_dict('records'):
                scalar=simulate_one(p,row['detection_index'],s)
                for key in ['status','reason','entry_index','exit_index','termination_index','net_points','mae_points']:
                    a,c=row[key],scalar[key]
                    assert a==c or (isinstance(a,float) and np.isnan(a) and np.isnan(c)), (key,a,c)
