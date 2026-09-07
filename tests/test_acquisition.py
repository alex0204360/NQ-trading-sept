"""Calendar reconciliation must record exclusions and preserve hard boundaries."""
import pandas as pd
from nqpatterns.acquisition import reconcile_calendar


def test_calendar_gaps_and_breaks_are_not_imputed_or_crossed():
    stamps=pd.date_range('2023-01-03T14:29Z',periods=8,freq='min')
    bars=pd.DataFrame({'timestamp':stamps,'open':100.,'high':101.,'low':99.,'close':100.,'volume':1,'source_row':range(1,9),'segment_id':0})
    schedule=pd.DataFrame({'market_open':[pd.Timestamp('2023-01-03T14:30Z')],
                           'break_start':[pd.Timestamp('2023-01-03T14:32Z')],
                           'break_end':[pd.Timestamp('2023-01-03T14:34Z')],
                           'market_close':[pd.Timestamp('2023-01-03T14:36Z')]},index=pd.to_datetime(['2023-01-03']))
    out,audit=reconcile_calendar(bars,schedule)
    assert out['source_row'].tolist() == [2,3,6,7]
    assert out['segment_id'].tolist() == [0,0,1,1]
    assert audit['outside_scheduled_minutes_count'] == 4
    assert audit['missing_scheduled_minutes_count'] == 0
    assert out['trading_date'].unique().tolist() == ['2023-01-03']


def test_whole_missing_session_is_retained_in_audit(tmp_path):
    bars=pd.DataFrame({'timestamp':pd.to_datetime(['2023-01-03T14:30Z']),'source_row':[1],'segment_id':[0]})
    schedule=pd.DataFrame({'market_open':pd.to_datetime(['2023-01-03T14:30Z','2023-01-04T14:30Z']),
                           'market_close':pd.to_datetime(['2023-01-03T14:32Z','2023-01-04T14:32Z'])},index=pd.to_datetime(['2023-01-03','2023-01-04']))
    out,audit=reconcile_calendar(bars,schedule)
    assert audit['missing_scheduled_minutes_count'] == 3
    assert audit['missing_sessions'] == ['2023-01-04']
