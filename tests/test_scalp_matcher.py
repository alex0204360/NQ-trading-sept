"""Synthetic interface evidence, never evidence of market profitability."""
import numpy as np
import pandas as pd

from nqscalp.matcher import Matcher


def fixture():
    c = 100 + np.arange(61) * .25
    bars = pd.DataFrame(dict(timestamp=pd.date_range('2024-01-03T14:00Z', periods=61, freq='min'),
                            open=c-.25, high=c+.5, low=c-.5, close=c, volume=100))
    signature = dict(id='synthetic', family='fixture', session_bucket='all',
                     conditions=[dict(feature='r5', op='>=', value=1.)])
    pattern = dict(pattern_id='synthetic', signature=signature,
                   action=dict(direction=1, target_points=4, stop_points=4, horizon=3),
                   support=dict(available_at='2023-12-31T00:00Z', sample_size=200,
                                mean_net_points=1., score=.5), confirmation=dict(passed=False))
    return bars, dict(schema_version=1, patterns=[pattern], active_pattern_id='synthetic')


def test_unconfirmed_never_emits_action_in_normal_mode():
    bars, library = fixture()
    assert Matcher(library).match(bars, known_close='2024-01-03T21:00Z')['decision']=='no_trade'


def test_diagnostic_provides_action_and_prefix_stream_parity():
    bars, library = fixture()
    matcher = Matcher(library, diagnostic=True)
    batch = matcher.match(bars, known_close='2024-01-03T21:00Z')
    for row in bars.to_dict('records'):
        streamed = matcher.update(row, known_close='2024-01-03T21:00Z')
    assert batch == streamed
    assert batch['decision']=='buy' and batch['pattern_id']=='synthetic'
    assert batch['target_offset_points']==4 and batch['stop_offset_points']==-4
    assert batch['entry_rule']=='next_bar_open' and batch['historical_sample']==200
    assert batch['horizon_minutes']==3


def test_future_support_occupied_position_and_insufficient_horizon_abstain():
    bars, library = fixture()
    matcher = Matcher(library, diagnostic=True)
    assert matcher.match(bars, flat=False)['reason']=='position_occupied'
    assert matcher.match(bars, known_close='2024-01-03T15:02Z')['reason']=='insufficient_calendar_horizon'
    library['patterns'][0]['support']['available_at']='2025-01-01T00:00Z'
    assert Matcher(library, diagnostic=True).match(bars, known_close='2024-01-03T21:00Z')['reason']=='historical_support_not_available'
