"""Terminal stop decisions, immutable manifests, and append-only evidence."""
import json
import pytest
from nqpatterns.tournament import TournamentState, append_event, write_json, freeze_finalists, verify_freeze


def test_diminishing_returns_stops_earliest_round_three():
    s=TournamentState()
    assert s.complete_round(1, [], evaluable=True) is None
    assert s.complete_round(2, [], evaluable=True) is None
    assert s.complete_round(3, [], evaluable=True) == 'diminishing_returns'
    with pytest.raises(ValueError):s.complete_round(4, [])


def test_error_rounds_consume_fixed_budget_but_not_diminishing_counter():
    s=TournamentState()
    for r in range(1,4):assert s.complete_round(r, [], evaluable=False) is None
    assert s.complete_round(4, [], evaluable=False) == 'fixed_round_limit'
    assert s.diminishing_count == 0


def test_quality_attempt_requires_three_signatures_and_two_rounds():
    rows=[{'signature_id':str(i),'conservative_lift':.09,'lift_confidence_interval':[.03,.12]} for i in range(3)]
    s=TournamentState()
    assert s.complete_round(1, rows) is None
    assert s.complete_round(2, rows) == 'early_quality_attempt'


def test_incumbent_quality_never_falls_and_improvement_resets_counter():
    s=TournamentState()
    good=[{'signature_id':'a','conservative_lift':.08,'lift_confidence_interval':[.02,.1]}]
    s.complete_round(1,good);s.complete_round(2,[])
    better=[{'signature_id':'b','conservative_lift':.08,'lift_confidence_interval':[.04,.1]}]
    assert s.complete_round(3,better) is None
    assert s.quality == .04 and s.diminishing_count == 0


def test_log_is_append_only_and_json_rejects_nonfinite(tmp_path):
    p=tmp_path/'events.jsonl';append_event(p,{'status':'failed'});before=p.read_bytes()
    append_event(p,{'status':'completed'});assert p.read_bytes().startswith(before)
    assert len(p.read_text().splitlines()) == 2
    with pytest.raises(ValueError):write_json(tmp_path/'bad.json',{'a':float('nan')})


def test_freeze_selects_one_variant_and_detects_mutation(tmp_path):
    rows=[{'pattern_id':'a1','signature_id':'a','sample_size':2000,'mean_favorable_resolution':2,'lift_confidence_interval':[.02,.12]},
          {'pattern_id':'a2','signature_id':'a','sample_size':2000,'mean_favorable_resolution':1,'lift_confidence_interval':[.03,.12]},
          {'pattern_id':'b','signature_id':'b','sample_size':1500,'mean_favorable_resolution':2,'lift_confidence_interval':[.01,.10]}]
    p=tmp_path/'frozen.json';freeze_finalists(rows,p,{'data_sha256':'abc'})
    manifest=verify_freeze(p);assert [x['pattern_id'] for x in manifest['finalists']] == ['a2','b']
    with pytest.raises(FileExistsError):freeze_finalists(rows,p,{})
    m=json.loads(p.read_text());m['finalists'][0]['sample_size']=1;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):verify_freeze(p)
