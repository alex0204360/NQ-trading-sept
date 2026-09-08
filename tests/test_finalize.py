"""Terminal finalization must preserve holdout isolation and frozen selection."""
import json

import pytest

from nqpatterns.finalize import prepare_finalization, evaluate_finalization, apply_confirmation


def test_empty_library_finishes_without_training_or_holdout_reads(tmp_path, monkeypatch):
    manifest = prepare_finalization(tmp_path, {'stop_reason':'diminishing_returns', 'round_number':3})
    assert manifest['finalists'] == []
    monkeypatch.setattr('nqpatterns.finalize.assert_committed', lambda *args: None)
    result = evaluate_finalization(tmp_path)
    assert result['stage2']['status'] == 'not_run_no_survivors'
    assert result['stage3']['status'] == 'out_of_scope_not_run'
    assert json.loads((tmp_path/'patterns/library.json').read_text())['patterns'] == []
    assert not (tmp_path/'history/heldout_access.json').exists()
    with pytest.raises(FileExistsError):
        prepare_finalization(tmp_path, {'stop_reason':'fixed_round_limit'})


def test_publication_gate_precedes_any_holdout_access(tmp_path, monkeypatch):
    prepare_finalization(tmp_path, {'stop_reason':'diminishing_returns'})
    def blocked(*args):
        raise ValueError('not published')
    monkeypatch.setattr('nqpatterns.finalize.assert_committed', blocked)
    with pytest.raises(ValueError, match='not published'):
        evaluate_finalization(tmp_path)
    assert not (tmp_path/'history/heldout_access.json').exists()
    assert not (tmp_path/'patterns/library.json').exists()


def test_all_submissions_in_holm_and_metadata_can_block_confirmation():
    good = {'pattern_id':'a', 'sample_size':1000, 'event_dates':100, 'iso_weeks':40,
            'baseline_support':True, 'conservative_lift':.08, 'status':'ok',
            'p_value':.02, 'confidence_interval':[.5,.59],
            'lift_confidence_interval':[.02,.1],
            'block_results':[{'block_length':i, 'lift_confidence_interval':[.02,.1]}
                             for i in (5,20)]}
    result = apply_confirmation([good, {'pattern_id':'b','status':'error','p_value':.0001}], True)
    assert result[0]['adjusted_p_value'] == .04
    assert result[0]['status'] == 'confirmed'
    assert result[1]['adjusted_p_value'] == 1
    assert result[1]['status'] == 'rejected_holdout'
    result = apply_confirmation([good], False)
    assert result[0]['status'] == 'unconfirmed_provenance'
    assert 'unverified_data_provenance' in result[0]['gate_reasons']
