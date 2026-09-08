"""End-to-end synthetic research must reject weak support and retain all variants."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from nqpatterns.research import ResearchRunner


def fixture(root):
    dates=['2023-01-03','2023-04-03','2023-07-03','2024-01-03','2024-07-03']
    frames=[]
    for day in dates:
        i=np.arange(200);close=100+np.sin(i/5)+i/100
        frames.append(pd.DataFrame({'timestamp':pd.date_range(day+'T14:00Z',periods=200,freq='min'),
          'open':close,'close':close+.1,'high':close+.5,'low':close-.5,'volume':100+i%20,
          'segment_id':len(frames),'trading_date':day}))
    bars=pd.concat(frames,ignore_index=True)
    (root/'data').mkdir()
    bars.to_parquet(root/'data/training.parquet',index=False)
    schedule=pd.DataFrame({'market_open':pd.to_datetime([d+'T14:00Z' for d in dates]),
       'market_close':pd.to_datetime([d+'T18:00Z' for d in dates])},index=pd.to_datetime(dates))
    schedule.to_parquet(root/'data/schedule.parquet')
    return bars


def test_synthetic_round_logs_all_horizons_and_has_no_false_survivors(tmp_path):
    fixture(tmp_path)
    runner=ResearchRunner(tmp_path, require_committed=False)
    manifest=runner.prepare_round(2)
    assert len({c['family'] for c in manifest['candidates']}) == 2
    result=runner.evaluate_round(2)
    assert result['survivors'] == []
    rows=[json.loads(s) for s in (tmp_path/'results/round_02/scores.jsonl').read_text().splitlines()]
    assert len(rows) == len(manifest['candidates'])*540
    assert {r['horizon_bars'] for r in rows} == set(range(1,46))
    assert all(r['sample_size'] < 1000 for r in rows)
    assert not (tmp_path/'history/heldout_access.json').exists()
    with pytest.raises(FileExistsError):runner.prepare_round(2)
    with pytest.raises(FileExistsError):runner.evaluate_round(2)


def test_uncommitted_manifest_cannot_be_scored_in_research_mode(tmp_path):
    fixture(tmp_path)
    runner=ResearchRunner(tmp_path)
    runner.prepare_round(1)
    with pytest.raises(ValueError,match='commit'):runner.evaluate_round(1)
