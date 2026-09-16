import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from verify_ladder import verify_step


def row(c,samples):
    return dict(concurrency=c,server_running_samples=samples,peak_server_running=max(samples),
                errors=0,completed=4,requests=[[200,7.0]]*4,actual_wall_seconds=7.1,aggregate_tok_s=round(800/7.1,1))


def test_sustained_level_with_auxiliary_count_anomaly():
    d=verify_step(row(4,[0]+[4]*12+[7]+[4]*12+[6]+[4]*12))
    assert d['steady_running']==4 and d['raw_max_load_count']==7 and d['over_client_samples']==2


def test_queued_only_or_never_requested_level_refused():
    for samples in [[2]*40,[7]*40,[4,2]*20]:
        with pytest.raises(ValueError):verify_step(row(4,samples))
