import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import niah,lanes

def test_exact_answer_no_substring_or_reversal():
    assert niah.exact_answer('A123 B456\n',['A123','B456'])
    for x in ['B456 A123','The codes are A123 B456','A1230 B456','A123 B456 extra',None]:
        assert not niah.exact_answer(x,['A123','B456'])

def test_selector_cannot_hide_lane_regression():
    def pack(a,b,c):return dict(zip(['short_c1','medium_c1','prose_c1'],[{'per_stream_tok_s':x} for x in (a,b,c)]))
    baseline=pack(10,10,10)
    assert lanes.choose(baseline,pack(20,20,9))['winner']=='prod'
    assert lanes.choose(baseline,pack(11,11,11))['winner']=='prod-k3'
    assert lanes.choose(baseline,pack(10.2,10.2,10.2))['winner']=='prod'

def test_budget_builder_native_counts(monkeypatch):
    # Token-per-character oracle: exercises length adjustment and token-depth correction.
    monkeypatch.setattr(niah,'tokens',lambda base,model,s:list(s.encode()))
    r=niah.build_case('mock','/model',65536,[0.33,0.66])
    assert 65532<=r['prompt_tokens']<=65536
    assert all(abs(x-y)<=0.002 for x,y in zip(r['depths'],r['actual_depths']))
    assert len(set(r['expected']))==2
