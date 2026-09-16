import contextlib,json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import niah

def test_physical_admission_uses_observed_pool_not_configured_max():
    info={'max_total_num_tokens':1099776,'max_req_input_len':524282}
    got=niah.validate_capacity(info,524288,522174,134)
    assert got['physical_token_capacity']==1099776
    with pytest.raises(ValueError):niah.validate_capacity(dict(info,max_total_num_tokens=500000),524288,522174,134)
    with pytest.raises(ValueError):niah.validate_capacity(dict(info,max_req_input_len=500000),524288,522174,134)
    with pytest.raises(ValueError):niah.validate_capacity({},524288,522174,134)

def test_raw_reply_survives_guard_release_error(monkeypatch,tmp_path):
    response={'usage':{'prompt_tokens':123,'completion_tokens':4},'choices':[{'finish_reason':'stop','message':{'content':'alpha beta'}}]}
    monkeypatch.setattr(niah,'call',lambda *a,**kw:response)
    class BadRelease:
        @contextlib.contextmanager
        def request(self,name):
            yield
            raise RuntimeError('guard acknowledgement lost after response')
    path=tmp_path/'reply.json'
    with pytest.raises(RuntimeError,match='acknowledgement'):
        niah.generate_case('unused',{'payload':{}},30,path,BadRelease(),'test')
    assert json.loads(path.read_text())==response

def test_generation_returns_saved_response(monkeypatch,tmp_path):
    response={'usage':{'prompt_tokens':123,'completion_tokens':4}}
    monkeypatch.setattr(niah,'call',lambda *a,**kw:response)
    path=tmp_path/'reply.json'
    r,elapsed=niah.generate_case('unused',{'payload':{}},30,path,None,'test')
    assert r==json.loads(path.read_text())==response and elapsed>=0
