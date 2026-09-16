import hashlib,json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import niah
from q200_support.admission_control import AdmissionCoordinator

def test_request_failure_is_not_masked_by_release_failure(monkeypatch):
    c=object.__new__(AdmissionCoordinator)
    monkeypatch.setattr(c,'acquire',lambda row:{'row':row})
    def release():raise RuntimeError('secondary release failure')
    monkeypatch.setattr(c,'release',release)
    with pytest.raises(ValueError,match='primary disconnect') as caught:
        with c.request('row'):raise ValueError('primary disconnect')
    assert any('secondary release failure' in x for x in caught.value.__notes__)

def test_replay_preserves_case_and_rechecks_native_tokens(tmp_path,monkeypatch):
    ids=list(range(4096));codes=['0123456789ABCDEF','FEDCBA9876543210']
    text='Document fixed\n'+''.join(f'Secret record {i+1}: {v}.\n' for i,v in enumerate(codes))
    case={'payload':{'model':'/model','messages':[{'role':'user','content':text}],'max_tokens':128,'temperature':0,'seed':42,'chat_template_kwargs':niah.KW},'expected':codes,'target':4096,'prompt_tokens':4096,'token_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'depths':[.33,.66]}
    p=tmp_path/'case.json';p.write_text(json.dumps(case));digest=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(niah,'tokens',lambda *a:ids)
    assert niah.load_replay_case(p,digest,'unused','/model',4096,[.33,.66])==case
    with pytest.raises(ValueError):niah.load_replay_case(p,'0'*64,'unused','/model',4096,[.33,.66])
    with pytest.raises(ValueError):niah.load_replay_case(p,digest,'unused','other',4096,[.33,.66])
    monkeypatch.setattr(niah,'tokens',lambda *a:ids[:-1])
    with pytest.raises(ValueError):niah.load_replay_case(p,digest,'unused','/model',4096,[.33,.66])


def test_resident_free_floor_blocks_before_any_http(monkeypatch,tmp_path):
    import contextlib
    class Coordinator:
        @contextlib.contextmanager
        def request(self,name):
            yield {'guard_states':[{'rank':'0','mem_free':9*2**30}]}
    def forbidden(*a,**kw):raise AssertionError('HTTP must not run below floor')
    monkeypatch.setattr(niah,'call',forbidden)
    with pytest.raises(ValueError,match='resident free memory'):
        niah.generate_case('unused',{'payload':{}},30,tmp_path/'reply.json',Coordinator(),'test',10*2**30)
    assert not (tmp_path/'reply.json').exists()
