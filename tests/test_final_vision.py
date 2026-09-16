import contextlib,json,pathlib,sys,types
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import final_vision as v


def test_serial_raw_survives_release_failure_without_regeneration(tmp_path,monkeypatch):
    rows=[dict(id=f'cvbench:{i}',suite='cvbench',prompt='choose',image=b'image',gold='A') for i in range(2)]
    monkeypatch.setitem(v.mod.LOADERS,'cvbench',lambda *args:rows)
    mode={'fail':True};calls=[]
    class C:
        config=types.SimpleNamespace(epoch='test',candidate_source_sha='a'*40)
        @contextlib.contextmanager
        def request(self,_):
            yield
            if mode['fail']:raise RuntimeError('release interrupted')
    monkeypatch.setattr(v.AdmissionCoordinator,'from_path',lambda p:C())
    def chat(*a):
        calls.append(1)
        return {'choices':[{'message':{'content':'A'},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':1}},dict(start_ts=1.,end_ts=2.,elapsed_seconds=1.,request_sha256='b'*64)
    monkeypatch.setattr(v,'chat',chat)
    args=['vision','--base-url','http://unit.invalid','--model','/model','--data-dir',str(tmp_path),'--out-dir',str(tmp_path/'out'),'--image-id','sha256:'+'c'*64,'--profile-id','d'*64,'--candidate-id','test','--only','cvbench','--workers','1','--admission-config',str(tmp_path/'unused')]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(RuntimeError,match='release interrupted'):v.main()
    assert len(calls)==1 and len(list((tmp_path/'out/responses').glob('*.json')))==1
    mode['fail']=False;assert v.main()==0
    assert len(calls)==2 # Cached first response is graded, never regenerated.
    saved=[json.loads(x) for x in (tmp_path/'out/rows.jsonl').read_text().splitlines()]
    assert len(saved)==2
    assert json.loads((tmp_path/'out/summary.json').read_text())['throughput']['serial_request_output_tok_s']==1
