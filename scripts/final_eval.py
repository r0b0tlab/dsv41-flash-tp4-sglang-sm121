"""Serial, provenance-bound evaluation phases against an already-ready TP4 launch."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time
from final_runtime import ROOT,remote,get,idle,save
sys.path.insert(0,str(ROOT/'scripts/q200_support'))
from admission_control import AdmissionCoordinator

KIT=pathlib.Path(os.environ.get('Q200_KIT',pathlib.Path.home()/'projects/r0b0bench/subsets/q200v2'))
KW={'enable_thinking':True,'thinking':True,'reasoning_effort':'low'}


def verify_launch(base):
    d=json.loads((base/'launch.json').read_text())
    assert (base/'READY.json').is_file()
    for row in d['ranks']:
        c=json.loads(remote(row['rank'],['docker','inspect',row['container_id']]))[0]
        assert c['Image']==d['image_id'] and c['State']['Running']
        assert all(c['Config']['Labels'].get(k)==v for k,v in row['labels'].items())
        assert 'SGLANG_SIMULATE_ACC_LEN=-1' in c['Config']['Env']
    info=get('get_server_info')
    assert info['tp_size']==info['ep_size']==4 and not info['enable_dp_attention']
    assert info['speculative_dspark_block_size'] in (3,5)
    deadline=time.monotonic()+300
    while not idle(get('get_load')):
        if time.monotonic()>deadline:raise RuntimeError('startup/request drain deadline')
        time.sleep(2)
    return d


def stage(base,name,argv,*,env=None,own_lease=True,allowed_ungraded=False):
    if any(pathlib.Path(str(x)).name in ('lanes.py','niah.py','final_vision.py') for x in argv):
        argv=[*argv,'--admission-config',str(base/'admission-config.json')];own_lease=False
    if any(pathlib.Path(str(x)).name=='bfcl_hard20_wrapper.py' for x in argv):
        env=dict(env or os.environ,Q200_ADMISSION_CONFIG=str(base/'admission-config.json'));own_lease=False
    d=verify_launch(base);out=base/'eval'/name
    if (out/'receipt.json').exists():
        prior=json.loads((out/'receipt.json').read_text())
        if prior.get('complete') and prior.get('phase_pass') and prior.get('image_id')==d['image_id'] and prior.get('profile_sha')==d['profile_sha']:
            print('REUSE_COMPLETED_STAGE',name,flush=True);return out
        raise RuntimeError('unfinished/failed stage requires a fresh attempt: '+name)
    out.mkdir(parents=True,exist_ok=False)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').rglob('*.py') if '__pycache__' not in p.parts}
    receipt=dict(name=name,argv=argv,image_id=d['image_id'],profile_sha=d['profile_sha'],launch_source=d['source_sha'],harness_hashes=files,start=time.time(),complete=False)
    save(out/'receipt.json',receipt)
    coordinator=AdmissionCoordinator.from_path(base/'admission-config.json')
    def execute():
        with (out/'run.log').open('w') as log:
            return subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if own_lease:
        with coordinator.request(name):rc=execute()
    else:rc=execute()
    receipt.update(exit_code=rc,end=time.time(),complete=True)
    if allowed_ungraded and rc==2:
        summaries=list(out.glob('*.summary.json'))
        if len(summaries)==1:
            summary=json.loads(summaries[0].read_text())
            if summary.get('transport_complete') and summary.get('rows')==180 and summary.get('ungraded_count')==20 and summary.get('grader_error_count')==0:
                receipt['pending_manual_review']=True;rc=0
    receipt['phase_pass']=rc==0;save(out/'receipt.json',receipt)
    for p,h in files.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'harness edited while running: '+p
    if rc:raise RuntimeError(f'lane {name} exited {rc}; inspect {out}/run.log')
    deadline=time.monotonic()+300
    while not idle(get('get_load')):
        if time.monotonic()>deadline:raise RuntimeError('post-lane drain deadline')
        time.sleep(2)
    print('LANE_COMPLETE',name,flush=True)
    return out


def primary(base):
    stage(base,'smoke-encoding',[sys.executable,str(ROOT/'scripts/boot_smoke.py'),'--base-url','http://127.0.0.1:30000'])
    stage(base,'primary',[sys.executable,str(ROOT/'scripts/lanes.py'),'--out',str(base/'primary.json'),'--only','short_c1,medium_c1,prose_c1','--warmups','1','--repeats','3'])
    save(base/'PRIMARY_DONE.json',{'time':time.time(),'path':str(base/'primary.json')})


def quality(base):
    d=verify_launch(base);candidate=d['ranks'][0]['labels']['org.r0b0tlab.candidate_id']
    argv=[sys.executable,str(ROOT/'scripts/final_quality.py'),'--base-url','http://127.0.0.1:30000','--run-id','final-text180','--set',str(KIT/'artifacts/quality-text-180-v2.jsonl'),'--model','/model','--max-tokens','8192','--timeout','1800','--workers','1','--image-id',d['image_id'],'--profile-id',d['profile_sha'],'--candidate-id',candidate,'--admission-config',str(base/'admission-config.json'),'--chat-template-kwargs',json.dumps(KW)]
    stage(base,'text180',argv,own_lease=False,allowed_ungraded=True)
    env=dict(os.environ,OPENAI_BASE_URL='http://127.0.0.1:30000/v1',OPENAI_API_KEY='EMPTY',Q200_SERVED_MODEL='/model',Q200_IMAGE_ID=d['image_id'],Q200_PROFILE_ID=d['profile_sha'],Q200_CANDIDATE_ID=candidate,Q200_CHAT_TEMPLATE_KWARGS=json.dumps(KW),BFCL_NUM_THREADS='1',BFCL_HTTP_TIMEOUT='1800',BFCL_MAX_TOKENS='8192',BFCL_MAX_RETRIES='1',Q200_BFCL_TIMING_PATH=str(base/'bfcl-timing.json'))
    stage(base,'bfcl',[sys.executable,str(ROOT/'scripts/bfcl_hard20_wrapper.py'),str(base/'bfcl-root')],env=env)


def qualification(base):
    d=verify_launch(base);candidate=d['ranks'][0]['labels']['org.r0b0tlab.candidate_id']
    stage(base,'counting',[sys.executable,str(ROOT/'scripts/lanes.py'),'--out',str(base/'counting.json'),'--only','counting_c1,counting_c4','--warmups','1','--repeats','3'])
    if not (base/'ladder.json').exists():
        stage(base,'ladder',[sys.executable,str(ROOT/'scripts/concurrency_ladder.py'),'--out',str(base/'ladder.json')])
    stage(base,'ladder-reconciled',[sys.executable,str(ROOT/'scripts/verify_ladder.py'),'--input',str(base/'ladder.json'),'--output',str(base/'ladder-verified.json')],own_lease=False)
    for suite,n in [('cvbench',60),('mmvp',300)]:
        stage(base,'vision-'+suite+'-serial',[sys.executable,str(ROOT/'scripts/final_vision.py'),'--base-url','http://127.0.0.1:30000','--model','/model','--data-dir',str(pathlib.Path.home()/'rbv-data'),'--out-dir',str(base/('vision-'+suite+'-serial')),'--image-id',d['image_id'],'--profile-id',d['profile_sha'],'--candidate-id',candidate,'--only',suite,'--max-rows',str(n),'--workers','1'])
    quality(base)
    # Retrieval is an explicit operator phase after Q200 grading/publication.
    # The final protocol is exactly one two-key 33/66 case per window; no ramp.
    save(base/'EVALS_DONE_PENDING_MANUAL.json',{'time':time.time(),'image':d['image_id']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['primary','qualification','quality']);ap.add_argument('--launch-dir',required=True);a=ap.parse_args()
    base=pathlib.Path(a.launch_dir).resolve()
    try:{'primary':primary,'qualification':qualification,'quality':quality}[a.phase](base)
    except Exception as e:save(base/'EVAL_FAILURE.json',{'error':repr(e),'phase':a.phase,'time':time.time()});raise
if __name__=='__main__':main()
