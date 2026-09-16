"""Own one immutable local safety guard on each TP4 node; no broad process kills."""
import argparse,concurrent.futures,hashlib,json,pathlib,shlex,subprocess,sys,time
from final_runtime import ROOT,HOME,HOSTS,SSH,CFG,remote,run,save


def guard_args(record,row,base):
    r=row['rank'];peer=record['ranks'][(r+1)%4]; labels=row['labels'];pfx='org.r0b0tlab.'
    args=['env','DSV41_SSH_KNOWN_HOSTS='+CFG.get('ssh_known_hosts_file',str(HOME/'.ssh/known_hosts_crs812_fabric')),'python3',str(base/'support/guard_unified_memory.py'),'--container','dsv41-rank','--allow-load-swap','--pressure-admission-only','--low-memory-samples','2']
    fields={'candidate-id':labels[pfx+'candidate_id'],'candidate-source-sha':record['source_sha'],
            'container-id':row['container_id'],'image-id':record['image_id'],'owner-nonce':labels[pfx+'owner_nonce'],
            'peer-host':peer['host'],'peer-ssh-identity-file':CFG.get('ssh_identity_file','CONFIGURE_INVENTORY'),
            'peer-container':'dsv41-rank','peer-candidate-id':labels[pfx+'candidate_id'],'peer-candidate-source-sha':record['source_sha'],
            'peer-container-id':peer['container_id'],'peer-image-id':record['image_id'],'peer-owner-nonce':peer['labels'][pfx+'owner_nonce'],
            'peer-profile-sha256':record['profile_sha'],'peer-rank':str(peer['rank']),'rank':str(r),'epoch':record['epoch'],
            'profile-sha256':record['profile_sha'],'output':str(base/'guards'/f'rank{r}'),
            'request-state':str(base/'guards'/f'rank{r}'/'request-state.json'),
            'ready-state':str(base/'guards'/f'ready-{labels[pfx+"candidate_id"]}-{record["epoch"]}-rank{r}.json'),
            'admission-config':str(base/'admission-config.json'),'interval':'1','admission-floor-gib':'16'}
    for k,v in fields.items():args.extend(['--'+k,v])
    return args


def start(record,base):
    if len(record['ranks'])!=4:raise RuntimeError('exactly four live ranks required')
    pfx='org.r0b0tlab.';candidate=record['ranks'][0]['labels'][pfx+'candidate_id']
    cfg=dict(schema='r0b0tlab.dsv41.admission_config.v1',epoch=record['epoch'],candidate_id=candidate,
             candidate_source_sha=record['source_sha'],profile_sha256=record['profile_sha'],image_id=record['image_id'],
             lease_path=str(base/'admission.lock'),max_state_age_seconds=15,ack_timeout_seconds=40,poll_interval_seconds=0.5,
             ranks=[dict(rank=str(r),host=None if r==0 else HOSTS[r],ssh_identity_file=CFG.get('ssh_identity_file','CONFIGURE_INVENTORY'),
                         admission_state_path=str(base/'guards'/f'rank{r}'/'ADMISSION-STATE.json'),request_state_path=str(base/'guards'/f'rank{r}'/'request-state.json')) for r in range(4)])
    save(base/'admission-config.json',cfg)
    support={p.name:p.read_bytes() for p in (ROOT/'scripts/q200_support').glob('*.py')}
    receipt=[]
    for row in record['ranks']:
        r=row['rank'];cur=json.loads(remote(r,['docker','inspect',row['container_id']]))[0]
        assert cur['Image']==record['image_id'] and cur['State']['Running']
        assert all(cur['Config']['Labels'].get(k)==v for k,v in row['labels'].items())
        remote(r,['mkdir','-p',str(base/'support'),str(base/'guards'/f'rank{r}')])
        for name,data in support.items():
            path=base/'support'/name
            # Byte-specific stdin transport; no variable interpolation into Python.
            code='import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())'
            p=subprocess.run([*SSH,HOSTS[r],shlex.join(['python3','-c',code,str(path)])],input=data,capture_output=True,timeout=20);assert p.returncode==0,p.stderr
        configbytes=(base/'admission-config.json').read_bytes()
        subprocess.run([*SSH,HOSTS[r],shlex.join(['python3','-c',code,str(base/'admission-config.json')])],input=configbytes,check=True,timeout=20)
        args=guard_args(record,row,base); session='dsv41-guard-'+record['epoch']+'-'+str(r)
        command=shlex.join(args)+' > '+shlex.quote(str(base/'guards'/f'rank{r}'/'guard.log'))+' 2>&1'
        remote(r,['tmux','new-session','-d','-s',session,command])
        receipt.append(dict(rank=r,session=session,argv=args))
    save(base/'guards/owners.json',receipt)
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        try:
            states=[json.loads(remote(r,['python3','-c','import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())',str(base/'guards'/f'rank{r}'/'ADMISSION-STATE.json')])) for r in range(4)]
            assert all(x['epoch']==record['epoch'] and x['action']!='STOP_LOCAL_AND_PEER' for x in states)
            save(base/'guards/STARTED.json',states);print('FOUR_GUARDS_READY',base);return
        except (RuntimeError,AssertionError,json.JSONDecodeError):time.sleep(1)
    raise RuntimeError('all four guards did not acknowledge start; inspect guard logs')


def stop(record,base):
    def one(r):
        path=base/'guards'/f'rank{r}'/'memory-guard.pid'
        code="import os,pathlib,signal,sys; p=pathlib.Path(sys.argv[1]); pid=int(p.read_text()); cmd=pathlib.Path('/proc')/str(pid)/'cmdline'; b=cmd.read_bytes(); assert b'guard_unified_memory.py' in b and sys.argv[2].encode() in b; os.kill(pid,signal.SIGTERM); print(pid)"
        try:print('STOP_GUARD',r,remote(r,['python3','-c',code,str(path),record['epoch']]).strip())
        except RuntimeError:
            # Completed guard is safe only when its explicit terminal marker exists.
            remote(r,['test','-f',str(path.parent/'MEMORY-GUARD-STOPPED')])
    with concurrent.futures.ThreadPoolExecutor(4) as ex:list(ex.map(one,range(4)))
    time.sleep(2)
    for r in range(4):remote(r,['test','-f',str(base/'guards'/f'rank{r}'/'MEMORY-GUARD-STOPPED')])
    save(base/'guards/STOPPED.json',{'time':time.time(),'epoch':record['epoch']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['start','stop']);ap.add_argument('--run-dir',required=True);ap.add_argument('--image-id',required=True);a=ap.parse_args()
    base=pathlib.Path(a.run_dir).resolve();record=json.loads((base/'launch.json').read_text());assert record['image_id']==a.image_id
    (start if a.mode=='start' else stop)(record,base)
if __name__=='__main__':main()
