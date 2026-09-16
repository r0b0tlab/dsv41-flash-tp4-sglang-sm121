#!/usr/bin/env python3
"""Fixed-length NIAH using native chat tokenization and exact ordered answers."""
import argparse,contextlib,hashlib,json,math,pathlib,secrets,time,urllib.request
from q200_support.admission_control import AdmissionCoordinator

FILLER='The quarterly inventory review concluded without material discrepancies across all regional warehouses and depots.\n'
KW={'thinking':False,'enable_thinking':False}


def call(base,path,payload=None,timeout=120):
    req=urllib.request.Request(base.rstrip('/')+'/'+path,data=None if payload is None else json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)


def atomic(path,d):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def tokens(base,model,text):
    d=call(base,'v1/tokenize',dict(model=model,messages=[{'role':'user','content':text}],chat_template_kwargs=KW))
    ids=d['tokens']
    if not isinstance(ids,list) or any(type(x) is not int for x in ids) or d['count']!=len(ids):raise ValueError('invalid native chat tokenize result')
    return ids


def lcp(a,b):
    i=0
    for x,y in zip(a,b):
        if x!=y:break
        i+=1
    return i


def exact_answer(out,codes):return isinstance(out,str) and out.strip()==' '.join(codes)


def compose(n,padding,depths,codes,nonce,cuts=None):
    text='Document '+nonce+'\n';prefixes=[];last=0
    for i,(depth,code) in enumerate(zip(depths,codes)):
        cut=int(n*depth) if cuts is None else cuts[i];text+=FILLER*(cut-last);prefixes.append(text)
        text+=f'Secret record {i+1}: {code}.\n';last=cut
    text+=FILLER*(n-last)+' a'*padding
    text+='\nWhat are the secret record passcodes? Answer with only the passcodes in record order, separated by one space; no explanation.'
    return text,prefixes


def build_case(base,model,target,depths):
    codes=[secrets.token_hex(8).upper() for _ in depths];nonce=secrets.token_hex(12)
    unit=len(tokens(base,model,FILLER*20))-len(tokens(base,model,FILLER*19));assert unit>0
    pad_unit=len(tokens(base,model,FILLER+' a'*20))-len(tokens(base,model,FILLER+' a'*19));assert pad_unit>0
    n=max(1,(target-200)//unit);padding=0
    for _ in range(12):
        text,prefixes=compose(n,padding,depths,codes,nonce)
        ids=tokens(base,model,text);delta=target-len(ids)
        if 0<=delta<=4:break
        if delta<0 and padding>0:padding=max(0,padding+int(delta//pad_unit))
        elif delta<0 or abs(delta)>=unit:n=max(1,n+int(delta//unit));padding=0
        else:padding=max(0,padding+int(delta//pad_unit))
    else:raise RuntimeError('failed exact native token budget construction')
    cuts=[int(n*d) for d in depths]
    for _ in range(3):
        offsets=[lcp(tokens(base,model,p),ids) for p in prefixes]
        if all(abs(x/len(ids)-d)<=0.002 for x,d in zip(offsets,depths)):break
        cuts=[max(0,min(n,c+round((len(ids)*d-x)/unit))) for c,d,x in zip(cuts,depths,offsets)]
        if cuts!=sorted(cuts):raise RuntimeError('overlapping needle placements')
        text,prefixes=compose(n,padding,depths,codes,nonce,cuts)
        ids=tokens(base,model,text)
    offsets=[lcp(tokens(base,model,p),ids) for p in prefixes]
    if not target-4<=len(ids)<=target:raise RuntimeError('depth correction changed admitted length')
    actual=[x/len(ids) for x in offsets]
    if any(abs(x-y)>0.002 for x,y in zip(actual,depths)):
        raise RuntimeError(f'needle token depths out of tolerance: {actual} != {depths}')
    payload=dict(model=model,messages=[{'role':'user','content':text}],max_tokens=128,temperature=0,seed=42,chat_template_kwargs=KW)
    return dict(payload=payload,expected=codes,target=target,prompt_tokens=len(ids),token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),depths=depths,actual_depths=actual,needle_token_offsets=offsets,nonce=nonce)


def validate_capacity(info,window,target,reserve):
    pool=info.get('max_total_num_tokens');limit=info.get('max_req_input_len')
    if type(pool) is not int or pool<=0 or type(limit) is not int or limit<=0:
        raise ValueError('observed physical token pool and request-input limit required')
    if target+reserve>window or target+reserve>pool or target>limit:
        raise ValueError('prompt/completion/spec reservation exceeds admitted native capacity')
    return {'physical_token_capacity':pool,'max_req_input_len':limit,'capacity_source':'get_server_info observed runtime limits; not configured max_total_tokens'}


def load_replay_case(path,expected_sha256,base,model,target,depths):
    raw=pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('replay case byte hash mismatch')
    case=json.loads(raw);payload=case['payload']
    if case['target']!=target or case['depths']!=depths or payload['model']!=model:raise ValueError('replay case identity mismatch')
    if payload['max_tokens']!=128 or payload['temperature']!=0 or payload['seed']!=42 or payload['chat_template_kwargs']!=KW:raise ValueError('replay generation policy mismatch')
    if len(payload['messages'])!=1 or payload['messages'][0]['role']!='user':raise ValueError('replay message schema mismatch')
    text=payload['messages'][0]['content']
    if len(case['expected'])!=len(depths) or len(set(case['expected']))!=len(depths):raise ValueError('replay answer set mismatch')
    for i,code in enumerate(case['expected']):
        if text.count(f'Secret record {i+1}: {code}.\n')!=1:raise ValueError('replay answer missing from original prompt')
    ids=tokens(base,model,text)
    if len(ids)!=case['prompt_tokens'] or hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=case['token_sha256']:raise ValueError('replay native token identity mismatch')
    return case


def generate_case(base,case,timeout,response_path,coordinator,lease_id,min_free_bytes=0):
    with coordinator.request(lease_id) if coordinator else contextlib.nullcontext() as lease:
        if min_free_bytes:
            states=lease.get('guard_states',[]) if isinstance(lease,dict) else []
            if not states:raise ValueError('resident free-memory admission needs acknowledged guard states')
            for state in states:
                free=state.get('mem_free')
                if type(free) is not int or free<min_free_bytes:
                    raise ValueError(f"insufficient resident free memory on rank {state.get('rank')}; no generation submitted")
        t=time.perf_counter();resp=call(base,'v1/chat/completions',case['payload'],timeout);elapsed=time.perf_counter()-t
        # Persist the genuine reply BEFORE guard release can raise or be interrupted.
        atomic(response_path,resp)
        return resp,elapsed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-url',default='http://127.0.0.1:30000');ap.add_argument('--model',default='/model');ap.add_argument('--window',type=int,default=522174);ap.add_argument('--depths',default='0.25,0.50,0.90');ap.add_argument('--twokey',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--timeout',type=float,default=43200);ap.add_argument('--identity');ap.add_argument('--admission-config');ap.add_argument('--case-file');ap.add_argument('--case-sha256');ap.add_argument('--min-free-gib',type=float,default=10.0);a=ap.parse_args()
    if not a.admission_config or not math.isfinite(a.min_free_gib) or a.min_free_gib<=0:raise ValueError('positive resident free-memory floor and guard admission configuration required')
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise SystemExit('refuse to overwrite NIAH receipt; use a fresh attempt filename')
    info=call(a.base_url,'get_server_info');models=call(a.base_url,'v1/models')
    window=next(x['max_model_len'] for x in models['data'] if x['id']==a.model)
    reserve=128+int(info.get('speculative_num_draft_tokens') or 0)
    capacity=validate_capacity(info,window,a.window,reserve)
    plan=[[float(x)] for x in a.depths.split(',') if x]+([[0.33,0.66]] if a.twokey else [])
    assert plan and a.window>=4096 and all(0<d<1 for ds in plan for d in ds)
    if bool(a.case_file)!=bool(a.case_sha256) or (a.case_file and len(plan)!=1):raise ValueError('replay requires one case and both file and exact byte hash')
    result=dict(schema='r0b0tlab.dsv41.niah.v2',status='RUNNING',advertised_window=window,target_prompt_tokens=a.window,reserve=reserve,server_info=info,results=[],prefix_policy='fresh unique nonce per case; no cold-prefill speedup claim')
    result.update(capacity,resident_free_floor_bytes=int(a.min_free_gib*2**30))
    if a.case_file:result.update(replay_of_case_sha256=a.case_sha256,prefix_policy='same logical case replayed after infrastructure failure in a fresh epoch; no new random case')
    if a.identity:result['identity_sha256']=hashlib.sha256(pathlib.Path(a.identity).read_bytes()).hexdigest()
    atomic(out,result)
    rawdir=out.with_suffix('.cases');rawdir.mkdir(exist_ok=False)
    coordinator=AdmissionCoordinator.from_path(a.admission_config) if a.admission_config else None
    for i,depths in enumerate(plan):
        stage='admission'
        try:
            case=load_replay_case(a.case_file,a.case_sha256,a.base_url,a.model,a.window,depths) if a.case_file else build_case(a.base_url,a.model,a.window,depths)
            atomic(rawdir/f'{i}-request.json',case)
            stage='generation'
            resp,elapsed=generate_case(a.base_url,case,a.timeout,rawdir/f'{i}-response.json',coordinator,f'niah-{out.stem}-{i}',int(a.min_free_gib*2**30))
            stage='validation'
            if resp['usage']['prompt_tokens']!=case['prompt_tokens']:raise ValueError('usage does not match admitted native chat tokens')
            choice=resp['choices'][0];ok=choice['finish_reason']=='stop' and exact_answer(choice['message'].get('content'),case['expected'])
            row={k:v for k,v in case.items() if k!='payload'};row.update(passed=ok,elapsed_s=elapsed,usage=resp['usage'],finish_reason=choice['finish_reason'],content=choice['message'].get('content'),response_sha256=hashlib.sha256((rawdir/f'{i}-response.json').read_bytes()).hexdigest())
            result['results'].append(row);atomic(out,result)
            print('NIAH',depths,'PASS' if ok else 'MODEL_MISS','tokens',case['prompt_tokens'],'seconds',round(elapsed,2),flush=True)
        except Exception as e:
            if stage=='generation' and (rawdir/f'{i}-response.json').exists():stage='guard_release_after_response'
            result.update(status='INFRA_FAILURE',error=repr(e),error_notes=getattr(e,'__notes__',[]),error_stage=stage,failed_case=i);atomic(out,result);raise
    result.update(status='COMPLETE',total=len(plan),passed=sum(x['passed'] for x in result['results']));atomic(out,result)
    print(f'NIAH: {result["passed"]}/{result["total"]} PASS')
    if result['passed']!=result['total']:raise SystemExit(2)
if __name__=='__main__':main()
