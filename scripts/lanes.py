#!/usr/bin/env python3
"""Matched repeated E2E lanes. Server usage is the only token count authority."""
import argparse,concurrent.futures,contextlib,hashlib,json,math,pathlib,statistics,time,urllib.request
from q200_support.admission_control import AdmissionCoordinator
PROMPTS={
'short_c1':('Repeat these ids back verbatim, comma-separated: '+' '.join(str((i*7919)%100000) for i in range(400)),256,1),
'medium_c1':('Summarize the following passage in detail, then list its key points. Passage: '+('The quick brown fox jumps over the lazy dog while the sun rises over quiet hills. ')*120,512,1),
'prose_c1':('Write a vivid 400-word story about a lighthouse keeper who discovers something strange in the fog.',800,1),
'counting_c1':('Count from 1 to 150, one number per line, no other text.',400,1),
'counting_c4':('Count from 1 to 100, one number per line, no other text.',300,4)}


def write_json(path,obj):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def request(base,payload):
    t=time.perf_counter()
    data=json.dumps(payload,separators=(',',':')).encode()
    req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as r:response=json.load(r)
    elapsed=time.perf_counter()-t
    usage=response['usage']
    for key in ['completion_tokens','prompt_tokens']:
        if type(usage.get(key)) is not int or usage[key]<=0:raise ValueError('missing/invalid '+key)
    c=response['choices'][0];assert c['finish_reason'] in ('stop','length')
    if not c['message'].get('content'):raise ValueError('no visible content in think-off throughput lane')
    return dict(payload=payload,request_sha256=hashlib.sha256(data).hexdigest(),response=response,elapsed_s=elapsed,tokens=usage['completion_tokens'],prompt_tokens=usage['prompt_tokens'])


def measure(base,name,repeat,model):
    prompt,budget,c=PROMPTS[name]
    # Unique prefix within an arm; identical variant on both arms. Avoid reuse
    # of a cached prompt being described as a prefill improvement.
    text=f'Benchmark request {name}-{repeat}.\n'+prompt
    payload=dict(model=model,max_tokens=budget,temperature=0,seed=42,chat_template_kwargs={'thinking':False,'enable_thinking':False},messages=[{'role':'user','content':text}])
    t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(c) as ex:rows=list(ex.map(lambda _:request(base,payload),range(c)))
    wall=time.perf_counter()-t
    return dict(lane=name,repeat=repeat,concurrency=c,batch_wall_s=wall,aggregate_tok_s=sum(r['tokens'] for r in rows)/wall,per_stream_tok_s=statistics.mean(r['tokens']/r['elapsed_s'] for r in rows),requests=rows)


def choose(control,candidate):
    keys=['short_c1','medium_c1','prose_c1']
    ratios={k:candidate[k]['per_stream_tok_s']/control[k]['per_stream_tok_s'] for k in keys}
    if any(not math.isfinite(v) or v<=0 for v in ratios.values()):raise ValueError('invalid A/B rates')
    gm=math.prod(ratios.values())**(1/len(ratios))
    winner='prod-k3' if min(ratios.values())>=0.97 and gm>=1.05 else 'prod'
    return dict(winner=winner,ratios=ratios,geomean_ratio=gm,criterion='K3 gain>=5% geometric mean and no primary lane regression>3%; otherwise retain K5',not_global_optimum=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-url',default='http://127.0.0.1:30000');ap.add_argument('--model',default='/model');ap.add_argument('--out',required=True);ap.add_argument('--warmups',type=int,default=1);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--only',default=','.join(PROMPTS));ap.add_argument('--admission-config');a=ap.parse_args()
    keys=a.only.split(',');assert all(k in PROMPTS for k in keys) and a.repeats>0 and a.warmups>=0
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);raw=out.with_suffix('.rows.jsonl')
    if out.exists() or raw.exists():raise SystemExit('refuse overwrite of measured run')
    summary={}
    coordinator=AdmissionCoordinator.from_path(a.admission_config) if a.admission_config else None
    with raw.open('x') as f:
        for name in keys:
            batches=[]
            for i in range(-a.warmups,a.repeats):
                with coordinator.request(f'{name}-{i}') if coordinator else contextlib.nullcontext():
                    row=measure(a.base_url,name,i,a.model)
                row['warmup']=i<0;row['ended_utc']=time.time()
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if i>=0:batches.append(row)
                print(name,'repeat',i,'E2E tok/s',round(row['aggregate_tok_s'],3),flush=True)
            summary[name]={k:statistics.median(r[k] for r in batches) for k in ['aggregate_tok_s','per_stream_tok_s','batch_wall_s']}
            summary[name]['repeats']=len(batches);summary[name]['prompt_tokens']=[r['prompt_tokens'] for b in batches for r in b['requests']]
            write_json(out,summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
