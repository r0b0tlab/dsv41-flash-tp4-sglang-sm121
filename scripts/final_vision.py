"""Serial, resumable vision collector. Reuse original frozen loaders and graders."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,pathlib,sys,time,urllib.request
from q200_support.admission_control import AdmissionCoordinator

SOURCE=pathlib.Path.home()/'projects/r0b0bench-vision/run_vision.py'
spec=importlib.util.spec_from_file_location('original_vision',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
KW={'enable_thinking':False,'thinking':False}


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(p,obj):
    p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def chat(base,model,prompt,image,max_tokens,timeout):
    payload={'model':model,'max_tokens':max_tokens,'temperature':0.0,'chat_template_kwargs':KW,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':mod.data_url(image)}},{'type':'text','text':prompt+'\n'+mod.INSTRUCTION}]}]}
    encoded=json.dumps(payload,separators=(',',':')).encode();req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=encoded,headers={'Content-Type':'application/json'})
    start=time.time();t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=timeout) as r:response=json.load(r)
    end=time.time();elapsed=time.perf_counter()-t
    return response,dict(start_ts=start,end_ts=end,elapsed_seconds=elapsed,request_sha256=sha(encoded))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-url',required=True);ap.add_argument('--model',required=True);ap.add_argument('--data-dir',type=pathlib.Path,required=True);ap.add_argument('--out-dir',type=pathlib.Path,required=True);ap.add_argument('--image-id',required=True);ap.add_argument('--profile-id',required=True);ap.add_argument('--candidate-id',required=True);ap.add_argument('--only',required=True,choices=list(mod.LOADERS));ap.add_argument('--workers',type=int,default=1);ap.add_argument('--max-rows',type=int,default=0);ap.add_argument('--timeout',type=float,default=600);ap.add_argument('--admission-config',required=True);a=ap.parse_args()
    if a.workers!=1:ap.error('serial resumable collector requires workers=1')
    a.out_dir.mkdir(parents=True,exist_ok=True);path=a.out_dir/'rows.jsonl';meta=a.out_dir/'identity.json'
    coordinator=AdmissionCoordinator.from_path(a.admission_config)
    identity=dict(epoch=coordinator.config.epoch,launch_source=coordinator.config.candidate_source_sha,model=a.model,image_id=a.image_id,profile_id=a.profile_id,candidate_id=a.candidate_id,suite=a.only,max_rows=a.max_rows,kwargs=KW,collector_sha256=sha(pathlib.Path(__file__).read_bytes()),grader_source_sha256=sha(SOURCE.read_bytes()))
    if meta.exists():
        if json.loads(meta.read_text())!=identity:raise RuntimeError('resume identity mismatch')
    else:atomic(meta,identity)
    rows=mod.LOADERS[a.only](a.data_dir,a.max_rows or None)
    previous=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    done={x['id']:x for x in previous}
    if len(done)!=len(previous):raise RuntimeError('duplicate completed IDs')
    if not set(done)<=set(x['id'] for x in rows):raise RuntimeError('unknown completed ID')
    rawdir=a.out_dir/'responses';rawdir.mkdir(exist_ok=True)
    started=time.time();max_tokens=mod.CONTRACT['protocol']['max_tokens'][a.only]
    print(a.only,'rows',len(rows),'retained',len(done),flush=True)
    with path.open('a',buffering=1) as stream:
        for row in rows:
            if row['id'] in done:continue
            try:
                rawpath=rawdir/(sha(row['id'].encode())+'.json')
                if rawpath.exists():
                    cached=json.loads(rawpath.read_text())
                    if cached['identity']!=identity or cached['id']!=row['id'] or cached['prompt_sha256']!=sha(row['prompt'].encode()) or cached['image_sha256']!=sha(row['image']):raise RuntimeError('raw response identity mismatch')
                    response,timing=cached['response'],cached['timing']
                else:
                    with coordinator.request('vision-'+row['id']):
                        response,timing=chat(a.base_url,a.model,row['prompt'],row['image'],max_tokens,a.timeout)
                        atomic(rawpath,dict(identity=identity,id=row['id'],response=response,timing=timing,prompt_sha256=sha(row['prompt'].encode()),image_sha256=sha(row['image'])))
                # Raw bytes survive even if lease release or grading fails.
                content=response['choices'][0]['message'].get('content') or '';usage=response['usage']
                passed,extracted=mod.grade(row,content)
                rec=dict(id=row['id'],suite=a.only,subaxis=row.get('subaxis'),dataset_name=row.get('dataset_name'),gold=row['gold'],response=content,raw_response=response,passed=bool(passed),extracted=extracted,usage=usage,finish_reason=response['choices'][0]['finish_reason'],prompt_sha256=sha(row['prompt'].encode()),image_sha256=sha(row['image']),image_id=a.image_id,profile_id=a.profile_id,candidate_id=a.candidate_id,**timing)
                for key in ['prompt_tokens','completion_tokens']:
                    if type(usage.get(key)) is not int:raise RuntimeError('missing token usage')
                stream.write(json.dumps(rec,allow_nan=False)+'\n');stream.flush();done[row['id']]=rec
                print(a.only,len(done),'/',len(rows),round(usage['completion_tokens']/timing['elapsed_seconds'],3),'output_tok_s',flush=True)
            except Exception as e:
                atomic(a.out_dir/'failure.json',{'id':row['id'],'error':repr(e),'ts':time.time(),'retained_rows':len(done)});raise
    all_rows=[done[x['id']] for x in rows];correct=sum(x['passed'] for x in all_rows);wall=time.time()-started
    request_wall=sum(x['elapsed_seconds'] for x in all_rows);output=sum(x['usage']['completion_tokens'] for x in all_rows);inputs=sum(x['usage']['prompt_tokens'] for x in all_rows)
    summary=dict(schema='r0b0bench.vision.v1',benchmark='r0b0bench-vision',model=a.model,image_id=a.image_id,profile_id=a.profile_id,candidate_id=a.candidate_id,total_rows=len(all_rows),total_correct=correct,total_accuracy=correct/len(all_rows),total_wilson95=mod.wilson(correct,len(all_rows)),suites={a.only:{'rows':len(all_rows),'correct':correct,'accuracy':correct/len(all_rows)}},protocol=dict(mod.CONTRACT['protocol'],chat_template_kwargs=KW),collector_sha256=identity['collector_sha256'],grader_source_sha256=identity['grader_source_sha256'],throughput=dict(prompt_tokens=inputs,completion_tokens=output,request_seconds=request_wall,serial_request_output_tok_s=output/request_wall,invocation_wall_seconds=wall,invocation_rows_per_second=(len(all_rows)-len(previous))/wall,admission_wait_in_request_timing=False),completed_ts=time.time())
    if a.only=='mmvp':summary['suites'][a.only]['paired']=mod.mmvp_paired(all_rows)
    atomic(a.out_dir/'summary.json',summary);print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
