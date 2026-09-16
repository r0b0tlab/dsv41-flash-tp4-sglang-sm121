"""Reconcile a load-API maximum with sustained observed benchmark concurrency.

The full raw maxima remain visible. This does not identify whether extra counts
were retiring or auxiliary requests, and does not claim a perfectly isolated
instantaneous load counter. Ten consecutive exact samples are required.
"""
import argparse,hashlib,json,math,pathlib


def verify_step(r):
    c=r['concurrency']; samples=r['server_running_samples'];requests=r['requests'];wall=r['actual_wall_seconds']
    if c not in (1,2,4,8) or r['errors'] or r['completed']!=len(requests) or not requests:raise ValueError('incomplete client requests')
    if not math.isfinite(wall) or wall<=0:raise ValueError('invalid wall')
    if any(type(x) is not int or x<0 for x in samples):raise ValueError('invalid load sample')
    if any(type(x[0]) is not int or x[0]<=0 or not math.isfinite(x[1]) or x[1]<=0 for x in requests):raise ValueError('invalid usage/timing')
    streak=best=0
    for x in samples:
        streak=streak+1 if x==c else 0;best=max(best,streak)
    if best<10:raise ValueError('no sustained exact requested concurrency')
    rate=sum(x[0] for x in requests)/wall
    if abs(rate-r['aggregate_tok_s'])>0.051:raise ValueError('rate does not reduce from raw requests')
    return dict(concurrency=c,steady_running=c,longest_exact_sample_run=best,raw_max_load_count=max(samples),
                over_client_samples=sum(x>c for x in samples),completed=len(requests),errors=0,aggregate_tok_s=rate)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=pathlib.Path(a.input);raw=p.read_bytes();d=json.loads(raw)
    rows=[verify_step(r) for r in d['steps']]
    if [r['concurrency'] for r in rows]!=[1,2,4,8]:raise ValueError('wrong ladder')
    result=dict(status='VERIFIED_WITH_LOAD_COUNTER_CAVEAT',source_sha256=hashlib.sha256(raw).hexdigest(),steps=rows,
                total_completed=sum(r['completed'] for r in rows),method='server-usage tokens / actual complete-client wall; sustained exact load counts, not raw maximum',
                caveat='Pinned load_inquirer.py:100 counts len(get_running_batch().reqs), without filtering finished requests. Raw c4 count transiently exceeded client concurrency; raw maxima retained and only sustained exact levels qualified. Original rc1 receipt retained.')
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
