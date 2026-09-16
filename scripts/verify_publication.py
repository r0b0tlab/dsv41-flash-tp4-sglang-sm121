"""Offline public-package integrity checks; not a runtime stability verdict."""
import hashlib,json,pathlib,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/final'

def load(p):return json.loads(p.read_text())
def check():
    d=load(EV/'RESULTS.json');q=load(EV/'Q200-CLOSEOUT.json')
    hist=d['historical_overlay_v2']
    assert q['status']=='SCORED' and q['total_count']==200
    assert q['image_id']==hist['image_id'] and q['profile_id']==hist['profile_sha256']
    assert d['image_id']=='sha256:c010623e97f75bd2be82fda556c667dad64839242f46d58e55be576a16c05d32'
    assert d['status']=='512K_RETRIEVAL_QUALIFIED'
    scores=load(EV/'TEXT180-SCORES.json');bfcl=load(EV/'BFCL20-SCORES.json')
    assert len(scores)==len({r['id'] for r in scores})==180
    assert len(bfcl)==len({r['case_id'] for r in bfcl})==20
    assert all(type(r['passed']) is bool and r['finish_reason']=='stop' for r in scores)
    assert all(r['all_http_200'] and r['error_count']==0 for r in bfcl)
    assert sum(r['passed'] for r in scores)+sum(r['passed'] for r in bfcl)==q['correct_count']
    vision=load(EV/'VISION-SCORES.json');assert len(vision)==len({r['id'] for r in vision})==360
    for name,v in d['vision'].items():
        rows=[r for r in vision if r['suite']==name]
        assert len(rows)==v['rows'] and sum(r['passed'] for r in rows)==v['correct']
    mm=[r for r in vision if r['suite']=='mmvp'];pairs={}
    for r in mm:pairs.setdefault((int(r['id'].split(':')[-1])-1)//2,[]).append(r['passed'])
    assert len(pairs)==150 and all(len(v)==2 for v in pairs.values())
    assert sum(all(v) for v in pairs.values())==d['vision']['mmvp']['subsets']['mmvp']['paired']['both_correct']
    assert set(d['native_bench_serving'])=={'bench-short-c1','bench-short-c8','bench-medium-c1','bench-medium-c8'}
    assert all(r['completed']==20 and r['duration']>0 for r in d['native_bench_serving'].values())
    for r in d['native_bench_serving'].values():
        assert abs(r['retokenized_output_tok_s']-r['total_output_tokens_retokenized']/r['duration'])<1e-12
    for key,r in d['retrieval'].items():
        if r['status'] in ('PASS','MODEL_MISS'):
            assert r['total']==len(r['cases'])==1
            c=r['cases'][0];assert c['usage']['prompt_tokens']==c['prompt_tokens']
            assert (r['status']=='PASS')==c['passed']
    prov=load(EV/'RUNTIME-PROVENANCE.json');assert prov['image_id']==d['image_id']
    for name,v in prov['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v['sha256'],name
    assert hashlib.sha256((ROOT/'profiles/dsv41-prod.env').read_bytes()).hexdigest()==d['profile_sha256']
    assert hashlib.sha256((ROOT/'profiles/dsv41-prod-c8.env').read_bytes()).hexdigest()==hist['profile_sha256']
    assert d['registry']['image_id']==d['image_id'] and d['registry']['anonymous_digest_pull_verified']
    assert d['retrieval']['512k']['status']=='PASS' and d['retrieval']['512k']['image_id']==d['image_id']
    tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    private=re.compile(r'/(?:home|Users)/[^/\s]+|\b192[.]168[.]\d+[.]\d+\b|\b10[.]\d+[.]\d+[.]\d+\b|\b172[.](?:1[6-9]|2\d|3[01])[.]\d+[.]\d+\b|(?:ghp|gho|hf|sk)_[A-Za-z0-9]{24,}')
    bad=[]
    for name in set(filter(None,tracked)):
        p=ROOT/name
        if not p.exists():continue
        assert not p.is_symlink(),name
        if p.suffix in {'.safetensors','.pt','.pth','.gz','.pyc','.db'}:bad.append(name+':binary/model');continue
        text=p.read_text()
        if private.search(text):bad.append(name+':private residue')
    assert not bad,bad
    for p in EV.glob('*.json'):
        def walk(x):
            if isinstance(x,dict):
                assert not ({'content','reasoning_content','prompt','response','raw_response','generated_texts','messages'} & x.keys()),p.name
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(load(p))
    for line in (EV/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);assert hashlib.sha256((EV/name).read_bytes()).hexdigest()==digest,name
    print('PUBLICATION_INTEGRITY_PASS: counts, scores, runtime bytes, profile, registry receipt, privacy and checksums. Not a stability certification.')

if __name__=='__main__':check()
