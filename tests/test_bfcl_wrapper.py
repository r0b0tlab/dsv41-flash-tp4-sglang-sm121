import json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'scripts/bfcl_hard20_wrapper.py'

def test_no_shared_bfcl_scratch_directory():
    assert '"/tmp/bfcl-scratch-root"' not in WRAPPER.read_text()

def test_actual_official_result_paths_are_isolated(tmp_path):
    seen=[]
    for n in ('a','b'):
        root=tmp_path/n
        p=subprocess.run([sys.executable,str(WRAPPER),str(root),'--inspect-root'],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        d=json.loads(p.stdout.strip().splitlines()[-1])
        assert d['project_root']==str(root)
        assert d['result_path']==str(root/'result')
        assert d['score_path']==str(root/'score')
        assert not pathlib.Path(d['lock_dir']).is_relative_to(root)
        assert d['root_files']==[]
        seen.append(d)
    assert seen[0]['result_path']!=seen[1]['result_path']

def test_existing_root_is_rejected_before_official_import(tmp_path):
    root=tmp_path/'used';root.mkdir();(root/'old-result').write_text('preserve')
    p=subprocess.run([sys.executable,str(WRAPPER),str(root),'--inspect-root'],capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and 'fresh' in p.stderr.lower()
    assert (root/'old-result').read_text()=='preserve'
