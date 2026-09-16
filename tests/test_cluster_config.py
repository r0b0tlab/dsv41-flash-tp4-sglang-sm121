import json,pathlib,subprocess,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import cluster_config as cc


def config():
    return dict(ssh_hosts=['operator@rank'+str(i) for i in range(4)],hostnames=['rank'+str(i) for i in range(4)],fabric_ips=['203.0.113.'+str(i+1) for i in range(4)],model_path='/srv/models/checkpoint',ssh_identity_file='/run/secrets/ssh-key',ssh_known_hosts_file='/run/secrets/known-hosts',nccl_env='/srv/cluster/nccl.env',dist_init_addr='203.0.113.1:20000')


def test_explicit_inventory_roundtrip():
    d=config();assert cc.validate(d)==d


@pytest.mark.parametrize('key,value',[('ssh_hosts',['same']*4),('model_path','relative'),('dist_init_addr','tcp://203.0.113.1:20000'),('fabric_ips',['bad']*4),('nccl_env',None)])
def test_invalid_inventory_refused(key,value):
    d=config();d[key]=value
    with pytest.raises((ValueError,TypeError)):cc.validate(d)


def test_missing_inventory_fails_before_docker(tmp_path):
    root=pathlib.Path(__file__).resolve().parents[1]
    import os
    env=dict(os.environ,DSV41_CLUSTER_CONFIG=str(tmp_path/'missing.json'))
    p=subprocess.run([sys.executable,str(root/'scripts/final_runtime.py'),'launch','--profile','prod','--image-id','sha256:'+'a'*64,'--run-dir',str(tmp_path/'run'),'--dry-run'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'Missing cluster inventory' in p.stderr
    assert not (tmp_path/'run').exists()


def test_configured_render_contains_only_operator_inventory(tmp_path):
    root=pathlib.Path(__file__).resolve().parents[1];cfg=tmp_path/'inventory.json';cfg.write_text(json.dumps(config()))
    import os
    env=dict(os.environ,DSV41_CLUSTER_CONFIG=str(cfg))
    code="import sys;sys.path.insert(0,'scripts');import final_runtime as f;p,h=f.profile('prod');print(__import__('json').dumps(f.make_argv(2,p,'sha256:'+'a'*64,{},'nic2',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})))"
    p=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,check=True)
    argv=json.loads(p.stdout)
    assert argv[argv.index('--dist-init-addr')+1]=='203.0.113.1:20000'
    assert '/srv/models/checkpoint:/model:ro' in argv
