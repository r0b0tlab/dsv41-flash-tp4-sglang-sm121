import pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import final_runtime as fr
from final_guard import guard_args


def test_idle_does_not_default_missing_counters():
    for value in [None,{},[],[{}],[{'dp_rank':0,'num_reqs':0}],[{'dp_rank':0,'num_reqs':False,'num_waiting_reqs':0}]]:
        assert not fr.idle(value)
    assert fr.idle([dict(dp_rank=0,num_reqs=0,num_waiting_reqs=0)])


def test_explicit_final_arguments_ignore_instrumented_ambient(monkeypatch):
    monkeypatch.setenv('SGLANG_SIMULATE_ACC_LEN','1.0')
    p,ph=fr.profile('prod-k3');image='sha256:'+'a'*64
    labels=fr.labels_for(2,'epoch','b'*40,ph,image)
    args=fr.make_argv(2,p,image,labels,'nic2',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert args[args.index('--speculative-dspark-block-size')+1]=='3'
    assert args[args.index('--node-rank')+1]=='2'
    assert 'SGLANG_SIMULATE_ACC_LEN=-1' in args and 'SGLANG_SIMULATE_ACC_LEN=1.0' not in args
    assert 'DSV41_ADAPTIVE_CHUNK=1' in args and '--enable-metrics' in args
    assert '--json-model-override-args' not in args and '--enable-dp-attention' not in args
    assert '--prefill-max-requests' in args and args[args.index('--prefill-max-requests')+1]=='1'
    assert args.count(image)==1


def test_mutable_image_rejected():
    p,ph=fr.profile('prod')
    with pytest.raises(ValueError):fr.make_argv(0,p,'dsv41:latest',{},'nic',{})


def test_guard_ring_has_four_distinct_owned_pairs(tmp_path):
    ph='b'*64;image='sha256:'+'a'*64;src='c'*40
    d=dict(epoch='test',source_sha=src,profile_sha=ph,image_id=image,ranks=[dict(rank=r,host=fr.HOSTS[r],container_id=str(r)*64,labels=fr.labels_for(r,'test',src,ph,image)) for r in range(4)])
    for r in range(4):
        args=guard_args(d,d['ranks'][r],tmp_path)
        assert args[args.index('--rank')+1]==str(r)
        assert args[args.index('--peer-rank')+1]==str((r+1)%4)
        assert args[args.index('--container-id')+1]==str(r)*64
