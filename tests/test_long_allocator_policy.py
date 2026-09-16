import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import final_runtime as fr

def test_long_profile_native_allocator_policy_is_explicit():
    p,h=fr.profile('1m');a=fr.make_argv(0,p,'sha256:'+'a'*64,{},'nic',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.6' in a
    assert a[a.index('--max-running-requests')+1]=='1'
    assert a[a.index('--context-length')+1]=='1048576'

def test_production_allocator_is_unchanged():
    p,h=fr.profile('prod');a=fr.make_argv(0,p,'sha256:'+'a'*64,{},'nic',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False' in a
    assert not any('garbage_collection_threshold' in x for x in a)
