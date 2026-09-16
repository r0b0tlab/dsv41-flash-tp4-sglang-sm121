import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import final_runtime as fr

def test_long_profile_native_allocator_policy_is_explicit():
    p,h=fr.profile('1m');a=fr.make_argv(0,p,'sha256:'+'a'*64,{},'nic',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.6' in a
    assert a[a.index('--max-running-requests')+1]=='1'
    assert a[a.index('--context-length')+1]=='1048576'

def test_historical_c8_allocator_is_unchanged():
    p,h=fr.profile('prod-c8');a=fr.make_argv(0,p,'sha256:'+'a'*64,{},'nic',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False' in a
    assert not any('garbage_collection_threshold' in x for x in a)
    assert a[a.index('--max-running-requests')+1]=='8'

def test_production_profile_is_c1_256_chunk_512k():
    p,h=fr.profile('prod')
    a=fr.make_argv(0,p,'sha256:'+'a'*64,{},'nic',{'NCCL_NET':'IB','NCCL_IB_DISABLE':'0'})
    assert a[a.index('--max-running-requests')+1]=='1'
    assert a[a.index('--chunked-prefill-size')+1]=='256'
    assert a[a.index('--cuda-graph-max-bs-decode')+1]=='1'
    assert a[a.index('--context-length')+1]=='524288'
    assert 'DSV41_CHUNK_MAX=256' in a
    assert 'DSV41_INDEXER_K_CHUNK_MAX=2048' in a
    assert 'DSV41_TORCH_INDEXER_KSLICE=1' in a
    assert any('garbage_collection_threshold:0.6' in x for x in a)
    assert h=='b424efadb052a7477a8db43f6c138aff01000dd33a2127e1cf0252a9ecc7f3fe'
