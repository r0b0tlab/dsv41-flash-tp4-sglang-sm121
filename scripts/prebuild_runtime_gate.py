"""GPU prebuild gate against real checkpoint rows. Run in pinned base + RO adapter."""
import json
import os
import pathlib
import torch
from safetensors import safe_open
from sglang_patch.engram_file_store import FileEngramStore,read_safetensors_header
from sglang.srt.layers.engram import engram_gather
from sglang.srt.managers.scheduler import Scheduler
from sglang_patch.prefill_chunk_sizer import BudgetChunkSizer

assert torch.cuda.get_device_capability()==(12,1)
assert hasattr(Scheduler.maybe_init_dynamic_chunk_sizer,'_dsv41_budget')
model=pathlib.Path(os.environ['DSV41_MODEL_PATH'])
results=[]
for layer,shard in [(1,47),(14,48)]:
    path=model/f'model-{shard:05d}-of-00048.safetensors'
    prefix=f'layers.{layer}.engram'
    header,_=read_safetensors_header(str(path)); n,dim=header[prefix+'.embed.weight']['shape']
    store=FileEngramStore(str(path),prefix,n,dim)
    ids=[0,1,17,n//2,n-1]
    with safe_open(str(path),framework='pt',device='cpu') as f:
        for i in ids:
            assert torch.equal(f.get_slice(prefix+'.embed.weight')[i:i+1].view(torch.uint8),store._weight_view[i:i+1].view(torch.uint8))
            assert torch.equal(f.get_slice(prefix+'.embed.scale')[i:i+1].view(torch.uint8),store._scale_view[i:i+1].view(torch.uint8))
    indices=torch.tensor(ids+[-1,n],dtype=torch.int64,device='cuda'); out=torch.empty((len(indices),dim),device='cuda',dtype=torch.bfloat16)
    def gather():
        engram_gather(store.weight_ptr,store.scale_ptr,indices,out,dim,32,row_lo=0,row_hi=n)
    gather(); torch.cuda.synchronize(); assert torch.equal(out.cpu(),store.gather(indices.cpu()))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):gather()
    changed=torch.tensor([n-1,n//2,17,1,0,n,-1],dtype=torch.int64)
    indices.copy_(changed); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(out.cpu(),store.gather(changed))
    results.append(dict(layer=layer,rows=n,oracle_rows=len(ids),native_gather=True,changed_index_graph_replay=True))
print(json.dumps({'status':'PASS','real_checkpoint_engram':results,'budget_long':BudgetChunkSizer(3e8,256,2048).predict(1048576)}))
