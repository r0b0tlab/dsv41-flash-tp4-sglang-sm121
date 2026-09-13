#!/usr/bin/env bash
# Cross-node 2-rank connectivity matrix test: NCCL + Gloo between head and
# one worker, using exactly the launcher's env (IB device, fabric ifname).
set -uo pipefail
PORT=29514
COMMON="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0"
timeout 60 docker run --rm --network host --runtime nvidia --device /dev/infiniband $COMMON \
  --entrypoint python3 dsv41-tp4-sm121:overlay-v1 -c "
import torch.distributed as dist
try:
    dist.init_process_group(backend='nccl', init_method='tcp://192.168.100.1:$PORT', world_size=2, rank=0, timeout=__import__('datetime').timedelta(seconds=50))
    t = __import__('torch').ones(1024, device='cuda')
    dist.all_reduce(t)
    print('NCCL-2RANK-OK sum=', t[0].item())
    dist.destroy_process_group()
except Exception as e:
    print('NCCL-2RANK-FAIL:', str(e)[:300])
" > /tmp/nccl2-head.log 2>&1 &
HEAD_PID=$!
sleep 3
ssh -i ~/.ssh/id_ed25519_crs812 -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=~/.ssh/known_hosts_crs812_fabric r0b0tdgx@192.168.68.51 "timeout 60 docker run --rm --network host --runtime nvidia --device /dev/infiniband $COMMON --entrypoint python3 dsv41-tp4-sm121:overlay-v1 -c \"
import torch.distributed as dist
try:
    dist.init_process_group(backend='nccl', init_method='tcp://192.168.100.1:$PORT', world_size=2, rank=1, timeout=__import__('datetime').timedelta(seconds=50))
    t = __import__('torch').ones(1024, device='cuda')
    dist.all_reduce(t)
    print('NCCL-2RANK-OK sum=', t[0].item())
    dist.destroy_process_group()
except Exception as e:
    print('NCCL-2RANK-FAIL:', str(e)[:300])
\"" 2>&1 | tail -1
wait $HEAD_PID
grep -E "NCCL-2RANK" /tmp/nccl2-head.log | tail -1
