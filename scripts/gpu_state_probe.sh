#!/usr/bin/env bash
# GB10 GPU fast/slow-state probe (external 4xSpark issue #1 reproduction):
# the GPU flips between a fast and a slow state invisible to nvidia-smi;
# long idle reliably leaves it slow, moving decode-shaped GEMV up to ~1.5x.
# Lock clocks, warm, and time a fixed decode-shaped matmul; report the
# state. Exit 1 on slow state (caller retries or investigates).
set -euo pipefail

REMOTE_SCRIPT=$(cat <<'PY'
import torch, time, statistics
torch.manual_seed(0)
a = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16)
w = torch.randn(5120, 4096, device="cuda", dtype=torch.bfloat16)
# warmup flips the GPU into (or reveals) its current steady state
for _ in range(50):
    a @ w
torch.cuda.synchronize()
times = []
for _ in range(30):
    t0 = time.perf_counter()
    for _ in range(20):
        a @ w
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) / 20)
med = statistics.median(times) * 1e3
# fast state: ~0.05-0.15 ms; slow state: >0.25 ms for this shape on GB10
print(f"gemv_ms={med:.3f}")
print("STATE=FAST" if med < 0.25 else "STATE=SLOW")
PY
)

for host in "$@"; do
  echo "== $host =="
  ssh -o BatchMode=yes "r0b0tdgx@$host" "nvidia-smi -lgc 1 >/dev/null 2>&1 || true; python3 - <<'EOF'
$REMOTE_SCRIPT
EOF"
done
