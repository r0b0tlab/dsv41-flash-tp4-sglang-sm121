#!/usr/bin/env bash
# dsv4.1-flash TP=4 launcher (fail-closed). Run from N1 (head, rank 0).
# Usage: scripts/serve.sh prod|1m
#
# Ranks: N1=0 (head), N2=1, N3=2, N4=3. Workers first, then rank 0.
set -euo pipefail

PROFILE="${1:?usage: serve.sh prod|1m}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/profiles/dsv41-${PROFILE}.env" || { echo "unknown profile $PROFILE"; exit 1; }

# --- cluster (CRS812 fabric addresses; SSH via crs812 key) -------------
declare -A MGMT FAB HOST
MGMT[0]=192.168.68.59;  FAB[0]=192.168.100.1; HOST[0]=r0b0t-dgx
MGMT[1]=192.168.68.51;  FAB[1]=192.168.100.2; HOST[1]=r0b0tdgx1
MGMT[2]=192.168.68.78;  FAB[2]=192.168.100.3; HOST[2]=gn100-2eea
MGMT[3]=192.168.68.56;  FAB[3]=192.168.100.4; HOST[3]=spark-4af5
SSH_KEY=$HOME/.ssh/id_ed25519_crs812
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric)

IMAGE=dsv41-tp4-sm121:overlay-v1
DIST_PORT=20000
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

# --- NCCL application env (never global) ------------------------------
# Ship the qualified fabric env into every container, with two per-node
# corrections applied INSIDE each rank's script (they are node-local facts):
#   1. GLOO_SOCKET_IFNAME must be the fabric-side NIC: with only
#      NCCL_SOCKET_IFNAME set, Gloo resolves through the default route and
#      cross-node pairs hit 127.0.0.1 -> "Connection refused".
#   2. NCCL_SOCKET_IFNAME must be the rank's own fabric interface, not the
#      management NIC the shared file names.
NCCL_ENV_FILE=$HOME/projects/crs812-cluster/hosts/crs812-nccl.env
NCCL_ENV=""
if [[ -f "$NCCL_ENV_FILE" ]]; then
  NCCL_ENV=$(grep -E '^export NCCL_' "$NCCL_ENV_FILE" \
             | grep -v 'NCCL_SOCKET_IFNAME' \
             | sed -e 's/^export /-e /' -e "s/'//g" | tr '\n' ' ')
fi

remote_env() { # rank -> env string consumed inside docker run
  cat <<EOF
-e DSV41_ENGRAM_FILE_STORE=$DSV41_ENGRAM_FILE_STORE \
-e DSV41_MODEL_PATH=/model \
-e PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
-e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=$SGLANG_FLASHINFER_MOE_FUSED_FINALIZE \
-e PORT=$PORT
EOF
}

launch_rank() {
  local r=$1
  echo "[$(date -u +%H:%M:%S)] launching rank $r on ${HOST[$r]}"
  ssh "${SSH_OPTS[@]}" "r0b0tdgx@${MGMT[$r]}" bash -s <<REMOTE
set -euo pipefail
# preflight: checkpoint index + engram shards present
for f in model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -f \$HOME/models/llm/dsv41/DeepSeek-V4.1-Flash/\$f ]] || { echo "rank $r preflight: missing \$f"; exit 3; }
done
# idle-gated: refuse if another container is already up
[[ -z \$(docker ps -q) ]] || { echo "rank $r preflight: containers running: \$(docker ps --format '{{.Names}}')"; exit 4; }
# Resolve THIS node's fabric interface (holds 192.168.100.<r+1>) and pin
# both Gloo and NCCL's TCP bootstrap to it.
FAB_IF=\$(ip -o addr | awk -v ip="${FAB[$r]}/" 'index(\$4, ip)==1{print \$2; exit}')
[[ -n "\$FAB_IF" ]] || { echo "rank $r: no iface holds ${FAB[$r]}"; exit 2; }
echo "rank $r fabric iface: \$FAB_IF"
docker run -d --name dsv41-rank --network host --ipc host \
  --runtime nvidia --device /dev/infiniband \
  --cap-add CAP_IPC_LOCK --ulimit memlock=-1 \
  -e HOSTNAME=${HOST[$r]} \
  -v \$HOME/models/llm/dsv41/DeepSeek-V4.1-Flash:/model:ro \\
  $(remote_env $r) ${NCCL_ENV} \\
  -e GLOO_SOCKET_IFNAME=\$FAB_IF -e NCCL_SOCKET_IFNAME=\$FAB_IF \\
  -e SGLANG_SM120_FLASHMLA_BACKEND=${SGLANG_SM120_FLASHMLA_BACKEND:-flashinfer} \
  $IMAGE \
  python3 -m sglang.launch_server \
    --model-path /model \
    --trust-remote-code \
    --tp-size $TP_SIZE --ep-size $EP_SIZE \
    --nnodes $NNODES --node-rank $r \
    --dist-init-addr ${FAB[0]}:$DIST_PORT \
    --host 0.0.0.0 --port $PORT \
    --context-length $CONTEXT_LENGTH \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size $DSPARK_BLOCK_SIZE \
    --reasoning-parser auto --tool-call-parser auto \
    $EXTRA_SGLANG_ARGS \
  > $HOME/dsv4.1-flash-rank.log 2>&1 || { docker rm -f dsv41-rank >/dev/null 2>&1 || true; exit 5; }
echo "rank $r container up: \$(docker ps --filter name=dsv41-rank --format '{{.ID}} {{.Status}}')"
REMOTE
}

# workers first (1..3), then head (0)
for r in 1 2 3; do launch_rank $r; sleep 2; done
launch_rank 0

echo "all ranks launched; polling /v1/models on head ..."
for i in $(seq 1 200); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" | grep -q DeepSeek; then
    echo "READY after ${i}x15s"; exit 0
  fi
  sleep 15
done
echo "TIMEOUT waiting for readiness; check logs/ per-rank"; exit 6
