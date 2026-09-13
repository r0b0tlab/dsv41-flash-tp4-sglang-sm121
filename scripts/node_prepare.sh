#!/usr/bin/env bash
# node_prepare.sh — per-rank host preparation before a dsv4.1-flash serve.
#
# Root cause of the 2026-09-13 N1/N3 freeze (see evidence/phase9 and the
# postmortem in docs/POSTMORTEM-2026-09-13.md): the sglang rank reserved
# 0.80 of the unified GB10 pool while ~27 GiB of stale weight-shard page
# cache plus non-essential residents (k3s on N3, stacked bench clients on
# N1) pushed physical DRAM past the point where NVRM could satisfy
# allocations. The sglang scheduler then retried allocations while holding
# the driver's global rmapi rw-semaphore, D- Blocking every GPU ioctl on
# the node (nvidia-smi 1105s+, health checks, runc) — the observed "freeze".
#
# This script makes a node "prepared like N2/N4 were": verify checkpoint
# files, require a MemAvailable floor, drop only page cache (never the
# running serve), and record the pre-launch state for the evidence trail.
#
# Usage: node_prepare.sh <rank> [mem_floor_mib]     (run ON the target node)
set -euo pipefail

RANK="${1:?usage: node_prepare.sh <rank> [mem_floor_mib]}"
FLOOR="${2:-24000}"
MODEL_DIR="$HOME/models/llm/dsv41/DeepSeek-V4.1-Flash"

echo "[prep] node $(hostname) rank=$RANK floor=${FLOOR}MiB"

# 1. Checkpoint index + tail shards present (fail closed, matches serve.sh).
for f in model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -f "$MODEL_DIR/$f" ]] || { echo "[prep] FAIL: missing $MODEL_DIR/$f"; exit 3; }
done
echo "[prep] checkpoint index + tail shards OK"

# 2. Unified-memory admission floor. On GB10 the GPU reservation competes
#    with host DRAM for the same physical pages; MemAvailable must be at or
#    above the floor before launch or NVRM allocation failures hang the node.
AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
echo "[prep] MemAvailable=${AVAIL}MiB"
if (( AVAIL < FLOOR )); then
  echo "[prep] FAIL: MemAvailable ${AVAIL}MiB < floor ${FLOOR}MiB — free memory before launching (do NOT raise mem_fraction_static)"
  exit 4
fi

# 3. Drop page cache only (dirty pages stay; nothing is fsync'd here).
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' \
  || echo "[prep] WARN: could not drop caches (no sudo) — proceeding, floor already verified"

AVAIL2=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
echo "[prep] post-drop MemAvailable=${AVAIL2}MiB"
if (( AVAIL2 < FLOOR )); then
  echo "[prep] FAIL: MemAvailable ${AVAIL2}MiB < floor ${FLOOR}MiB after cache drop"
  exit 5
fi

# 4. Record pre-launch state (evidence trail).
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$HOME/dsv4.1-flash/logs"
{
  echo "ts=$TS rank=$RANK host=$(hostname)"
  grep -E 'MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree' /proc/meminfo
} >> "$HOME/dsv4.1-flash/logs/node-prepare.log"
echo "[prep] OK rank=$RANK MemAvailable=${AVAIL2}MiB ts=$TS"
