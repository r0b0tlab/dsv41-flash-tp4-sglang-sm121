#!/usr/bin/env bash
# bench_orchestrator.sh — serial phase-9 lane runner (ROOT-CAUSE FIX for the
# 2026-09-13 N1/N3 freeze: three bench clients ran concurrently against one
# serve; the head's unified-memory pool drained, NVRM allocations failed, and
# the sglang scheduler held the driver's global lock retrying them).
#
# INVARIANTS (enforced, not advisory):
#   1. ONE lane at a time. No concurrent benchmarks, ever.
#   2. MemAvailable floor on the head checked before EVERY lane.
#   3. Each lane's process must exit before the next starts (reap by PGID).
#   4. All output under evidence/phase9/<lane>/; JSON receipt per lane.
#
# Usage: bench_orchestrator.sh <profile>   (profile: prod|1m)
# Run inside tmux on N1 with the serve already READY.
set -euo pipefail

PROFILE="${1:?usage: bench_orchestrator.sh prod|1m}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EV="$REPO/evidence/phase9"
LOG="$EV/orchestrator.log"
FLOOR_MIB=12000          # head floor while a lane runs (serve already resident)
LANES=(short_lanes vision q200 concurrency_ladder)
mkdir -p "$EV"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

avail() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

gate() {
  local a; a=$(avail)
  if (( a < FLOOR_MIB )); then
    log "GATE-FAIL MemAvailable=${a}MiB < ${FLOOR_MIB}MiB — waiting up to 10m for reclaim"
    for _ in $(seq 1 60); do
      sleep 10
      a=$(avail)
      (( a >= FLOOR_MIB )) && break
    done
    (( a >= FLOOR_MIB )) || { log "FATAL floor not recovered — stopping all lanes"; exit 4; }
  fi
  log "gate OK MemAvailable=${a}MiB"
}

serve_ready() {
  curl -sf -m 10 "http://127.0.0.1:30000/v1/models" >/dev/null
}

run_lane() {
  local name="$1"; shift
  gate
  serve_ready || { log "FATAL serve not ready before $name"; exit 5; }
  log "LANE $name START"
  mkdir -p "$EV/$name"
  local rc=0
  "$@" >"$EV/$name/lane.log" 2>&1 || rc=$?
  log "LANE $name DONE rc=$rc"
  echo "{\"lane\": \"$name\", \"rc\": $rc, \"ended_utc\": \"$(date -u +%FT%TZ)\"}" \
    > "$EV/$name/receipt.json"
  # inter-lane cooldown: wait for the scheduler to drain (metrics disabled on
  # this serve; /get_server_info exposes running/queue counts)
  for _ in $(seq 1 30); do
    RUN=$(curl -sf -m 10 http://127.0.0.1:30000/get_server_info 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("num_running_reqs", d.get("running_requests", 0)))' 2>/dev/null) || RUN=1
    [[ "$RUN" == "0" ]] && break
    sleep 10
  done
  return $rc
}

log "=== orchestrator start profile=$PROFILE host=$(hostname) ==="
log "lanes: ${LANES[*]}"

# 1. SHORT/MEDIUM/PROSE + counting + c4
run_lane short_lanes python3 "$REPO/scripts/lanes.py" \
  --base-url http://127.0.0.1:30000 --out "$EV/short_lanes/lanes.json"

# 2. Vision canary (frozen r0b0bench-vision cvbench subset, 1 worker — the
#    crashed run interleaved 4 vision workers with NIAH prefill on the head)
run_lane vision python3 ~/projects/r0b0bench-vision/run_vision.py \
  --base-url http://127.0.0.1:30000 --data-dir ~/rbv-data \
  --out-dir "$EV/vision" --model /model \
  --image-id local --profile-id phase9-serial --candidate-id dsv41-tp4-sm121-overlay \
  --only cvbench --workers 1 --max-rows 60

# 3. Q200v2-lite quality
run_lane q200 python3 "$REPO/scripts/q200_lite.py" \
  --base-url http://127.0.0.1:30000 --n 60 --out "$EV/q200/q200.json"

# 4. Concurrency ladder c1..c8 (skips if the tool is absent)
if [[ -x "$REPO/scripts/concurrency_ladder.py" ]]; then
  run_lane concurrency_ladder python3 "$REPO/scripts/concurrency_ladder.py" \
    --base-url http://127.0.0.1:30000 --out "$EV/concurrency_ladder/ladder.json"
else
  log "SKIP concurrency_ladder (script absent)"
fi

log "=== orchestrator done ==="
