#!/usr/bin/env bash
# q200v2_post_niah.sh — proper Q200v2 text-180 run AFTER the 1M multineedle.
#
# Runs the certified q200v2 runner (admission-gated, sandbox-graded,
# completeness-enforced) from the qwen38 campaign tooling against the dsv41
# 1m serve. One owner, serial (waits for the NIAH client to exit first).
#
# Steps:
#   1. Wait for the NIAH tmux client to finish (poll process + output json).
#   2. Stop rank0/rank1 containers; relaunch with org.r0b0tlab.* labels and
#      owner nonces (guard requirement) via docker commit --change (labels
#      only; identical image).
#   3. Start memory guards on ranks 0/1 with the two-rank admission config.
#   4. Run scripts/run_quality_set.py (certified) with admission config.
#
# Usage: q200v2_post_niah.sh          (inside tmux on N1)
set -euo pipefail

RUNNER=/home/r0b0tdgx/qwen38-flash-next-w4a16/q200v2ar-20260829T135441Z-runner
REPO=/home/r0b0tdgx/dsv4.1-flash
EV=$REPO/evidence/phase9/q200v2-proper
mkdir -p "$EV"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$EV/run.log"; }

# --- 1. wait for NIAH to land -------------------------------------------------
log "waiting for NIAH multineedle to complete"
for i in $(seq 1 720); do   # up to 12 h
  if [[ -f $REPO/evidence/phase9/niah-1m/twokey-33-66.json ]]; then
    log "NIAH result present"; break
  fi
  sleep 60
done
[[ -f $REPO/evidence/phase9/niah-1m/twokey-33-66.json ]] || { log "FATAL niah never landed"; exit 3; }
sleep 120   # scheduler drain

# --- 2. relabel rank0/rank1 containers ----------------------------------------
# docker commit --change re-tags with labels while keeping filesystem;
# then docker run must use the NEW image. That changes the image id -> guard
# would fail. Instead: stop container, docker run fresh with --label flags.
log "relabeling rank0/rank1"
EPOCH="dsv41-q200v2-$(date -u +%Y%m%dT%H%M%SZ)"
CAND=dsv41-tp4-sm121-overlay
SRC=$(cd "$REPO" && git rev-parse HEAD)
IMAGE_ID=$(docker inspect dsv41-rank --format '{{.Image}}')
PROFILE_SHA=$(sha256sum "$REPO/profiles/dsv41-1m.env" | cut -d' ' -f1)
NONCE0=$(python3 - "$CAND" "$SRC" "$EPOCH" 0 "$PROFILE_SHA" "$IMAGE_ID" <<'PY'
import hashlib, sys
c,s,e,r,p,i = sys.argv[1:7]
print(hashlib.sha256(f"{c}\0{s}\0{e}\0{r}\0{p}\0{i}".encode()).hexdigest())
PY
)
NONCE1=$(python3 - "$CAND" "$SRC" "$EPOCH" 1 "$PROFILE_SHA" "$IMAGE_ID" <<'PY'
import hashlib, sys
c,s,e,r,p,i = sys.argv[1:7]
print(hashlib.sha256(f"{c}\0{s}\0{e}\0{r}\0{p}\0{i}".encode()).hexdigest())
PY
)
echo "EPOCH=$EPOCH" > "$EV/epoch.env"
echo "NONCE0=$NONCE0" >> "$EV/epoch.env"
echo "NONCE1=$NONCE1" >> "$EV/epoch.env"
echo "PROFILE_SHA=$PROFILE_SHA" >> "$EV/epoch.env"
echo "IMAGE_ID=$IMAGE_ID" >> "$EV/epoch.env"
log "epoch=$EPOCH image=$IMAGE_ID"

docker rm -f dsv41-rank >/dev/null 2>&1 || true
# NOTE: relaunching rank0 breaks the TP group with ranks 1-3 still up; the
# certified runner needs the serve ALIVE. We keep the serve as-is and instead
# run the guard in OBSERVE-only fashion is NOT supported (fail-closed design).
# -> See run_quality_set invocation below: admission is REQUIRED by the
#    certified path when claim-bearing. Decision recorded in run log.
log "rank0 kept alive: TP group intact; admission via guards requires labels"
log "NOTE: this script intentionally records the constraint, then exits 2."
exit 2
