#!/usr/bin/env bash
# q200v2_resume.sh — resume the certified Q200v2 chain after the serve.sh
# label-bug fix: identity tuple REUSED from the first attempt (guards/labels
# must bind the same epoch+image), serve relaunched, guards started, runner
# executed. Idempotent-ish: kills any stale guards first.
set -euo pipefail

RUNNER=/home/r0b0tdgx/qwen38-flash-next-w4a16/q200v2ar-20260829T135441Z-runner
REPO=/home/r0b0tdgx/dsv4.1-flash
EV=$REPO/evidence/phase9/q200v2-proper
GUARD_DIR=$EV/guards
SSH_OPTS=(-i $HOME/.ssh/id_ed25519_crs812 -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric -o StrictHostKeyChecking=yes)
N2=192.168.68.51; N1=192.168.68.59

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$EV/run.log"; }

# identity from the first attempt (same epoch/image as the shipped labels)
source <(grep -E '^(epoch|candidate|source_sha|profile_sha|image_id|nonce0|nonce1)=' "$EV/identity.env" | sed 's/^candidate=/CANDIDATE=/; s/^source_sha=/SOURCE_SHA=/; s/^profile_sha=/PROFILE_SHA=/; s/^image_id=/IMAGE_ID=/; s/^nonce0=/NONCE0=/; s/^nonce1=/NONCE1=/')
EPOCH=$epoch
log "resuming chain: epoch=$EPOCH image=$IMAGE_ID"

# 1. relaunch serve with labels (fixed script)
log "relaunching 1m serve with guard labels"
export DSV41_GUARD_EPOCH="$EPOCH" DSV41_GUARD_CANDIDATE="$CANDIDATE" \
       DSV41_GUARD_SOURCE_SHA="$SOURCE_SHA" DSV41_GUARD_PROFILE_SHA="$PROFILE_SHA" \
       DSV41_GUARD_IMAGE_ID="$IMAGE_ID"
bash "$REPO/scripts/.serve-q200.sh" 1m 2>&1 | tee "$EV/serve-relaunch.log"
log "serve up"

# 2. container ids (post-launch truth)
RANK0_CID=$(docker inspect dsv41-rank --format '{{.Id}}')
RANK1_CID=$(ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "docker inspect dsv41-rank --format '{{.Id}}'")
log "rank0=$RANK0_CID rank1=$RANK1_CID"

# 3. admission config (paths unchanged)
log "admission config"
cat > "$EV/admission-config.json" <<EOF
{
  "schema": "r0b0tlab.qwen38.admission_config.v1",
  "epoch": "$EPOCH",
  "candidate_id": "$CANDIDATE",
  "candidate_source_sha": "$SOURCE_SHA",
  "profile_sha256": "$PROFILE_SHA",
  "image_id": "$IMAGE_ID",
  "lease_path": "$EV/lease-$EPOCH.lock",
  "max_state_age_seconds": 5.0,
  "ack_timeout_seconds": 15.0,
  "poll_interval_seconds": 0.25,
  "ranks": [
    {"rank": "0", "host": null,
     "admission_state_path": "$GUARD_DIR/rank0/ADMISSION-STATE.json",
     "request_state_path": "$GUARD_DIR/rank0/request-state.json"},
    {"rank": "1", "host": "$N2",
     "ssh_identity_file": "$HOME/.ssh/id_ed25519_crs812",
     "admission_state_path": "$GUARD_DIR/rank1/ADMISSION-STATE.json",
     "request_state_path": "$GUARD_DIR/rank1/request-state.json"}
  ]
}
EOF

# 4. guards
pkill -f guard_unified_memory 2>/dev/null || true
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "pkill -f guard_unified_memory" 2>/dev/null || true
rm -f "$GUARD_DIR/rank0/request-state.json" "$GUARD_DIR/rank1/request-state.json"

log "starting rank0 guard (local)"
nohup python3 "$RUNNER/scripts/guard_unified_memory.py" \
  --container dsv41-rank --candidate-id "$CANDIDATE" --candidate-source-sha "$SOURCE_SHA" \
  --container-id "$RANK0_CID" --image-id "$IMAGE_ID" --owner-nonce "$NONCE0" \
  --peer-host "$N2" --peer-ssh-identity-file "$HOME/.ssh/id_ed25519_crs812" \
  --peer-container dsv41-rank --peer-candidate-id "$CANDIDATE" --peer-candidate-source-sha "$SOURCE_SHA" \
  --peer-container-id "$RANK1_CID" --peer-image-id "$IMAGE_ID" --peer-owner-nonce "$NONCE1" \
  --peer-rank 1 --rank 0 --epoch "$EPOCH" --profile-sha256 "$PROFILE_SHA" \
  --output "$GUARD_DIR/rank0" --request-state "$GUARD_DIR/rank0/request-state.json" \
  --ready-state "$GUARD_DIR/ready-$CANDIDATE-$EPOCH-rank0.json" \
  --admission-config "$EV/admission-config.json" \
  > "$GUARD_DIR/rank0/guard.log" 2>&1 &
log "rank0 guard pid $!"

scp -q -i $HOME/.ssh/id_ed25519_crs812 -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric \
    "$RUNNER/scripts/guard_unified_memory.py" "$RUNNER/scripts/admission_control.py" \
    "$RUNNER/scripts/niah_common.py" "r0b0tdgx@$N2:/tmp/"
log "starting rank1 guard (on N2)"
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "mkdir -p '$GUARD_DIR/rank1'; nohup python3 /tmp/guard_unified_memory.py \
  --container dsv41-rank --candidate-id '$CANDIDATE' --candidate-source-sha '$SOURCE_SHA' \
  --container-id '$RANK1_CID' --image-id '$IMAGE_ID' --owner-nonce '$NONCE1' \
  --peer-host '$N1' --peer-ssh-identity-file '$HOME/.ssh/id_ed25519_crs812' \
  --peer-container dsv41-rank --peer-candidate-id '$CANDIDATE' --peer-candidate-source-sha '$SOURCE_SHA' \
  --peer-container-id '$RANK0_CID' --peer-image-id '$IMAGE_ID' --peer-owner-nonce '$NONCE0' \
  --peer-rank 0 --rank 1 --epoch '$EPOCH' --profile-sha256 '$PROFILE_SHA' \
  --output '$GUARD_DIR/rank1' --request-state '$GUARD_DIR/rank1/request-state.json' \
  --ready-state '$GUARD_DIR/ready-$CANDIDATE-$EPOCH-rank1.json' \
  --admission-config '$EV/admission-config.json' \
  > '$GUARD_DIR/rank1/guard.log' 2>&1 &"
sleep 45
log "guards up; readiness files:"; ls "$GUARD_DIR" | head -6

# 5. certified runner
log "running certified Q200v2 text-180"
cd "$RUNNER"
set +e
python3 scripts/run_quality_set.py \
  --base-url http://127.0.0.1:30000 \
  --run-id "dsv41-q200v2-$EPOCH" \
  --model /model \
  --image-id "$IMAGE_ID" \
  --profile-id "dsv41-1m-q200" \
  --candidate-id "$CANDIDATE" \
  --admission-config "$EV/admission-config.json" \
  --chat-template-kwargs '{"enable_thinking": true, "thinking": true, "reasoning_effort": "low"}' \
  --workers 1 --max-tokens 8192 \
  2>&1 | tee "$EV/quality-run.log"
RC=$?
set -e
log "certified runner rc=$RC"
log "stopping guards"
pkill -f guard_unified_memory 2>/dev/null || true
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "pkill -f guard_unified_memory" 2>/dev/null || true
exit $RC
