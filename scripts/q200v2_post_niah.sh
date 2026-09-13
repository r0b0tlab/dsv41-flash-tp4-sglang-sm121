#!/usr/bin/env bash
# q200v2_post_niah.sh — certified Q200v2 text-180 run after the 1M multineedle.
#
# One tmux owner on N1. Serial: waits for the NIAH result, then:
#   1. Identity tuple (epoch, nonces) computed head-side.
#   2. Full serve relaunch on overlay-v1-q200 with org.r0b0tlab.* labels
#      (serve.sh DSV41_GUARD_* plumbing, 1m profile).
#   3. Guards on ranks 0 (local N1) and 1 (ssh N2) via the certified
#      guard_unified_memory; admission config bound to both.
#   4. Certified run_quality_set.py (sandbox-graded humaneval, strict ifeval,
#      manual rubric carried, completeness-enforced).
set -euo pipefail

RUNNER=/home/r0b0tdgx/qwen38-flash-next-w4a16/q200v2ar-20260829T135441Z-runner
REPO=/home/r0b0tdgx/dsv4.1-flash
EV=$REPO/evidence/phase9/q200v2-proper
GUARD_DIR=$EV/guards
mkdir -p "$EV" "$GUARD_DIR/rank0" "$GUARD_DIR/rank1"
SSH_OPTS=(-i $HOME/.ssh/id_ed25519_crs812 -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric -o StrictHostKeyChecking=yes)
N1=192.168.68.59; N2=192.168.68.51

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$EV/run.log"; }

# --- 1. wait for NIAH ---------------------------------------------------------
log "waiting for NIAH multineedle result"
for i in $(seq 1 720); do
  [[ -f $REPO/evidence/phase9/niah-1m/twokey-33-66.json ]] && break
  sleep 60
done
[[ -f $REPO/evidence/phase9/niah-1m/twokey-33-66.json ]] || { log "FATAL niah never landed"; exit 3; }
log "NIAH landed: $(cat $REPO/evidence/phase9/niah-1m/twokey-33-66.json | head -c 200)"
sleep 180   # scheduler drain

# --- 2. identity tuple --------------------------------------------------------
EPOCH="dsv41q200v2-$(date -u +%Y%m%dT%H%M%SZ)"
CANDIDATE=dsv41-tp4-sm121-overlay-q200
SOURCE_SHA=$(git -C "$REPO" rev-parse HEAD)
PROFILE_SHA=$(sha256sum "$REPO/profiles/dsv41-1m.env" | cut -d' ' -f1)
IMAGE_ID=$(docker image inspect dsv41-tp4-sm121:overlay-v1-q200 --format '{{.Id}}')
NONCE() { python3 -c "import hashlib,sys; print(hashlib.sha256('\0'.join(sys.argv[1:]).encode()).hexdigest())" "$CANDIDATE" "$SOURCE_SHA" "$EPOCH" "$1" "$PROFILE_SHA" "$IMAGE_ID"; }
N0=$(NONCE 0); N1_=$(NONCE 1)
printf 'epoch=%s\ncandidate=%s\nsource_sha=%s\nprofile_sha=%s\nimage_id=%s\nnonce0=%s\nnonce1=%s\n' \
  "$EPOCH" "$CANDIDATE" "$SOURCE_SHA" "$PROFILE_SHA" "$IMAGE_ID" "$N0" "$N1_" > "$EV/identity.env"
log "identity: epoch=$EPOCH image=$IMAGE_ID"

# --- 3. ship the q200 image to N2/N3/N4 (rank containers need it) -------------
log "shipping overlay-v1-q200 image to workers"
for ip in 192.168.68.51 192.168.68.78 192.168.68.56; do
  ssh "${SSH_OPTS[@]}" r0b0tdgx@$ip 'docker image inspect dsv41-tp4-sm121:overlay-v1-q200 >/dev/null 2>&1' \
    || docker save dsv41-tp4-sm121:overlay-v1-q200 | ssh "${SSH_OPTS[@]}" r0b0tdgx@$ip 'docker load' >/dev/null
done

# --- 4. relaunch serve with labels ---------------------------------------------
log "tearing down old serve"
docker rm -f dsv41-rank >/dev/null 2>&1 || true
for ip in 192.168.68.51 192.168.68.78 192.168.68.56; do
  ssh "${SSH_OPTS[@]}" r0b0tdgx@$ip 'docker rm -f dsv41-rank >/dev/null 2>&1' || true
done
sleep 10

log "relaunching 1m serve with guard labels"
export DSV41_GUARD_EPOCH="$EPOCH" DSV41_GUARD_CANDIDATE="$CANDIDATE" \
       DSV41_GUARD_SOURCE_SHA="$SOURCE_SHA" DSV41_GUARD_PROFILE_SHA="$PROFILE_SHA" \
       DSV41_GUARD_IMAGE_ID="$IMAGE_ID"
# serve.sh hardcodes IMAGE=dsv41-tp4-sm121:overlay-v1; override via env
sed 's|^IMAGE=dsv41-tp4-sm121:overlay-v1$|IMAGE=dsv41-tp4-sm121:overlay-v1-q200|' \
  "$REPO/scripts/serve.sh" > "$REPO/scripts/.serve-q200.sh"
chmod +x "$REPO/scripts/.serve-q200.sh"
bash "$REPO/scripts/.serve-q200.sh" 1m 2>&1 | tee "$EV/serve-relaunch.log"
log "serve relaunched"

# --- 5. admission config --------------------------------------------------------
LEASE=$EV/lease-$EPOCH.lock
RANK0_CID=$(docker inspect dsv41-rank --format '{{.Id}}')
RANK1_CID=$(ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "docker inspect dsv41-rank --format '{{.Id}}'")
cat > "$EV/admission-config.json" <<EOF
{
  "schema": "r0b0tlab.qwen38.admission_config.v1",
  "epoch": "$EPOCH",
  "candidate_id": "$CANDIDATE",
  "candidate_source_sha": "$SOURCE_SHA",
  "profile_sha256": "$PROFILE_SHA",
  "image_id": "$IMAGE_ID",
  "lease_path": "$LEASE",
  "max_state_age_seconds": 5.0,
  "ack_timeout_seconds": 15.0,
  "poll_interval_seconds": 0.25,
  "ranks": [
    {
      "rank": "0",
      "host": null,
      "admission_state_path": "$GUARD_DIR/rank0/ADMISSION-STATE.json",
      "request_state_path": "$GUARD_DIR/rank0/request-state.json"
    },
    {
      "rank": "1",
      "host": "$N2",
      "ssh_identity_file": "$HOME/.ssh/id_ed25519_crs812",
      "admission_state_path": "$GUARD_DIR/rank1/ADMISSION-STATE.json",
      "request_state_path": "$GUARD_DIR/rank1/request-state.json"
    }
  ]
}
EOF
log "admission config written"

# --- 6. start guards -------------------------------------------------------------
log "starting rank0 guard (local)"
python3 "$RUNNER/scripts/guard_unified_memory.py" \
  --container dsv41-rank --candidate-id "$CANDIDATE" --candidate-source-sha "$SOURCE_SHA" \
  --container-id "$RANK0_CID" --image-id "$IMAGE_ID" --owner-nonce "$N0" \
  --peer-host "$N2" --peer-ssh-identity-file "$HOME/.ssh/id_ed25519_crs812" \
  --peer-container dsv41-rank --peer-candidate-id "$CANDIDATE" --peer-candidate-source-sha "$SOURCE_SHA" \
  --peer-container-id "$RANK1_CID" --peer-image-id "$IMAGE_ID" --peer-owner-nonce "$N1_" \
  --peer-rank 1 --rank 0 --epoch "$EPOCH" --profile-sha256 "$PROFILE_SHA" \
  --output "$GUARD_DIR/rank0" --request-state "$GUARD_DIR/rank0/request-state.json" \
  --ready-state "$GUARD_DIR/ready-$CANDIDATE-$EPOCH-rank0.json" \
  --admission-config "$EV/admission-config.json" \
  > "$GUARD_DIR/rank0/guard.log" 2>&1 &
GUARD0_PID=$!
log "rank0 guard pid $GUARD0_PID"

log "starting rank1 guard (on N2 via ssh)"
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "mkdir -p $GUARD_DIR/rank1" 2>/dev/null || true
scp -q -i $HOME/.ssh/id_ed25519_crs812 -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric \
    "$RUNNER/scripts/guard_unified_memory.py" "$RUNNER/scripts/admission_control.py" \
    "$RUNNER/scripts/niah_common.py" "r0b0tdgx@$N2:/tmp/" 
# rank1 guard needs its own admission paths on N2 (its local view)
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "python3 /tmp/guard_unified_memory.py \
  --container dsv41-rank --candidate-id '$CANDIDATE' --candidate-source-sha '$SOURCE_SHA' \
  --container-id '$RANK1_CID' --image-id '$IMAGE_ID' --owner-nonce '$N1_' \
  --peer-host '$N1' --peer-ssh-identity-file '$HOME/.ssh/id_ed25519_crs812' \
  --peer-container dsv41-rank --peer-candidate-id '$CANDIDATE' --peer-candidate-source-sha '$SOURCE_SHA' \
  --peer-container-id '$RANK0_CID' --peer-image-id '$IMAGE_ID' --peer-owner-nonce '$N0' \
  --peer-rank 0 --rank 1 --epoch '$EPOCH' --profile-sha256 '$PROFILE_SHA' \
  --output '$GUARD_DIR/rank1' --request-state '$GUARD_DIR/rank1/request-state.json' \
  --ready-state '$GUARD_DIR/ready-$CANDIDATE-$EPOCH-rank1.json' \
  --admission-config '$EV/admission-config.json' \
  > '$GUARD_DIR/rank1/guard.log' 2>&1 &"
log "guards started; waiting 30s for readiness"
sleep 30

# --- 7. certified runner ---------------------------------------------------------
log "running certified Q200v2 text-180"
cd "$RUNNER"
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
log "certified runner rc=$RC"
log "stopping guards"
kill $GUARD0_PID 2>/dev/null || true
ssh "${SSH_OPTS[@]}" r0b0tdgx@$N2 "pkill -f guard_unified_memory" 2>/dev/null || true
exit $RC
