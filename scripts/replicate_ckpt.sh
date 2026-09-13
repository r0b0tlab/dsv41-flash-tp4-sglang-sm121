#!/usr/bin/env bash
# Replicate the verified checkpoint from N2 to N1/N3/N4 over the CRS812
# fabric (rsync in tmux, one durable owner per destination).
# Usage (from any node with the source tree): scripts/replicate_ckpt.sh
set -euo pipefail

SRC_HOST=r0b0tdgx@192.168.100.2
SRC_PATH='~/models/llm/dsv41/DeepSeek-V4.1-Flash/'
SSH_KEY=$HOME/.ssh/id_ed25519_crs812
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes
          -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric)

for dst in 192.168.100.1 192.168.100.3 192.168.100.4; do
  echo "starting replication -> $dst"
  ssh "${SSH_OPTS[@]}" "$SRC_HOST" bash -s <<REMOTE
set -euo pipefail
mkdir -p ~/models/llm/dsv41
tmux new-session -d -s rsync-$dst "rsync -a --info=progress2 \
  -e 'ssh -i $SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812_fabric' \
  $SRC_PATH r0b0tdgx@$dst:~/models/llm/dsv41/DeepSeek-V4.1-Flash/ \
  > ~/rsync-$dst.log 2>&1; echo RC=\\\$? >> ~/rsync-$dst.log"
REMOTE
done
echo "all rsync owners launched; verify each with:"
echo "  ssh <dst> 'tail -1 ~/rsync-<dst-ip>.log; du -sb ~/models/llm/dsv41/DeepSeek-V4.1-Flash'"
