# GPU readiness gates for dsv4.1-flash TP=4 (2026-09-12, pre-boot)

Before any TP=4 boot, ALL FOUR GPUs must be free and stay free:

1. **N1 (r0b0t-dgx)** — qwen3.8-flash-next benchmark finalizing.
   Container `qwen38fn-99f9d1e08a3b0c5cc71a6ccd24996900`, tmux
   `qwen38fn-v101`. GATE: user confirms closure. Do not stop it ourselves.
2. **N2 (r0b0tdgx1)** — GPU idle, no containers. READY (once checkpoint
   replica is staged).
3. **N3 (gn100-2eea)** — GPU idle, no containers (GLM serve already gone).
   READY (once checkpoint replica staged).
4. **N4 (spark-4af5)** — GPU idle BUT an idle ComfyUI process
   (44.2 GiB RSS, `~/comfyui/comfyui-env/bin/python main.py ... --port 8188`)
   holds unified memory. GATE: user OK to stop it (it would starve rank 3).

Cluster disk: all nodes ≥2.3T free. Checkpoint staging target: 476 GiB per
node + 0 engram pack (file-backed design needs no pack).
