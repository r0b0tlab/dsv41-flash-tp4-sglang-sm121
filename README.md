# dsv4.1-flash — DeepSeek-V4.1-Flash TP=4 SGLang on 4×GB10 (SM121)

Publication package: SGLang TP=4/EP=4 serving of `deepseek-ai/DeepSeek-V4.1-Flash`
(fp8 checkpoint, unmodified) on the 4-node CRS812 DGX Spark (GB10/SM121)
cluster, with a from-scratch **Engram** local NVMe store that removes the
per-lookup TP all-reduce from the DSpark speculative path.

## Headline results (phase 9, final fingerprint, serial lanes)

| Lane | Result | Notes |
|---|---|---|
| Serve envelope | `mem_fraction_static=0.80`, `max_total_tokens=1 099 776`, KV fp8_e4m3 | capacity unchanged vs pre-fix |
| Concurrency ladder c1→c8 | 53→160 tok/s aggregate; peak running-req 1→8 proved from rank decode logs; 0 errors | counting-100 prompt, 300 max_tokens, thinking off, temp 0, 60 s/step, tokens from server `usage`, warm serve (accept-len ≥ 5.5) |
| Warm throughput lanes | short_c1 16.9, medium_c1 13.5, prose_c1 10.5, counting_c1 38.6, counting_c4 112.3 tok/s | all: temp 0, tokens from server `usage`, warm serve. Per lane: short = 400 random 5-digit ids in / 256 out (random ids → DSpark accept-len ≈ 1.7, speculation near-useless); medium = ~2 K-token passage / 512 out; prose = 800 out free-form story (accept ≈ 1.1); counting = count-to-N lists (accept ≈ 4→5.8). c1 = 1 stream, c4 = 4 streams. **Not comparable across prompt classes** — effective decode ≈ accept-len × step rate (~9–10 steps/s at bs1) |
| Vision canary (cvbench Count) | 38/60 = 63.3% (Wilson95 50.7–74.4) | r0b0bench-vision v1.0 contract `2b80e543…`, 1 worker, serial, base64 jpeg, max_tokens 32, thinking off |
| Q200v2 text-180 (certified runner) | **gsm8k 96.25% (77/80) · humaneval 100% (40/40, sandbox-graded) · ifeval 97.5% (39/40) · hard_reasoning 100% (20/20, independent manual review) — 176/180 = 97.8%** | native thinking (effort low), temp from server default, 1 worker, admission-gated two-rank memory guards, run identity `4138fb13…`; all 180 rows finish=stop; evidence `evidence/phase9/q200v2-proper/` |
| NIAH 512k ladder | **deferred** | 1M-profile multineedle runs post-publication (ongoing, see below) |

Cold-start note: first minutes after serve boot, DSpark accept length climbs
from ~1.6 to ≥5.8 as the drafter warms; throughput lanes must be measured
after warm-up (phase-8 protocol) or numbers understate by ~4×.

## Ongoing (post-publication)

- **1M-context multineedle (two-key 33/66)** — **documented limit after 3 attempts** (2026-09-13/14):
  the KV fill during a 1M-token prefill exhausts host-side unified memory at
  ~50% progress (~517K tokens) on every configuration tried: the stock 1m
  profile, a corrected lane profile (max_total_tokens 1 049 088,
  max_running_requests 1), and the same lane headless (gdm stopped). The wall
  is the model's KV allocation profile, not host software. Each attempt was
  aborted by the node-safety watch at the NVRM `NV_ERR_NO_MEMORY` precursor
  with zero node losses. Full receipts:
  `evidence/phase9/niah-1m/twokey-33-66{,-v2,-v3}.json`. 512K-window
  operation is fully supported (see lanes above).
- ~~Certified Q200v2~~ **COMPLETE 2026-09-14** — see headline table.

## Reproduce

```bash
docker pull ghcr.io/r0b0tlab/dsv41-flash-tp4-sglang-sm121:overlay-v1   # anonymous pull
scripts/serve.sh prod        # launches ranks on all 4 nodes (workers first)
scripts/bench_orchestrator.sh prod   # serial phase-9 lanes with memory gates
```

Requirements per node: GB10/SM121, Docker + NVIDIA runtime, RDMA fabric
(CRS812), checkpoint at `~/models/llm/dsv41/DeepSeek-V4.1-Flash`, ≥24 GiB
MemAvailable at launch (enforced by `scripts/node_prepare.sh`).

See `docs/POSTMORTEM-2026-09-13.md` for the N1/N3 freeze root cause and the
node-preparation invariants enforced by this package (admission floor +
`--weight-loader-drop-cache-after-load` + serial lanes + no leftover
cluster services on serving nodes).
