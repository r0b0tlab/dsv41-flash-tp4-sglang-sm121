# dsv4.1-flash — DeepSeek-V4.1-Flash TP=4 SGLang on 4×GB10 (SM121)

Publication package: SGLang TP=4/EP=4 serving of `deepseek-ai/DeepSeek-V4.1-Flash`
(fp8 checkpoint, unmodified) on the 4-node CRS812 DGX Spark (GB10/SM121)
cluster, with a from-scratch **Engram** local NVMe store that removes the
per-lookup TP all-reduce from the DSpark speculative path.

## Headline results (phase 9, final fingerprint, serial lanes)

| Lane | Result | Notes |
|---|---|---|
| Serve envelope | `mem_fraction_static=0.80`, `max_total_tokens=1 099 776`, KV fp8_e4m3 | capacity unchanged vs pre-fix |
| Concurrency ladder c1→c8 | 53→160 tok/s aggregate; peak running-req 1→8 proved from rank decode logs; 0 errors | serial lane, scheduler-sampled |
| Warm throughput lanes | short_c1 16.9, medium_c1 13.5, prose_c1 10.5, counting_c1 38.6, counting_c4 112.3 tok/s | DSpark accept-len 5.8+ at steady state; lanes.py own harness |
| Vision canary (cvbench Count) | 38/60 = 63.3% (Wilson95 50.7–74.4) | r0b0bench-vision v1.0 contract `2b80e543…`, 1 worker, serial |
| Q200v2 text-180 | gsm8k 15/15; humaneval/hard_reasoning/ifeval **not scored** by the lite harness (needs exec sandbox + ifeval strict + manual rubric — see `evidence/phase9/q200/lane.log`) | full Q200v2 requires the r0b0bench qwen38 sandbox driver; scored separately post-publication |
| NIAH 512k ladder | **deferred** | 1M-profile multineedle runs post-publication (ongoing, see below) |

Cold-start note: first minutes after serve boot, DSpark accept length climbs
from ~1.6 to ≥5.8 as the drafter warms; throughput lanes must be measured
after warm-up (phase-8 protocol) or numbers understate by ~4×.

## Ongoing (post-publication)

- **1M-context multineedle (two-key 33/66)** on the long profile
  (`profiles/dsv41-1m.env`, 1 048 576 ctx) — launched after package
  publication; result appended to this table and `evidence/phase9/niah-1m/`
  when complete.

## Reproduce

```bash
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
