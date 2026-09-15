# Decode-lever A/B results (2026-09-15, uncommitted pending decision)

All runs: sglang.bench_serving, random dataset, 20 prompts, thinking OFF via
extra-request-body, streaming, warm drafter pass first, seed 42, prod image.

## Configurations compared
- PROD (TP=4, moe a2a none, no DP): original numbers from session analysis;
  raw logs lost to a script-dir rename mistake (documented below).
- MAXPERF (DP-attention dp=4, dp-lm-head, cont-decode-steps=2, vision zeroed):
  `bench-maxperf-final/` (identical copy in bench-maxperf-nosps/).
- MAXPERF+SPS (adds calibrated SPS table 2-cell + SGLANG_RAGGED_VERIFY_MODE=static):
  `bench-maxperf-sps/`.

## Output tok/s (c1 / c8)
- PROD:        short 6.62/15.21 · medium 6.83/12.14
- MAXPERF:     short 5.62/14.89 · medium 6.44/14.63
- MAXPERF+SPS: short 5.52/17.22 · medium 6.59/15.03  ← best c8 on both shapes

## Mean TPOT ms (c1 / c8)
- PROD:        short 174.9/384.4 · medium 116.3/439.7
- MAXPERF:     short 150.2/375.0 · medium 100.4/310.1
- MAXPERF+SPS: short 154.7/355.3 · medium 100.0/262.8  ← best c8 TPOT both shapes

## Median ITL ms (c8)
- MAXPERF:     short 180.4 · medium 129.6
- MAXPERF+SPS: short 163.8 · medium 128.5

## Mean TTFT ms (c8)
- PROD:        short ~6.1K · medium 24.9K
- MAXPERF+SPS: short 6.0K · medium 54.5K  (DP splits chunked prefill 4096→1024/group)

## Notes
- SPS gain at c8: +15% tok/s short, +3% medium; TPOT -5%/-15%. At c1 neutral
  (single stream = 1 req/group; ragged scheduler has no batch to trade).
- SPS table has only 2 cells (batch_tokens 6,12) because max_running_requests=8
  caps the sweep; wider cells need a higher-max profile run.
- DeepEP is architecturally unavailable on GB10 (IPC shared-memory handles fail
  under CUDA-graph capture; no NVLink). Dead end, verified twice.
- Predictable-text single-stream (counting probe): ~19 tok/s maxperf vs ~9.4
  prod — DP-attention doubles per-stream decode where the drafter can predict.

## v3/v4 amendments (2026-09-15 evening)
- static + SPS table = documented no-op (server warns); the "MAXPERF+SPS" rows above
  are effectively MAXPERF+instrumentation. Still the best measured config.
- compact mode: CRASHES with the engram file store (engram requires equal verify
  blocks per request; ragged fill violates the assert). Incompatible, not fixable
  without engine changes.
- cap-accept mode: boots and runs, but REGRESSES vs static with this 2-cell table
  (c8 short 12.4 vs 17.2 tok/s; medium 13.4 vs 15.0; TPOT +30-35%) — the table was
  profiled only at batch_tokens 6/12 (max_running=8 cap during profiling) and
  mis-caps at real batch sizes. Raw logs: bench-maxperf-v4-capaccept/.
- To make SPS actually pay: re-profile with max_running>=32 (wide table), then
  re-run cap-accept. Profiler tooling + plumbing all committed.
- FINAL RANKING (random-text decode): static-DP-attention maxperf > prod TP=4 >
  cap-accept-with-narrow-table. Predictable-text single-stream: maxperf ~2x prod.
