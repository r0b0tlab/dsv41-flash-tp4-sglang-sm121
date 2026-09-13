# Engram no-collective proof — live profile (2026-09-13, prod profile boot 11)

Method: `POST /start_profile` (num_steps≈20, CPU+GPU) on the live TP=4 serve;
one streaming counting generation; trace from rank 0
(`/tmp/dsv41-prof/1789315087.131737-TP-0-EP-0.trace.json.gz`, 3.4 s span,
77,528 kernel events, ~21.4 decode steps estimated from the 1714 mHC-post
kernels = 80/step).

## Result: zero NCCL attributable to Engram

- `_engram_gather_kernel` fires 40× per trace (~2/step: layers 1+14).
- **0/119 engram kernels have any NCCL kernel within 50 µs** — no
  all-reduce/all-gather follows the gather; lookups resolve entirely from
  the node-local file-backed tables.
- Per-step NCCL totals: ~82 all-reduce + ~7 broadcast + ~5 all-gather.
  Upstream's own TP4 accounting is ~88 all-reduce/step for the model's
  residual/TP collectives — our count is AT the model baseline with none
  added by Engram. (A sharded-Engram ablation would show +2/step here;
  that arm is unnecessary to prove the negative: the gather kernel runs,
  and no collective is adjacent.)

## Interpretation

The file-backed shared-layout store (checkpoint shards 47/48 mmap'd
read-only, `engram_gather` reading through the page cache over ATS) is the
production Engram path on this serve. Tables: 384,006,168 + 384,016,682
rows, 264 B/row, 189.1 GiB total, zero dedicated RAM.

Trace artifacts: `~/dsv4.1-flash/evidence/phase7/trace-rank0.json.gz`.
