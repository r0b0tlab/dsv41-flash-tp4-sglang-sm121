# Bounded prefill work on GB10

The pinned SGLang V4 low-ratio prefill indexer materializes query-by-prefix scores and candidate masks. This is distinct from the persistent KV pool. The old 1M aborts cannot establish a hardware-wide KV limit: an approximately 14-byte/query/key transient coefficient reported by independent GB10 testing is consistent with their scale, but has not been causally measured on this deployment.

The closeout adapter uses a conservative integer budget: T * (history + T) <= 300,000,000 token-squared. Page size is 256; continuation chunks range from 256 to 2048. At history 0/100000/200000/524288/1046462 the chosen sizes are 2048/2048/1280/512/256. Impossible minimum chunks fail closed. Static first chunks are 512 for 512K and 256 for 1M, including cache-hit first chunks. Final profiles serialize prefill requests so a second long cached prefix does not invalidate the chosen bound.

The candidate-copy budget is 256 MiB, not an assertion that all transient allocations fit in 256 MiB. CUDA graph replay, model semantics, telemetry and full-window retrieval still require live qualification. MADV_RANDOM is a locality hint, not a measured speedup. Mapped Engram pages consume real unified memory through the page cache.

References: pinned engine da64c5cbb8cf6bfd39be19da43573fdfd484c43a, python/sglang/srt/managers/scheduler.py (dynamic_chunk_sizer.predict); python/sglang/srt/layers/attention/deepseek_v4_backend.py (_low_ratio_index_topk_dense, _publish_or_consume_candidates). Independent discussion: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/79f656a65f189239cc575bf5c5d1b5cf579d4c41/docs/chunked-prefill-memory.md . No community adapter code is copied.
