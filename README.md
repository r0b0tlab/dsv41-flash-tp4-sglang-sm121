# dsv4.1-flash — DeepSeek-V4.1-Flash TP=4 SGLang on 4×GB10 (SM121)

Publication package: SGLang TP=4/EP=4 serving of `deepseek-ai/DeepSeek-V4.1-Flash`
on the 4-node CRS812 DGX Spark (GB10/SM121) cluster, with a from-scratch
Engram local-store that eliminates the per-lookup TP all-reduce.

Status: bring-up. See `~/.hermes/plans/2026-09-12_184943-dsv41-flash-tp4-sglang-sm121.md`.

Baseline requirements: multimodality (vision) + DSpark speculative decoding,
SM121-optimized, standard qualification gates before publication.
