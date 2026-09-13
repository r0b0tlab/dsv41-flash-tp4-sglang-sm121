# Optimal profile: hybrid sparse-MLA (2026-09-13)

Config: flashinfer decode kernel (<=64-token batches, incl. DSPARK verify
bs8x6=48) + upstream Triton kernel for >64-token prefill calls. Installed at
import by the adapter (prefill fn replaced); env SGLANG_SM120_FLASHMLA_BACKEND
stays flashinfer so decode dispatches to _flash_mla_flashinfer.

Warm results (think-off, usage-block tokens, wall-clock e2e incl TTFT,
acceptance 3.67):
  short_c1   47.3 tok/s   (v1 triton-only: 27.1)
  medium_c1  29.1 tok/s   (11.7)
  prose_c1   22.4 tok/s   (13.7; ref vLLM 24.4)
  counting_c1 60.8 tok/s  (37.1; ref vLLM c1-mean 43.1 -> +41%)
  counting_c4 119.5 tok/s (62.5)

Cold-start note: first lanes run after boot under-rates while flashinfer
autotune caches populate; treat post-warmup numbers as the profile.
