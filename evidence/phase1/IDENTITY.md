# Frozen identity — dsv4.1-flash (2026-09-12)

## Model
- Repo: deepseek-ai/DeepSeek-V4.1-Flash (ungated, MIT)
- Revision: dba1be0a40aa45a94ad051997016db3960a90277 (2026-09-10)
- 763B params, 48 shards, usedStorage 510,302,811,119 B (~475.5 GiB)

## Runtime image (pin by digest; tag is moving)
- Image: lmsysorg/sglang:dev-dsv41
- OCI index digest:  sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
- arm64 manifest:    sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9
- arm64 config:      sha256:381b27ffa19bfbade2bf69bb103395e59a7e82ab0adf176b15b492e9df5aaf7b
- Created: 2026-09-11T18:23:37Z
- SGLang source commit (image labels ai.sglang.build.commit / org.opencontainers.image.revision):
  da64c5cbb8cf6bfd39be19da43573fdfd484c43a
  (built by GH Actions run 34631178591)
- No stable SGLang release contains V4.1 support as of this freeze
  (latest tag v0.5.19 = 2026-09-05, predates day-0). Re-check for a stable
  tag before final publication.

## Cluster (target)
- 4× GB10/SM121, CRS812 switched RoCE (VERDICT 2026-09-11), TP=4/EP=4.
