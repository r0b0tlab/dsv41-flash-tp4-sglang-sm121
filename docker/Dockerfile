FROM lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9
# Base pinned by arm64 manifest digest (index 4a5d132a...). SGLang source
# commit baked in: da64c5cbb8cf6bfd39be19da43573fdfd484c43a.

COPY adapter/ /opt/dsv41/adapter/

ENV PYTHONPATH=/opt/dsv41/adapter:${PYTHONPATH:-}

# Plain HTTP probe; python3 -S so the check never imports the engine (and
# never triggers the sitecustomize engram hook).
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
  CMD python3 -S /opt/dsv41/adapter/health.py || exit 1
