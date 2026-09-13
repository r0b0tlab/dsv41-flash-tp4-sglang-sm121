"""sitecustomize: install dsv4.1-flash adapter hooks at interpreter start.

DSV41_ENGRAM_FILE_STORE=1 (set by the serve profiles / launcher) enables the
file-backed Engram table store. Runs in every non-``-S`` python process of
the serving image; the patch is class-level and inert until an
EngramEmbedding is constructed, so API/tokenizer processes pay nothing but
the sglang import they perform anyway. The container healthcheck runs with
``python3 -S`` and skips this file entirely.
"""

import os

if os.environ.get("DSV41_ENGRAM_FILE_STORE", "0") == "1":
    model_path = os.environ.get("DSV41_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError(
            "DSV41_ENGRAM_FILE_STORE=1 requires DSV41_MODEL_PATH to point at "
            "the DeepSeek-V4.1-Flash checkpoint directory")

    import sglang.srt.layers.engram  # noqa: F401  (heavy; engine imports it anyway)

    from sglang_patch.engram_file_store import install as _install_engram

    _install_engram(model_path)
