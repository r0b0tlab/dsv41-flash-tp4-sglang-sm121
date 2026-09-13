"""TDD for the file-backed Engram table store.

Run: python3 -m pytest tests/test_engram_store.py -q
Covers the three properties the serving path depends on:
  1. geometry — the store derives row counts and contiguous byte ranges from
     the real checkpoint layout (fixed-config fixtures here).
  2. fidelity — gather output is bit-identical to a reference dequant of the
     same rows.
  3. no-collective — the forward path used at TP>1 is the direct gather,
     never the owned-rows + all-reduce path.
"""

import ctypes
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapter"))

from sglang_patch import engram_file_store as store_mod  # noqa: E402


def write_shard(path: Path, rows: int, dim: int = 256, seed: int = 0):
    """Build a minimal checkpoint-like shard: header + contiguous
    embed.weight [rows, dim] u8 + embed.scale [rows, dim//32] u8."""
    rng = np.random.default_rng(seed)
    # Keep weight bytes in the finite e4m3 range: 0x7F and 0xFF are NaN in
    # float8_e4m3fn, and NaN != NaN would defeat bit-exact assertions.
    w = rng.integers(1, 0x60, size=(rows, dim), dtype=np.uint8)
    s = rng.integers(120, 136, size=(rows, dim // 32), dtype=np.uint8)
    header = {
        "layers.0.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, dim],
                                          "data_offsets": [0, rows * dim]},
        "layers.0.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, dim // 32],
                                         "data_offsets": [rows * dim, rows * dim + rows * dim // 32]},
    }
    # Two-pass: the header length depends on the offset digit counts, which
    # are stable across the passes — pad with spaces (as safetensors does)
    # so the rewrite keeps the byte length fixed.
    def build(base):
        h = {
            "layers.0.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, dim],
                                              "data_offsets": [base, base + rows * dim]},
            "layers.0.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, dim // 32],
                                             "data_offsets": [base + rows * dim,
                                                              base + rows * dim + rows * dim // 32]},
        }
        return json.dumps(h).encode()

    # Iterate to a fixed point: header length -> base -> header length.
    # Pad with trailing spaces (safetensors does the same) so the length is
    # stable regardless of offset digit counts.
    base = 8 + len(build(0))
    for _ in range(4):
        hb = build(base)
        total = 8 + len(hb)
        if total == base:
            break
        base = total
    else:
        raise AssertionError("header length did not converge")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(w.tobytes())
        f.write(s.tobytes())
    return w, s, base


def reference_dequant(w: np.ndarray, s: np.ndarray):
    """float8_e4m3 dequant with e8m0 block scales, bf16 out — mirrors the
    upstream engram_gather contract (scale per 32 channels)."""
    wt = torch.frombuffer(w.reshape(-1), dtype=torch.uint8).view(torch.float8_e4m3fn)
    st = torch.frombuffer(s.reshape(-1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    wt = wt.reshape(w.shape[0], w.shape[1]).float()
    st = st.reshape(s.shape[0], s.shape[1]).float()
    st = st.repeat_interleave(32, dim=1)
    return (wt * st).to(torch.bfloat16)


def test_geometry_from_real_layout(tmp_path):
    rows = 1000
    shard = tmp_path / "shard.safetensors"
    write_shard(shard, rows)
    st = store_mod.FileEngramStore(str(shard), "layers.0.engram", rows, 256)
    assert st.rows == rows
    assert st.dim == 256
    assert st.weight_ptr != st.scale_ptr
    # pointers differ by exactly the weight byte count
    assert st.scale_ptr - st.weight_ptr == rows * 256


def test_gather_bit_exact(tmp_path):
    rows = 4096
    shard = tmp_path / "shard.safetensors"
    w, s, _ = write_shard(shard, rows, seed=7)
    st = store_mod.FileEngramStore(str(shard), "layers.0.engram", rows, 256)

    ids = torch.tensor([0, 1, 5, rows - 1, 123, 777], dtype=torch.int64)
    out = st.gather(ids)  # CPU reference gather in the store module
    ref = reference_dequant(w[ids.numpy()], s[ids.numpy()])
    assert torch.equal(out, ref)


def test_gather_out_of_range_rows_zero(tmp_path):
    rows = 128
    shard = tmp_path / "shard.safetensors"
    w, s, _ = write_shard(shard, rows, seed=3)
    st = store_mod.FileEngramStore(str(shard), "layers.0.engram", rows, 256)
    ids = torch.tensor([0, rows, rows + 5, 2**40], dtype=torch.int64)
    out = st.gather(ids)
    assert torch.equal(out[0], reference_dequant(w[0:1], s[0:1])[0])
    assert torch.equal(out[1], torch.zeros(256, dtype=torch.bfloat16))
    assert torch.equal(out[2], torch.zeros(256, dtype=torch.bfloat16))
    assert torch.equal(out[3], torch.zeros(256, dtype=torch.bfloat16))


def test_shared_layout_shim_no_allreduce():
    """The shim must present layout='shared' and the installed forward must
    be the direct gather: any call into tensor_model_parallel_all_reduce
    during an Engram lookup is a design failure. Asserts the static
    contract (shim attributes); the live no-NCCL proof runs at admission
    (Phase 6/7) via the profiler."""
    shim = store_mod._FileHostTableShim()
    assert shim.layout == "shared"
    assert store_mod._InstallState.installed is False
    # gather path itself performs no collectives by construction: it only
    # touches the mmap views.
    import inspect
    src = inspect.getsource(store_mod.FileEngramStore.gather)
    assert "all_reduce" not in src and "dist" not in src
