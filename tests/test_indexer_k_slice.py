"""Exact K-sliced topk for the SM121 Torch prefill indexer.

indexer.scores returns [t, n] after the head reduction. Slice n, take per-slice
topk, merge. Union-of-per-slice-topk is exact for global topk.
"""
import pathlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'adapter'))
from sglang_patch.torch_indexer_budget import k_chunk_for, sliced_exact_topk  # noqa: E402


def test_k_chunk_cap_independent_of_score_budget():
    # 1024 MiB / (32 heads * 2 B * 256 query) = 65536 from the score formula alone.
    assert k_chunk_for(1024 << 20, 32, 256, k_chunk_max=10**9) == 65536
    assert k_chunk_for(1024 << 20, 32, 256) == 2048
    assert k_chunk_for(16 << 20, 32, 256) == 1024
    assert k_chunk_for(1024 << 20, 32, 256, k_chunk_max=1) == 1


def test_sliced_topk_matches_full():
    torch.manual_seed(0)
    rows, lc, k, k_chunk = 4, 300, 16, 64
    scores = torch.randn(rows, lc)
    lens = torch.tensor([300, 280, 129, 16])
    full = sliced_exact_topk(scores, lens, k, lc)
    part = sliced_exact_topk(scores, lens, k, k_chunk)
    assert torch.equal(full.sort(dim=-1).values, part.sort(dim=-1).values)


def test_out_of_range_masked():
    scores = torch.zeros(1, 8)
    scores[0, 7] = 9
    lens = torch.tensor([4])  # last four keys illegal
    idx = sliced_exact_topk(scores, lens, k=2, k_chunk=3)
    assert int(idx.max()) < 4


def test_install_rejects_full_prefix_dequant(monkeypatch):
    from sglang_patch.torch_indexer_budget import apply_to_backend, k_chunk_for

    seen = []

    class Pool:
        def get_low_ratio_index_k_dequant(self, layer_id, slots):
            n = int(slots.numel())
            seen.append(n)
            return torch.zeros(n, 8)

    class Indexer:
        index_topk = 4
        is_candidate_source = False
        uses_candidates = False
        n_heads = 2

        def queries(self, q_lora, freqs):
            return torch.zeros(q_lora.shape[0], self.n_heads, 8)

        def head_weights(self, x):
            return torch.ones(x.shape[0], self.n_heads)

        def scores(self, q, index_k, weights):
            return torch.randn(q.shape[0], index_k.shape[0])

    class Layer:
        compress_ratio = 1
        layer_id = 0
        indexer = Indexer()
        freqs_cis = torch.zeros(32, 8)

    class Core:
        def sparse_page_indices(self, ratio):
            return torch.full((8, 4), -1, dtype=torch.int32)

        def sparse_raw_indices(self, ratio):
            return torch.full((8, 4), -1, dtype=torch.int32)

    class Backend:
        def __init__(self):
            self.token_to_kv_pool = Pool()
            self.forward_metadata = types.SimpleNamespace(core_metadata=Core())
            self.req_to_token = torch.arange(32).unsqueeze(0).repeat(4, 1)
            self.candidate_masks = None

        def _low_ratio_index_topk_torch(self, layer, x, q_lora, req, pos):
            raise AssertionError('unpatched torch indexer must not run')

    module = types.SimpleNamespace(DeepseekV4AttnBackend=Backend, _TORCH_INDEXER_SCORE_BUDGET_BYTES=1024<<20, _TORCH_INDEXER_K_CHUNK_MAX=4)
    apply_to_backend(module)
    b = Backend()
    req = torch.zeros(8, dtype=torch.int64)
    pos = torch.arange(8)
    x = torch.zeros(8, 4)
    q_lora = torch.zeros(8, 4)
    b._low_ratio_index_topk_torch(Layer(), x, q_lora, req, pos)
    assert seen and max(seen) <= 4 + 256
