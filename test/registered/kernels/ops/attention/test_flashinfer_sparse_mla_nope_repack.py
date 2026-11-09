"""SM120 native-NoPE sparse MLA: the pool repack and the per-slot gather agree.

flashinfer_sparse_mla_forward presents GLM-5.3-Flash's 528 B NoPE rows to the
656 B RoPE kernel either by repacking the whole KV pool (when the call references
at least as many slots as the pool has rows) or by gathering the referenced rows.
Both must feed the kernel identical bytes, so their outputs must be bit-identical.
"""

import unittest
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip(
        "FlashInfer SM120 sparse MLA requires CUDA SM 12.x.", allow_module_level=True
    )
pytest.importorskip("flashinfer.mla")

from sglang.kernels.ops.attention import flash_mla_sm120  # noqa: E402

# GLM-5.3-Flash geometry: 64 heads, kv_lora_rank 512, no RoPE, index_topk 2048,
# rows of 512 fp8 latents + 4 fp32 group scales.
HEADS, LORA, TOPK, PAGE, ROW_BYTES = 64, 512, 2048, 64, 528


def _kv_rows(n_rows: int, generator: torch.Generator) -> torch.Tensor:
    latent = torch.randn(n_rows, LORA, generator=generator, device="cuda") * 2
    scales = torch.rand(n_rows, 4, generator=generator, device="cuda") * 0.5 + 0.75
    return torch.cat(
        [
            latent.to(torch.float8_e4m3fn).view(torch.uint8),
            scales.view(torch.uint8).view(n_rows, 16),
        ],
        dim=1,
    )


def _indices(tokens: int, n_rows: int, generator: torch.Generator) -> torch.Tensor:
    # Variable top-k lengths with -1 padding; the last token selects nothing.
    indices = torch.full((tokens, TOPK), -1, dtype=torch.int32, device="cuda")
    for t in range(tokens - 1):
        length = int(torch.randint(1, TOPK + 1, (1,), generator=generator, device="cuda"))
        perm = torch.randperm(n_rows, generator=generator, device="cuda")
        indices[t, :length] = perm[:length].to(torch.int32)
    return indices


class TestNopePoolRepackMatchesGather(CustomTestCase):
    def _forward(self, q, kv_rows, indices, workspace):
        n_pages = kv_rows.shape[0] // PAGE
        repacks = []
        real_scratch = flash_mla_sm120._nope_repack_scratch

        def recording_scratch(tag, n_rows, device):
            repacks.append(tag)
            return real_scratch(tag, n_rows, device)

        with patch.object(flash_mla_sm120, "_nope_repack_scratch", recording_scratch):
            out = flash_mla_sm120.flashinfer_sparse_mla_forward(
                q=q,
                kv_cache=kv_rows.view(n_pages, PAGE, ROW_BYTES),
                indices=indices.clone(),
                seq_lens=torch.ones(q.shape[0], dtype=torch.int32, device="cuda"),
                workspace_buffer=workspace,
                page_size=PAGE,
                kv_cache_dim=ROW_BYTES,
                qk_nope_head_dim=256,
                kv_lora_rank=LORA,
                qk_rope_head_dim=0,
                sparse_mla_top_k=TOPK,
                sm_scale=1.0 / 24.0,
                skip_softmax_threshold_scale_factor=None,
            )
        torch.cuda.synchronize()
        return out, repacks

    def test_pool_and_gather_paths_are_bit_identical(self):
        workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
        for tokens in (2, 4, 13):
            with self.subTest(tokens=tokens):
                # A pool of exactly tokens * TOPK rows sits on the repack threshold;
                # one more page of unreferenced rows drops below it.
                n_rows = tokens * TOPK
                kv_rows = _kv_rows(n_rows + PAGE, generator)
                indices = _indices(tokens, n_rows, generator)
                q = (
                    torch.randn(tokens, HEADS, LORA, generator=generator, device="cuda")
                    / 4
                ).to(torch.bfloat16)

                pooled, pool_tags = self._forward(q, kv_rows[:n_rows], indices, workspace)
                gathered, gather_tags = self._forward(q, kv_rows, indices, workspace)

                self.assertEqual(pool_tags, ["pool"])
                self.assertEqual(set(gather_tags), {"compact"})
                self.assertTrue(torch.isfinite(pooled).all())
                self.assertTrue(torch.equal(pooled, gathered))


if __name__ == "__main__":
    unittest.main()
