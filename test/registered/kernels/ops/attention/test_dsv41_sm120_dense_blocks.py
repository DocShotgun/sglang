from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
PAGE_SIZE = 64
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode_fp4(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles (low first) with one ue8m0 exponent per 32 elements, as fp64."""
    codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor(LEVELS, device=codes.device, dtype=torch.float64)
    values = torch.where((codes & 8) != 0, -levels[codes & 7], levels[codes & 7])
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float64), exponents.long() - 127
    )
    return values * scales.repeat_interleave(32, dim=-1)


def _reference_logits(q_fp4, q_sf, weights, slots, lens, table):
    """fp64 logits from the packed queries and the pool's codes, -inf past lens."""
    rows, width = slots.shape
    position = torch.arange(width, device=slots.device)
    safe = torch.where(position[None, :] < lens[:, None], slots, 0)
    page, offset = safe // PAGE_SIZE, safe % PAGE_SIZE
    payload = table[
        page[:, :, None],
        offset[:, :, None] * 64 + torch.arange(64, device=table.device),
    ]
    exponents = table[
        page[:, :, None],
        PAGE_SIZE * 64 + offset[:, :, None] * 4 + torch.arange(4, device=table.device),
    ]
    keys = _decode_fp4(payload, exponents)  # [rows, width, 128]
    q_codes = q_fp4.view(torch.uint8).view(rows, -1, 64)
    q_exp = torch.stack([(q_sf.view(rows, -1) >> (8 * i)) & 255 for i in range(4)], -1)
    queries = _decode_fp4(q_codes, q_exp)  # [rows, heads, 128]
    scores = torch.bmm(queries, keys.transpose(1, 2)).clamp_min(0)
    logits = (scores * weights.double()[:, :, None]).sum(dim=1)
    return logits.masked_fill(position[None, :] >= lens[:, None], -torch.inf)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 candidate decode on DeepGEMM dense paged scores",
)
def test_sm120_candidate_decode_scores_selects_and_maps():
    """The SM120 candidate decode must score with DeepGEMM exactly as the reference
    indexer scores, keep the dense-block candidate rules (newest block, top block
    maxima, top-k within the source's blocks) and store pool slots sorted, -1 padded."""
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks_sm120
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks_sm120 import (
        Sm120DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs

    torch.manual_seed(1017)
    device = torch.device("cuda")
    requests, group, heads, topk, topk_blocks = 2, 6, 32, 16, 4
    rows, npages = requests * group, 5
    capacity = npages * PAGE_SIZE

    # Request-local compressed position j lives in pool slot page_of[req, j // 64] * 64 + j % 64.
    page_of = torch.randperm(requests * npages, device=device).view(requests, npages)
    position = torch.arange(capacity, device=device)
    req_to_token = (
        page_of[:, position // PAGE_SIZE] * PAGE_SIZE + position % PAGE_SIZE
    ).int()
    keys = torch.randint(-8, 9, (requests * capacity, HEAD_DIM), device=device)
    table = torch.zeros(
        requests * npages, PAGE_SIZE * 68, device=device, dtype=torch.uint8
    )
    store_fp4_index_k_cache(
        (keys / 8).to(torch.bfloat16),
        table,
        req_to_token.flatten().long(),
        page_size=PAGE_SIZE,
        rne=True,
    )

    # The reference decodes the packed queries, so RoPE and fp4 rounding need no model.
    source_q, consumer_q = (
        (torch.randint(-8, 9, (rows, heads * HEAD_DIM), device=device) / 8).to(
            torch.bfloat16
        )
        for _ in range(2)
    )
    raw_weights = (torch.randint(-8, 9, (rows, heads), device=device) / 8).to(
        torch.bfloat16
    )
    indexer = SimpleNamespace(
        n_heads=heads,
        n_local_heads=heads,
        index_head_dim=HEAD_DIM,
        index_topk=topk,
        head_weight_scale=0.5,
        wq_b=lambda q_lora: (q_lora, None),
        head_weights_raw=lambda x: x,
    )
    pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda layer_id: table)
    backend = Sm120DenseBlocksBackend(
        token_to_kv_pool=pool,
        req_to_token=req_to_token,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=8,
        use_deep_gemm_prefill=True,
    )
    req = torch.arange(rows, device=device, dtype=torch.int32) // group
    slots = req_to_token[req.long()].long()
    freqs_cis = torch.ones(capacity + 1, 32, device=device, dtype=torch.complex64)
    real = dense_blocks_sm120.deep_gemm_fp4_paged_mqa_logits

    for visible in (1, 7, 9, 33, 65, 200, capacity):
        lens = (
            visible - (group - 1) + torch.arange(rows, device=device) % group
        ).clamp_min(1)
        metadata = PagedIndexerMetadata(
            page_size=256,
            compressed_page_size=PAGE_SIZE,
            page_table=page_of[req.long()].int(),
            compressed_seq_lens=lens.int(),
            use_topk_v2=False,
            force_deep_gemm_metadata=True,
            compress_ratio=1,
            rows_per_request=group,
        )
        outputs, scored = [], []

        def record(q, *args):
            logits = real(q, *args)
            scored.append((q, logits.clone()))
            return logits

        def run(q_lora, call, *extra):
            out = torch.full(
                (rows + 1, topk + 8), 12345, device=device, dtype=torch.int32
            )
            inputs = DecodeInputs(
                indexer=indexer,
                layer_id=0,
                compress_ratio=1,
                freqs_cis=freqs_cis,
                x=raw_weights,
                q_lora=q_lora,
                positions=lens - 1,
                req_rows=req,
                paged_metadata=metadata,
                is_verify=True,
                out_page_indices=out,
            )
            outputs.append(out)
            return call(inputs, *extra)

        with patch.object(
            dense_blocks_sm120, "deep_gemm_fp4_paged_mqa_logits", side_effect=record
        ):
            published = run(source_q, backend.publish_decode)
            run(consumer_q, backend.consume_decode, published)

        weights = (raw_weights.float() * 0.5).to(torch.bfloat16)
        references = []
        for (q_fp4, q_sf), logits in scored:
            ref = _reference_logits(
                q_fp4.flatten(0, 1), q_sf.flatten(0, 1), weights, slots, lens, table
            )
            live = torch.isfinite(ref)
            torch.testing.assert_close(
                logits[:, :capacity][live].double(), ref[live], atol=0, rtol=0
            )
            references.append(ref)
        source_ref, consumer_ref = references

        for row, length in enumerate(lens.tolist()):
            blocks = published.blocks[row]
            blocks = blocks[blocks >= 0].long()
            nblocks = (length + 7) // 8
            assert (
                blocks.numel() == blocks.unique().numel() == min(topk_blocks, nblocks)
            )
            assert nblocks - 1 in blocks.tolist()
            maxima = torch.full(
                (nblocks * 8,), -torch.inf, device=device, dtype=torch.float64
            )
            maxima[:length] = source_ref[row, :length]
            maxima = maxima.view(-1, 8).amax(-1)
            maxima[-1] = torch.inf
            torch.testing.assert_close(
                maxima[blocks].sort().values,
                maxima.topk(blocks.numel()).values.sort().values,
            )
            keep = torch.isin(position // 8, blocks)
            consumer_row = consumer_ref[row].masked_fill(~keep, -torch.inf)
            for out, scores in zip(outputs, (source_ref[row], consumer_row)):
                pages = out[row][out[row] >= 0]
                selected = (slots[row, :, None] == pages[None, :].long()).nonzero()[
                    :, 0
                ]
                count = min(topk, int(torch.isfinite(scores).sum()))
                assert pages.numel() == selected.unique().numel() == count
                torch.testing.assert_close(
                    scores[selected].sort().values,
                    scores.topk(count).values.sort().values,
                )
                expected = torch.full_like(out[row], -1)
                expected[:count] = slots[row, selected.sort().values].int()
                torch.testing.assert_close(out[row], expected)
        for out in outputs:
            assert (out[rows] == -1).all().item()
