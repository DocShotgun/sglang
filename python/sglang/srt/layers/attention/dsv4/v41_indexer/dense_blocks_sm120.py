"""SM120 decode of the DSv4.1 candidate layers on dense block ids.

DeepGEMM has no SM120 paged sparse MQA logits kernel, so SM120 serves candidates on
dense block ids. ``DenseBlocksBackend`` decodes those through its non-paged scores,
which gather a slot table and score every column up to the page table's width (the
context limit) in Triton. This backend scores with DeepGEMM's dense paged fp4
kernel instead, as the SM120 full top-k layers and prefill already do, and keeps the
same candidate selection and top-k. Prefill is inherited unchanged.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import finish_paged_indexer_topk
from sglang.kernels.ops.attention.dsv4.index_logits import (
    deep_gemm_fp4_paged_mqa_logits,
)

from .dense_blocks import BlockIds, DenseBlocksBackend
from .scoring import get_deep_gemm_decode_data
from .types import DecodeInputs


class Sm120DenseBlocksBackend(DenseBlocksBackend):
    def publish_decode(self, inputs: DecodeInputs) -> Optional[BlockIds]:
        blocks = []
        for rows, logits, lens in self._decode_logits(inputs):
            # DeepGEMM leaves columns past each row's length unwritten; both selections read them.
            columns = torch.arange(logits.shape[1], device=logits.device)
            logits.masked_fill_(columns[None, :] >= lens[:, None], -torch.inf)
            blocks.append(
                select_candidate_block_ids(
                    logits,
                    lens[:, None],
                    topk_blocks=self.topk_blocks,
                    block_size=self.block_size,
                )
            )
            k = min(inputs.indexer.index_topk, logits.shape[1])
            idx = logits.topk(k, dim=-1, sorted=False).indices
            self._write(inputs, rows=rows, logits=logits, lens=lens, idx=idx)
        if not blocks:
            return None
        return BlockIds(blocks=blocks[0] if len(blocks) == 1 else torch.cat(blocks))

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[BlockIds],
    ) -> None:
        for rows, logits, lens in self._decode_logits(inputs):
            assert published is not None
            blocks = published.blocks[rows]
            assert blocks.shape[0] == lens.shape[0]
            k = min(inputs.indexer.index_topk, logits.shape[1])
            idx = topk_among_blocks(logits, lens, blocks, k, block_size=self.block_size)
            self._write(inputs, rows=rows, logits=logits, lens=lens, idx=idx)

    def _decode_logits(
        self, inputs: DecodeInputs
    ) -> Iterator[Tuple[slice, torch.Tensor, torch.Tensor]]:
        """Per DeepGEMM row chunk: its rows, fp32 logits [rows, lmax] valid below
        each row's compressed length, and those lengths. Nothing when nothing is
        visible, as ``decode_scores`` returns None."""
        inputs.reset_outputs()
        ratio = inputs.compress_ratio
        metadata = inputs.paged_metadata
        bs = inputs.req_rows.shape[0]
        lmax = min(metadata.max_compressed_seq_len, self.req_to_token.shape[1] // ratio)
        if bs == 0 or lmax == 0:
            return
        lens = (inputs.positions + 1) // ratio
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        for rows, schedule in metadata.row_chunks():
            q, seq_lens, page_table = data.paged_logits_args(metadata, rows)
            logits = deep_gemm_fp4_paged_mqa_logits(
                q,
                data.k_cache,
                data.weights[rows],
                seq_lens,
                page_table,
                schedule,
                metadata.max_compressed_seq_len,
            )
            yield rows, logits[:, :lmax], lens[rows]

    def _write(
        self,
        inputs: DecodeInputs,
        *,
        rows: slice,
        logits: torch.Tensor,
        lens: torch.Tensor,
        idx: torch.Tensor,
    ) -> None:
        # Sorted pool slots, -1 past a row's length, as write_decode stores them.
        finish_paged_indexer_topk(
            idx,
            logits,
            lens,
            inputs.req_rows[rows],
            self.req_to_token,
            inputs.out_page_indices[rows],
            None,
            inputs.compress_ratio,
            False,
        )
