"""Stable within-expert row order for moe_align_block_size output."""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_align_stable_order import order_rows_within_experts
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = "cuda"


def _sorted_by_run_and_id(sorted_ids, expert_ids, post_padded, block_size, numel):
    """Reference: a stable sort of the live slots by (expert run, row id)."""
    positions = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    new_run = torch.ones(expert_ids.shape, dtype=torch.int64, device=DEVICE)
    new_run[1:] = expert_ids[1:] != expert_ids[:-1]
    run = torch.cumsum(new_run, 0)
    block = (positions // block_size).clamp(max=expert_ids.numel() - 1)
    key = run[block] * (numel + 1) + sorted_ids.to(torch.int64).clamp(max=numel)
    key = torch.where(
        positions < post_padded[0],
        key,
        (expert_ids.numel() + 2) * (numel + 1) + positions,
    )
    return sorted_ids[torch.sort(key, stable=True).indices]


def _routes(tokens, topk, experts, generator):
    """Distinct experts per token, plus the cases KT routing produces: masked
    pad rows (-1), CPU routes remapped to expert 0, and one hot expert."""
    ids = torch.rand(tokens, experts, device=DEVICE, generator=generator)
    ids = ids.argsort(dim=1)[:, :topk].to(torch.int32)
    if tokens > 3:
        ids[: tokens // 7] = -1
        ids[tokens // 7 : tokens // 5, : max(1, topk // 2)] = 0
    hot = torch.rand(tokens, device=DEVICE, generator=generator) < 0.3
    ids[hot, 0] = experts - 1
    return ids


@pytest.mark.parametrize(
    "tokens, topk, experts, block_size, inactive_prefix",
    [
        (3, 8, 8, 8, False),  # fewer experts than one scan tile of bins
        (160, 6, 384, 48, True),  # block size not a power of two
        (2048, 1, 64, 16, False),  # KT's compact rows
        (8192, 8, 384, 64, True),  # more chunks than one scan tile
        (8192, 6, 256, 32, False),
    ],
)
def test_matches_a_stable_sort_by_expert_run(
    tokens, topk, experts, block_size, inactive_prefix
):
    generator = torch.Generator(device=DEVICE).manual_seed(tokens * topk + experts)
    ids = _routes(tokens, topk, experts, generator)
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        ids, block_size, experts, return_inactive_prefix_count=inactive_prefix
    )
    expected = _sorted_by_run_and_id(
        sorted_ids.clone(), expert_ids, post_padded, block_size, ids.numel()
    )
    out = order_rows_within_experts(
        sorted_ids,
        expert_ids,
        post_padded,
        block_size=block_size,
        numel=ids.numel(),
        num_experts=experts,
    )
    assert torch.equal(out, expected)


def test_graph_replay_orders_new_routes():
    """A captured graph must order the routes present at replay time."""
    tokens, topk, experts, block_size = 2048, 6, 384, 16
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    static_ids = _routes(tokens, topk, experts, generator)

    def align_and_order():
        sorted_ids, expert_ids, post_padded = moe_align_block_size(
            static_ids, block_size, experts, return_inactive_prefix_count=True
        )
        out = order_rows_within_experts(
            sorted_ids,
            expert_ids,
            post_padded,
            block_size=block_size,
            numel=static_ids.numel(),
            num_experts=experts,
        )
        return out, expert_ids, post_padded

    align_and_order()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_experts, captured_total = align_and_order()
    for _ in range(5):
        static_ids.copy_(_routes(tokens, topk, experts, generator))
        graph.replay()
        sorted_ids, expert_ids, post_padded = moe_align_block_size(
            static_ids, block_size, experts, return_inactive_prefix_count=True
        )
        expected = _sorted_by_run_and_id(
            sorted_ids, expert_ids, post_padded, block_size, static_ids.numel()
        )
        total = int(post_padded[0])
        assert torch.equal(captured_total, post_padded)
        assert torch.equal(
            captured_experts[: total // block_size], expert_ids[: total // block_size]
        )
        assert torch.equal(captured[:total], expected[:total])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
