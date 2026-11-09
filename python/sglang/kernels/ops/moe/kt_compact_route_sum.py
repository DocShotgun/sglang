"""Per-token weighted sum of KT's compact GPU-route rows, in a fixed order.

The compact path runs only the GPU-routed (token, slot) pairs through the GPU
experts, one row each, ordered token-major. Each program sums one token's rows
in slot order in fp32, so the result does not depend on scheduling the way an
atomic scatter-add does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_sum_kernel(
    Rows,
    Weights,
    Starts,
    Counts,
    Out,
    hidden,
    stride_rows,
    stride_out,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    start = tl.load(Starts + token).to(tl.int64)
    count = tl.load(Counts + token)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.static_range(TOPK):
        if j < count:
            row = start + j
            weight = tl.load(Weights + row)
            values = tl.load(Rows + row * stride_rows + cols, mask=mask, other=0.0)
            acc += values.to(tl.float32) * weight
    tl.store(Out + token * stride_out + cols, acc.to(Out.dtype.element_ty), mask=mask)


def compact_route_sum(
    rows: torch.Tensor,
    weights: torch.Tensor,
    gpu_route: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``out[t] = sum_j rows[r_j] * weights[r_j]`` over token t's GPU routes, in
    route order: rows [R, H] with unit stride in H, weights fp32 [R], gpu_route
    bool [T, topk] whose row-major True entries are the R rows' (token, slot)."""
    tokens, topk = gpu_route.shape
    hidden = rows.shape[1]
    assert rows.stride(1) == 1 and weights.dtype == torch.float32
    assert weights.shape == (rows.shape[0],)
    counts = gpu_route.sum(dim=1, dtype=torch.int32)
    starts = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    out = torch.empty((tokens, hidden), dtype=out_dtype, device=rows.device)
    if tokens == 0:
        return out
    block = 1024
    _route_sum_kernel[(tokens, triton.cdiv(hidden, block))](
        rows,
        weights,
        starts,
        counts,
        out,
        hidden,
        rows.stride(0),
        out.stride(0),
        topk,
        block,
        num_warps=4,
    )
    return out
