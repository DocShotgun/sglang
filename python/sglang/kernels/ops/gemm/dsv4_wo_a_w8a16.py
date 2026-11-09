"""DeepSeek-V4 wo_a with fp8 e4m3 weights and bf16 activations (W8A16).

The checkpoint's wo_a is fp8 with 32x32 power-of-two block scales, which the
loader dequantizes to bf16. Re-deriving fp8 codes that dequantize to the same
bf16 bits halves the weight bytes a decode step reads; the GEMM converts each
weight tile back to those bf16 values and accumulates in fp32 as the bf16
einsum does, so only the summation order differs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

BLOCK = 32
# Measured crossover on RTX PRO 6000 (G=8, R=1024, D=4096): above 256 tokens the
# bf16 GEMM on a dequantized copy is as fast or faster.
MAX_GEMM_TOKENS = 256
# (max tokens, BM, BN, BK, num_warps, num_stages), tuned with DRAM-cold weights
# on RTX PRO 6000.
_GEMM_CONFIGS = (
    (16, 16, 32, 128, 4, 4),
    (32, 32, 64, 128, 4, 4),
    (64, 32, 128, 128, 4, 4),
    (128, 64, 128, 128, 8, 3),
    (256, 128, 128, 128, 8, 3),
)
_REQUANT_ROWS = 1024

_scratch: dict[Tuple[torch.device, Tuple[int, ...]], torch.Tensor] = {}


def requantize_exact(
    weight: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """fp8 e4m3 codes [N, D] and fp32 power-of-two scales [N/32, D/32] that
    dequantize to ``weight`` (bf16 [N, D]) bit for bit, or None if none exist."""
    n, d = weight.shape
    if weight.dtype != torch.bfloat16 or n % BLOCK or d % BLOCK:
        return None
    codes = torch.empty((n, d), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty(
        (n // BLOCK, d // BLOCK), dtype=torch.float32, device=weight.device
    )
    for start in range(0, n, _REQUANT_ROWS):
        rows = weight[start : start + _REQUANT_ROWS]
        blocks = rows.float().view(-1, BLOCK, d // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=(1, 3))
        # The smallest 2^k with amax <= 448 * 2^k. Any exact source scale is at
        # least this large, so dividing by it only shifts exponents up.
        mantissa, exponent = torch.frexp(amax)
        k = exponent - 9 + (mantissa > 0.875).to(exponent.dtype)
        scale = torch.where(
            amax > 0, torch.ldexp(torch.ones_like(amax), k), torch.ones_like(amax)
        )
        q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
        back = (q.float() * scale[:, None, :, None]).to(torch.bfloat16)
        if not torch.equal(
            back.view(torch.int16).flatten(), rows.view(torch.int16).flatten()
        ):
            return None
        codes[start : start + _REQUANT_ROWS] = q.view(-1, d)
        scales[start // BLOCK : (start + rows.shape[0]) // BLOCK] = scale
    return codes, scales


@triton.jit
def _dequant_kernel(W, S, Out, D: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    w = tl.load(W + row * D + cols).to(tl.float32)
    s = tl.load(S + (row // 32) * (D // 32) + cols // 32)
    tl.store(Out + row * D + cols, (w * s).to(tl.bfloat16))


def reserve_dequant_scratch(
    shape: Tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """The shared bf16 buffer ``dequantize_to_scratch`` writes; allocate it
    before graph capture so no capture pool owns it."""
    key = (device, tuple(shape))
    if key not in _scratch:
        _scratch[key] = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return _scratch[key]


def dequantize_to_scratch(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """bf16 copy of fp8 ``weight`` [G, R, D] in a buffer shared by every layer of
    that shape; valid until the next call."""
    out = reserve_dequant_scratch(weight.shape, weight.device)
    groups, rows, dim = weight.shape
    chunk = math.gcd(dim, 1024)
    _dequant_kernel[(groups * rows, dim // chunk)](
        weight, scale, out, dim, chunk, num_warps=4
    )
    return out


@triton.jit
def _wo_a_w8a16_kernel(
    X,
    W,
    S,
    Y,
    M,
    stride_xm,
    stride_xg,
    R: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m, pid_n, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    groups = tl.num_programs(2)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    scale_rows = (group * R + pid_n * BN) // 32 + tl.arange(0, BN // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(
            X + m[:, None] * stride_xm + group * stride_xg + k[None, :],
            m[:, None] < M,
            0,
        )
        w = tl.load(W + (group * R + n[None, :]) * D + k[:, None]).to(tl.float32)
        s = tl.load(
            S
            + scale_rows[None, :] * (D // 32)
            + (k0 // 32 + tl.arange(0, BK // 32))[:, None]
        )
        w = tl.reshape(
            tl.reshape(w, (BK // 32, 32, BN // 32, 32)) * s[:, None, :, None], (BK, BN)
        )
        acc = tl.dot(x, w.to(tl.bfloat16), acc)
    tl.store(
        Y + (m[:, None] * groups + group) * R + n[None, :],
        acc.to(tl.bfloat16),
        m[:, None] < M,
    )


def wo_a_w8a16(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``einsum('tgd,grd->tgr', x, dequant(weight, scale))`` for
    T <= MAX_GEMM_TOKENS: x bf16 [T, G, D] with unit stride in D, weight fp8 e4m3
    [G, R, D], scale fp32 [G, R/32, D/32]; returns contiguous bf16 [T, G, R]."""
    tokens = x.shape[0]
    groups, rows, dim = weight.shape
    assert tokens <= MAX_GEMM_TOKENS and x.shape[1:] == (groups, dim)
    assert x.dtype == torch.bfloat16 and x.stride(2) == 1
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
    assert scale.dtype == torch.float32 and scale.is_contiguous()
    assert scale.shape == (groups, rows // BLOCK, dim // BLOCK)
    out = torch.empty((tokens, groups, rows), dtype=torch.bfloat16, device=x.device)
    if tokens == 0:
        return out
    _, bm, bn, bk, warps, stages = next(c for c in _GEMM_CONFIGS if tokens <= c[0])
    assert rows % bn == 0 and dim % bk == 0
    _wo_a_w8a16_kernel[(triton.cdiv(tokens, bm), rows // bn, groups)](
        x,
        weight,
        scale,
        out,
        tokens,
        x.stride(0),
        x.stride(1),
        rows,
        dim,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out
