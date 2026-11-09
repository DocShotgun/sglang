"""The W8A16 wo_a must compute with exactly the loader's bf16 weights."""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm.dsv4_wo_a_w8a16 import (
    MAX_GEMM_TOKENS,
    dequantize_to_scratch,
    requantize_exact,
    wo_a_w8a16,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="needs a CUDA GPU with fp8 e4m3 conversions",
)

DEVICE = "cuda"
GROUPS, RANK, DIM = 8, 1024, 4096


def _checkpoint_bf16(block: int, seed: int) -> torch.Tensor:
    """Dequantize a random fp8 checkpoint as the loader does: codes * 2^k per
    block, rounded to bf16. Codes cover zeros, subnormals and +-448."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    n = GROUPS * RANK
    codes = torch.randint(0, 256, (n, DIM), device=DEVICE, generator=generator)
    codes = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    codes = torch.nan_to_num(codes, nan=0.0)
    codes[:block, :block] = 0.0
    exponents = torch.randint(
        -40, 10, (n // block, DIM // block), device=DEVICE, generator=generator
    )
    scale = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    blocks = codes.view(n // block, block, DIM // block, block)
    return (blocks * scale[:, None, :, None]).view(n, DIM).to(torch.bfloat16)


@pytest.mark.parametrize("block", [32, 128])
def test_requantized_weight_dequantizes_to_the_same_bits(block):
    # V4.1 checkpoints use 32x32 blocks, V4 128x128; a 128 block's scale also
    # serves each of its 32x32 sub-blocks.
    weight = _checkpoint_bf16(block, seed=block)
    codes, scales = requantize_exact(weight)
    restored = dequantize_to_scratch(
        codes.view(GROUPS, RANK, DIM), scales.view(GROUPS, RANK // 32, DIM // 32)
    )
    assert torch.equal(
        restored.flatten(0, 1).view(torch.int16), weight.view(torch.int16)
    )


def test_requantize_rejects_weights_with_no_fp8_form():
    # bf16 values with more than 3 mantissa bits in a block would change if
    # forced onto fp8 codes; the layer must then stay in bf16.
    weight = torch.randn(GROUPS * RANK, DIM, device=DEVICE).to(torch.bfloat16)
    assert requantize_exact(weight) is None


@pytest.mark.parametrize("tokens", [0, 1, 16, 17, 64, 65, 129, MAX_GEMM_TOKENS])
def test_w8a16_matches_the_bf16_einsum(tokens):
    weight = _checkpoint_bf16(32, seed=1)
    codes, scales = requantize_exact(weight)
    # The attention output reaches wo_a as a [T, G, D] view with a padded row
    # stride.
    o = torch.randn(tokens, GROUPS * DIM + 512, device=DEVICE).to(torch.bfloat16)
    o = o[:, : GROUPS * DIM].view(tokens, GROUPS, DIM)
    y = wo_a_w8a16(
        o,
        codes.view(GROUPS, RANK, DIM),
        scales.view(GROUPS, RANK // 32, DIM // 32),
    )
    wo_a = weight.view(GROUPS, RANK, DIM)
    exact = torch.einsum("tgd,grd->tgr", o.double(), wo_a.double())
    reference = torch.einsum("tgd,grd->tgr", o, wo_a)
    assert y.shape == reference.shape and y.is_contiguous()
    # Only the fp32 summation order differs from the bf16 einsum, so the error
    # stays at the bf16 output rounding plus fp32 accumulation noise; a wrong
    # scale or tile index is off by about magnitude / sqrt(D).
    magnitude = torch.einsum("tgd,grd->tgr", o.double().abs(), wo_a.double().abs())
    bound = (
        (reference.double() - exact).abs()
        + exact.abs() * 2.0**-8
        + magnitude * 2.0**-16
    )
    assert torch.all((y.double() - exact).abs() <= bound)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
