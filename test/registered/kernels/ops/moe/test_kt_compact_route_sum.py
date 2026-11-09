"""KT's compact GPU-route sum on CUDA: reproducible bits, one rounding of the sum."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = "cuda"


class _ExpertScalingGpuMethod:
    """Scales each row by its expert's vector, so every route adds a distinct row."""

    def __init__(self, expert_scales):
        self.expert_scales = expert_scales

    def apply(self, layer, dispatch_output):
        experts = dispatch_output.topk_output.topk_ids.view(-1)
        return SimpleNamespace(
            hidden_states=dispatch_output.hidden_states * self.expert_scales[experts]
        )


def _compact_output(hidden, topk_ids, topk_weights, expert_scales, num_gpu_experts):
    wrapper = object.__new__(KTEPWrapperMethod)
    wrapper.num_gpu_experts = num_gpu_experts
    wrapper.gpu_method = _ExpertScalingGpuMethod(expert_scales)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.zeros_like(topk_weights),
    )
    dispatch = StandardDispatchOutput(
        hidden_states=hidden, hidden_states_scale=None, topk_output=topk_output
    )
    gpu_route = (topk_ids >= 0) & (topk_ids < num_gpu_experts)
    return wrapper._gpu_output_compact(None, dispatch, topk_output, gpu_route)


def test_compact_sum_is_reproducible_and_rounds_once():
    """Repeated prefills of one prompt must give identical bits. The per-token
    sum over several GPU routes was an fp32 atomic scatter-add on CUDA, whose
    order (and so the bf16 rounding of a few elements) changed between runs."""
    tokens, topk, hidden_size, experts, num_gpu = 512, 8, 5120, 64, 48
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    scale = torch.exp(torch.randn(tokens, 1, device=DEVICE, generator=generator))
    hidden = torch.randn(tokens, hidden_size, device=DEVICE, generator=generator)
    hidden = (hidden * scale).to(torch.bfloat16)
    topk_ids = torch.randint(
        0, experts, (tokens, topk), device=DEVICE, generator=generator
    )
    topk_ids[:8] = num_gpu  # tokens with only CPU routes
    topk_weights = torch.rand(tokens, topk, device=DEVICE, generator=generator)
    expert_scales = torch.rand(
        experts, hidden_size, device=DEVICE, generator=generator
    ).to(torch.bfloat16)
    args = (hidden, topk_ids, topk_weights, expert_scales, num_gpu)

    first = _compact_output(*args)
    for _ in range(10):
        assert torch.equal(_compact_output(*args), first)

    gpu_route = topk_ids < num_gpu
    rows = hidden[:, None, :] * expert_scales[topk_ids]
    terms = rows.double() * topk_weights.double()[..., None] * gpu_route[..., None]
    exact = terms.sum(1)
    magnitude = terms.abs().sum(1)
    exponent = torch.floor(torch.log2(exact.abs().clamp_min(1e-38)))
    half_ulp = torch.ldexp(torch.ones_like(exact), (exponent - 8).to(torch.int32))
    # One bf16 rounding of the sum, plus fp32 accumulation error.
    assert torch.all((first.double() - exact).abs() <= half_ulp + magnitude * 2.0**-22)
    assert torch.all(first[:8] == 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
