"""MiMo vision attention needs GQA, local windows, and softmax sinks."""

import pytest
import torch

from sglang.srt.layers.attention.vision import VisionSdpaAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("single_precision", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("use_sinks,window", [(False, -1), (False, 1), (True, 1)])
def test_packed_vision_sdpa_matches_reference(
    kv_heads, single_precision, dtype, device, use_sinks, window
):
    torch.manual_seed(0)
    tokens, heads, dim = 6, 4, 64
    q = torch.randn(tokens, heads, dim).to(device=device, dtype=dtype)
    k = torch.randn(tokens, kv_heads, dim).to(device=device, dtype=dtype)
    v = torch.randn(tokens, kv_heads, dim).to(device=device, dtype=dtype)
    sinks = torch.linspace(-1, 1, heads, device=device) if use_sinks else None
    cu_seqlens = torch.tensor([0, 2, tokens], dtype=torch.int32)
    layer = VisionSdpaAttention(
        head_dim=dim,
        num_heads=heads,
        num_kv_heads=kv_heads,
        flatten_batch=True,
        softmax_in_single_precision=single_precision,
    )
    actual = layer(
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
        window_size=(window, window),
        s_aux=sinks,
    )
    expected = torch.empty_like(q)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for token in range(start, end):
            lo = max(start, token - window) if window >= 0 else start
            hi = min(end, token + window + 1) if window >= 0 else end
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                scores = k[lo:hi, kv_head].float() @ q[token, head].float() / dim**0.5
                if sinks is not None:
                    scores = torch.cat([scores, sinks[head : head + 1]])
                probs = scores.softmax(-1)[: hi - lo]
                expected[token, head] = probs @ v[lo:hi, kv_head].float()
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
