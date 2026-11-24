"""FA4 vision attention must apply MiMo's local window and softmax sinks."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _fa4_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _reference(q, k, v, cu_seqlens, window, sinks):
    scale = q.shape[-1] ** -0.5
    group = q.shape[1] // k.shape[1]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:end].float().transpose(0, 1)
        ks = k[start:end].float().repeat_interleave(group, 1).transpose(0, 1)
        vs = v[start:end].float().repeat_interleave(group, 1).transpose(0, 1)
        scores = qs @ ks.transpose(1, 2) * scale
        pos = torch.arange(end - start, device=q.device)
        far = (pos[None, :] - pos[:, None]).abs() > window
        scores = scores.masked_fill(far, -torch.inf)
        sink = sinks.float().view(-1, 1, 1).expand(-1, end - start, 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
        out[start:end] = (probs @ vs).transpose(0, 1)
    return out


@pytest.mark.skipif(not _fa4_available(), reason="requires CUDA and FA4 (CuTe)")
def test_fa4_vision_window_and_sinks_match_reference():
    from sglang.srt.layers.attention.vision import VisionFlash4Attention

    torch.manual_seed(0)
    # MiMo-V2.6 ViT: 32 query / 8 kv heads of 40 (packed GQA cannot compile at 40
    # on SM120), +-64 window; segments shorter and longer than the window.
    heads, kv_heads, dim, window = 32, 8, 40, 64
    cu_seqlens = torch.tensor([0, 50, 400, 1400], dtype=torch.int32, device="cuda")
    tokens = int(cu_seqlens[-1])
    q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(tokens, kv_heads, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(tokens, kv_heads, dim, dtype=torch.bfloat16, device="cuda")
    sinks = (torch.randn(heads, device="cuda") * 2).to(torch.bfloat16)

    actual = VisionFlash4Attention()(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        bsz=1,
        seq_len=tokens,
        window_size=(window, window),
        s_aux=sinks,
    )

    expected = _reference(q, k, v, cu_seqlens, window, sinks)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
