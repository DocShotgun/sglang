"""MiMo's audio tokenizer encoder attention must be causal and keep its local window,
like the checkpoint's reference encoder."""

import os
import socket

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention import vision
from sglang.srt.models.mimo_audio import AudioEncoderAttention
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EMBED_DIM = 32
NUM_HEADS = 4
CU_SEQLENS = torch.tensor([0, 5, 12], dtype=torch.int32)


@pytest.fixture(scope="module")
def gloo_world():
    """a one-rank cpu process group, so the row-parallel proj can run"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    publish_build_topology(tp_size=1)
    initialize_model_parallel(backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _build(monkeypatch, backend, window_size):
    monkeypatch.setattr(
        vision.VisionAttention,
        "_determine_attention_backend",
        lambda self, passed_backend: backend,
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        get_context().override_server_args(),
    ):
        layer = AudioEncoderAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            window_size=window_size,
            causal=True,
        )
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return layer


def _reference(layer, x, window_left):
    qkv = F.linear(x, layer.attn.qkv_proj.weight, layer.attn.qkv_proj.bias)
    # [heads, tokens, head_dim]
    q, k, v = qkv.unflatten(-1, (3, NUM_HEADS, -1)).permute(1, 2, 0, 3)
    out = torch.empty_like(q)
    for start, end in zip(CU_SEQLENS[:-1].tolist(), CU_SEQLENS[1:].tolist()):
        pos = torch.arange(end - start)
        distance = pos[:, None] - pos[None, :]
        allowed = (distance >= 0) & ((distance <= window_left) | (window_left < 0))
        seg = slice(start, end)
        out[:, seg] = F.scaled_dot_product_attention(
            q[:, seg], k[:, seg], v[:, seg], attn_mask=allowed
        )
    out = out.transpose(0, 1).flatten(1)
    return F.linear(out, layer.attn.proj.weight, layer.attn.proj.bias)


@pytest.mark.parametrize("window_size", [(2, 0), (-1, -1)])
def test_audio_encoder_attention_is_causal_and_windowed(
    monkeypatch, gloo_world, window_size
):
    torch.manual_seed(0)
    layer = _build(monkeypatch, "sdpa", window_size)
    x = torch.randn(int(CU_SEQLENS[-1]), EMBED_DIM)

    with torch.no_grad():
        actual = layer(x, CU_SEQLENS, max_seqlen=7)
        expected = _reference(layer, x, window_size[0])

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
