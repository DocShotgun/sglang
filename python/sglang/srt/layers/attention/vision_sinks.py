"""Attention sinks and local windows for vision backends.

A sink is one more logit per head, with a zero value, in every row's softmax. A
kernel without sinks but with the log-sum-exp gives the same result after a
per-row rescale, which is how the FA4 backend serves MiMo's vision tower on SM120.
"""

import functools
from typing import Optional

import torch

from sglang.srt.utils import get_device_capability

# Backends whose forward applies VisionAttention's sinks and window; the others
# would run a sink layer as plain full attention.
SINK_BACKENDS = frozenset({"sdpa", "fa3", "fa4", "xpu_attn"})


def check_sink_backend(backend: str, use_sink: bool) -> None:
    if use_sink and backend not in SINK_BACKENDS:
        raise ValueError(
            f"Multimodal attention backend {backend!r} ignores attention sinks and "
            "local windows, which this vision model uses; pass "
            f"--mm-attention-backend with one of {sorted(SINK_BACKENDS)}"
        )


@functools.cache
def _is_sm12x() -> bool:
    return get_device_capability()[0] == 12


def fa4_pack_gqa(head_dim: int, num_heads: int, num_kv_heads: int) -> Optional[bool]:
    """FA4's pack_gqa argument: None (its own choice) unless packing cannot compile."""
    # Measured on SM120: packed GQA fails to compile (cute.copy pred shape) at head
    # dims 40/56/72/88 and compiles at 48/64/80/96/112; unpacked compiles at all.
    if num_heads != num_kv_heads and head_dim % 16 != 0 and _is_sm12x():
        return False
    return None


def fold_attention_sinks(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> None:
    """Rescale out in place by L / (L + exp(sink)), L = exp(lse), per head and row.

    out: [tokens, heads, head_dim]; lse: [heads, tokens] fp32 natural-log
    sum of exp over each row's keys; sinks: [heads] logits.
    """
    out.mul_(torch.sigmoid(lse - sinks.float()[:, None]).t().unsqueeze(-1))
