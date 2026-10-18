"""Unit tests for the compact GPU-route path in KTEPWrapperMethod."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _RecordingGpuMethod:
    """Doubles every row and records the weights the wrapper passed in."""

    def __init__(self):
        self.topk_weights = None

    def apply(self, layer, dispatch_output):
        self.topk_weights = dispatch_output.topk_output.topk_weights
        return SimpleNamespace(hidden_states=dispatch_output.hidden_states * 2.0)


def _wrapper(num_gpu_experts=2):
    wrapper = object.__new__(KTEPWrapperMethod)
    wrapper.num_gpu_experts = num_gpu_experts
    wrapper.gpu_method = _RecordingGpuMethod()
    return wrapper


def _dispatch(topk_ids, topk_weights):
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.zeros_like(topk_weights),
    )
    hidden = torch.arange(topk_ids.shape[0] * 4, dtype=torch.float32).view(
        topk_ids.shape[0], 4
    )
    return (
        StandardDispatchOutput(
            hidden_states=hidden, hidden_states_scale=None, topk_output=topk_output
        ),
        topk_output,
    )


def test_compact_skips_padding_and_cpu_routes():
    wrapper = _wrapper()
    topk_ids = torch.tensor([[-1], [0], [1], [3]])
    topk_weights = torch.tensor([[0.5], [0.25], [0.75], [0.1]])
    dispatch, topk_output = _dispatch(topk_ids, topk_weights)
    gpu_route = (topk_ids >= 0) & (topk_ids < wrapper.num_gpu_experts)

    rows, tok_idx = wrapper._gpu_rows_compact(
        None, dispatch, topk_output, gpu_route
    )

    # Only experts 0 and 1 are GPU routes; -1 and 3 are not.
    assert tok_idx.tolist() == [1, 2]
    # The GPU method must see unit weights: the real weight is applied once,
    # in the scatter, regardless of the method's combine semantics.
    assert torch.equal(wrapper.gpu_method.topk_weights, torch.ones_like(topk_weights[:2]))
    expected = (
        dispatch.hidden_states[tok_idx] * 2.0 * topk_weights[1:3]
    )
    torch.testing.assert_close(rows, expected)


def test_compact_returns_none_without_gpu_routes():
    wrapper = _wrapper()
    topk_ids = torch.tensor([[-1], [3]])
    topk_weights = torch.tensor([[0.5], [0.5]])
    dispatch, topk_output = _dispatch(topk_ids, topk_weights)
    gpu_route = (topk_ids >= 0) & (topk_ids < wrapper.num_gpu_experts)

    assert wrapper._gpu_rows_compact(None, dispatch, topk_output, gpu_route) == (
        None,
        None,
    )
