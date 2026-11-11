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

    rows, route_weights, tok_idx = wrapper._gpu_rows_compact(
        None, dispatch, topk_output, gpu_route
    )

    # Only experts 0 and 1 are GPU routes; -1 and 3 are not.
    assert tok_idx.tolist() == [1, 2]
    # The GPU method must see unit weights: the real weight is applied once,
    # in the scatter, regardless of the method's combine semantics.
    assert torch.equal(wrapper.gpu_method.topk_weights, torch.ones_like(topk_weights[:2]))
    torch.testing.assert_close(rows, dispatch.hidden_states[tok_idx] * 2.0)
    torch.testing.assert_close(route_weights, topk_weights[1:3])


def test_compact_returns_none_without_gpu_routes():
    wrapper = _wrapper()
    topk_ids = torch.tensor([[-1], [3]])
    topk_weights = torch.tensor([[0.5], [0.5]])
    dispatch, topk_output = _dispatch(topk_ids, topk_weights)
    gpu_route = (topk_ids >= 0) & (topk_ids < wrapper.num_gpu_experts)

    assert wrapper._gpu_rows_compact(None, dispatch, topk_output, gpu_route) == (
        None,
        None,
        None,
    )


def test_compact_top_k_sum_accumulates_in_fp32():
    """Routes of one token must be weighted and summed in fp32 and rounded to the
    output dtype once: weighting in bf16 misrounds (3.0 * 0.3 gives 0.9023
    instead of bf16(0.9) = 0.8984), and a bf16 running sum drops small terms
    (256 + 1 + 1 stays 256 under CUDA bf16 atomics)."""
    wrapper = _wrapper(num_gpu_experts=3)
    topk_ids = torch.tensor([[0, 1, 2], [0, 5, 5]])
    topk_weights = torch.tensor([[1.0, 1.0 / 256, 1.0 / 256], [0.3, 0.5, 0.5]])
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.zeros_like(topk_weights),
    )
    hidden = torch.tensor([[128.0] * 4, [1.5] * 4], dtype=torch.bfloat16)
    dispatch = StandardDispatchOutput(
        hidden_states=hidden, hidden_states_scale=None, topk_output=topk_output
    )
    gpu_route = topk_ids < wrapper.num_gpu_experts

    output = wrapper._gpu_output_compact(None, dispatch, topk_output, gpu_route)

    assert output.dtype == torch.bfloat16
    exact_token1 = (torch.tensor(3.0) * torch.tensor(0.3)).to(torch.bfloat16).item()
    assert output.tolist() == [[258.0] * 4, [exact_token1] * 4]


def test_routing_dump_ignores_padded_routes(monkeypatch):
    """KT_ROUTING_DUMP must count padded (-1) routes as nothing; a raw -1 index
    reaching index_add_ is a device-side assert on CUDA and kills the server."""
    import threading

    from sglang.srt.layers.moe import kt_ep_wrapper

    # The dump thread would write a file 30 s later; counting is what is tested.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(kt_ep_wrapper, "_routing_state", {})
    topk_ids = torch.tensor([[1, -1], [2, 1], [-1, -1]])

    kt_ep_wrapper._routing_count(
        torch.device("cpu"), 0, topk_ids, rank=0, num_layers=2, num_experts=4
    )

    assert kt_ep_wrapper._routing_state[None]["prefill"][0].tolist() == [0, 2, 1, 0]
    assert kt_ep_wrapper._routing_state[None]["decode"][0].tolist() == [0, 0, 0, 0]


def test_routing_dump_counts_single_row_calls_as_decode(monkeypatch):
    import threading

    from sglang.srt.layers.moe import kt_ep_wrapper

    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(kt_ep_wrapper, "_routing_state", {})
    for ids in ([[1, 3]], [[2, 1], [1, 0]]):
        kt_ep_wrapper._routing_count(
            torch.device("cpu"),
            1,
            torch.tensor(ids),
            rank=0,
            num_layers=2,
            num_experts=4,
        )

    state = kt_ep_wrapper._routing_state[None]
    assert state["decode"][1].tolist() == [0, 1, 0, 1]
    assert state["prefill"][1].tolist() == [1, 2, 1, 0]
