"""Unit tests for KTEPWrapperMethod: GPU routes, --kt-expert-profile and the routing dump."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class _RecordingGpuMethod:
    """Doubles every row and records the ids and weights the wrapper passed in."""

    def __init__(self):
        self.topk_ids = None
        self.topk_weights = None

    def apply(self, layer, dispatch_output):
        self.topk_ids = dispatch_output.topk_output.topk_ids
        self.topk_weights = dispatch_output.topk_output.topk_weights
        return SimpleNamespace(hidden_states=dispatch_output.hidden_states * 2.0)


def _wrapper(num_gpu_experts=2, gpu_expert_slots=None):
    wrapper = object.__new__(KTEPWrapperMethod)
    wrapper.num_gpu_experts = num_gpu_experts
    wrapper.gpu_expert_slots = gpu_expert_slots
    wrapper._gpu_slot_table = None
    wrapper.routed_output_scale = None
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


def _save(tmp_path, obj, name="profile.pt"):
    path = str(tmp_path / name)
    torch.save(obj, path)
    return path


def test_profile_prefers_decode_counts(tmp_path):
    """Placement ranks decode routes: prefill-ranked experts catch fewer of them."""
    decode = torch.tensor([[0, 5, 1, 0], [3, 0, 0, 2]])
    logical = torch.tensor([[9, 0, 0, 0], [9, 0, 0, 0]])
    path = _save(tmp_path, {"decode_count": decode, "logical_count": logical})

    counts, key = kt_ep_wrapper._load_expert_profile(path, num_layers=2, num_experts=4)

    assert key == "decode_count"
    assert torch.equal(counts, decode.double())


def test_profile_accepts_recorder_and_legacy_dump_shapes(tmp_path):
    # sglang's expert distribution recorder (and old KT): [buffer, layers, experts].
    recorder = torch.arange(2 * 3 * 4).view(2, 3, 4)
    path = _save(tmp_path, {"logical_count": recorder}, "recorder.pt")
    counts, _ = kt_ep_wrapper._load_expert_profile(path, num_layers=3, num_experts=4)
    assert torch.equal(counts, recorder.sum(0).double())

    # Earlier KT_ROUTING_DUMP files carry the spare MTP rows.
    extra = kt_ep_wrapper._ROUTING_DUMP_EXTRA_LAYERS
    legacy = torch.ones(3 + extra, 4, dtype=torch.int32)
    path = _save(tmp_path, {"decode_count": legacy}, "legacy.pt")
    counts, _ = kt_ep_wrapper._load_expert_profile(path, num_layers=3, num_experts=4)
    assert counts.shape == (3, 4)

    path = _save(tmp_path, torch.ones(3, 4), "raw.pt")
    counts, key = kt_ep_wrapper._load_expert_profile(path, num_layers=3, num_experts=4)
    assert key == "tensor" and counts.shape == (3, 4)


@pytest.mark.parametrize(
    "obj, message",
    [
        ({"decode_count": torch.ones(3, 5)}, "shape"),
        ({"decode_count": torch.ones(4, 4)}, "layers"),
        ({"decode_count": -torch.ones(3, 4)}, "negative"),
        ({"rank": 0}, "no decode_count"),
    ],
)
def test_profile_rejects_a_mismatched_file(tmp_path, obj, message):
    path = _save(tmp_path, obj)
    with pytest.raises(ValueError, match=message):
        kt_ep_wrapper._load_expert_profile(path, num_layers=3, num_experts=4)


def test_profile_slots_take_the_most_routed_experts_in_id_order(tmp_path):
    # Layer 0: expert 4 is hottest, experts 0 and 2 tie; ties go to the lower id.
    # Layer 1 has no routes (a dense layer).
    counts = torch.tensor([[5, 1, 5, 0, 9, 0], [0, 0, 0, 0, 0, 0]])
    path = _save(tmp_path, {"decode_count": counts})

    slots = kt_ep_wrapper._expert_profile_slots(path, 2, 6, 2)

    assert slots[0].tolist() == [0, -1, -1, -1, 1, -1]
    assert slots[1].tolist() == [-1] * 6
    assert slots.dtype == torch.int32


def test_profile_slots_ignore_the_default_device(tmp_path):
    """Model construction runs under a CUDA default device; building the slots there
    must not mix freshly made tensors with the CPU profile (a startup crash)."""
    counts = torch.tensor([[5, 1, 5, 0, 9, 0], [0, 3, 0, 4, 0, 0]])
    path = _save(tmp_path, {"decode_count": counts})

    with torch.device("meta"):
        slots = kt_ep_wrapper._layer_gpu_slots(path, 0, 2, 6, 2)

    assert slots.device.type == "cpu"
    assert slots.tolist() == [0, -1, -1, -1, 1, -1]


def test_profile_layer_slots_none_for_experts_0_to_n_and_error_without_routes(tmp_path):
    """A profile that picks experts 0..N-1 must take today's code path unchanged."""
    counts = torch.tensor([[7, 6, 0, 1], [0, 0, 0, 0], [0, 1, 0, 9]])
    path = _save(tmp_path, {"decode_count": counts})

    assert kt_ep_wrapper._layer_gpu_slots(path, 0, 3, 4, 2) is None
    assert kt_ep_wrapper._layer_gpu_slots(path, 2, 3, 4, 2).tolist() == [-1, 0, -1, 1]
    with pytest.raises(ValueError, match="no routes recorded for MoE layer 1"):
        kt_ep_wrapper._layer_gpu_slots(path, 1, 3, 4, 2)


def test_gpu_expert_slot_for_the_weight_loader():
    default = _wrapper()
    assert [default.gpu_expert_slot(e) for e in range(4)] == [0, 1, -1, -1]

    placed = _wrapper(gpu_expert_slots=torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    assert [placed.gpu_expert_slot(e) for e in range(4)] == [-1, 0, -1, 1]


def test_gpu_routes_without_a_profile_are_unchanged():
    wrapper = _wrapper()
    topk_ids = torch.tensor([[-1, 0], [1, 3]], dtype=torch.int32)

    ids, route = wrapper._gpu_routes(topk_ids)

    assert ids is topk_ids
    assert route.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gpu_routes_map_experts_to_slots(device, dtype):
    wrapper = _wrapper(gpu_expert_slots=torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    topk_ids = torch.tensor([[3, 0], [-1, 1]], dtype=dtype, device=device)

    ids, route = wrapper._gpu_routes(topk_ids)

    # Padding (-1) reads the table's trailing -1 entry and stays off the GPU.
    assert ids.tolist() == [[1, -1], [-1, 0]]
    assert route.tolist() == [[True, False], [False, True]]
    assert ids.dtype == dtype and ids.device == topk_ids.device


def _gpu_only_wrapper(gpu_expert_slots):
    wrapper = _wrapper(gpu_expert_slots=gpu_expert_slots)
    wrapper.tp_rank = 1  # no CPU work on this rank
    wrapper.__dict__["_log2phy_cache"] = None
    return wrapper


def test_apply_dummy_rows_send_slots_and_zero_cpu_routes():
    wrapper = _gpu_only_wrapper(torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    topk_ids = torch.tensor([[3, 0], [-1, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.75]])

    wrapper.apply(None, _dispatch(topk_ids, topk_weights)[0])

    # CPU routes and padding go to slot 0 with weight 0.
    assert wrapper.gpu_method.topk_ids.tolist() == [[1, 0], [0, 0]]
    assert wrapper.gpu_method.topk_weights.tolist() == [[0.5, 0.0], [0.0, 0.75]]


def test_apply_compact_rows_send_slots():
    wrapper = _gpu_only_wrapper(torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    # 16 tokens take the compact path; only experts 1 and 3 are GPU routes.
    topk_ids = torch.tensor([[e % 4, (e + 1) % 4] for e in range(16)], dtype=torch.int32)
    topk_weights = torch.full(topk_ids.shape, 0.5)

    wrapper.apply(None, _dispatch(topk_ids, topk_weights)[0])

    slot = {1: 0, 3: 1}
    expected = [[slot[e]] for e in topk_ids.reshape(-1).tolist() if e in slot]
    assert wrapper.gpu_method.topk_ids.tolist() == expected


def test_routing_snapshot_saves_exact_layers_and_extra_rows():
    """Dump files hold num_hidden_layers rows, so profiles match the model exactly."""
    extra = kt_ep_wrapper._ROUTING_DUMP_EXTRA_LAYERS
    st = {
        "prefill": torch.zeros(2 + extra, 3, dtype=torch.int32),
        "decode": torch.zeros(2 + extra, 3, dtype=torch.int32),
        "num_layers": 2,
    }
    st["prefill"][1, 2] = 4
    st["decode"][0, 1] = 7

    snap = kt_ep_wrapper._routing_snapshot(st, rank=0)
    assert snap["decode_count"].tolist() == [[0, 7, 0], [0, 0, 0]]
    assert snap["prefill_count"].tolist() == [[0, 0, 0], [0, 0, 4]]
    assert snap["logical_count"].tolist() == [[0, 7, 0], [0, 0, 4]]
    assert "extra_layer_count" not in snap

    st["decode"][2, 0] = 1  # a draft layer with id num_hidden_layers
    snap = kt_ep_wrapper._routing_snapshot(st, rank=0)
    assert snap["extra_layer_count"][0].tolist() == [1, 0, 0]
    assert snap["decode_count"].shape == (2, 3)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"init_expert_location": "placement.json"}, "--init-expert-location"),
        ({"kt_max_deferred_experts_per_token": 2}, "deferred"),
    ],
)
def test_profile_rejects_conflicting_options(fields, message):
    with get_context().override_server_args(kt_expert_profile="p.pt", **fields):
        with pytest.raises(ValueError, match=message):
            kt_ep_wrapper._check_expert_profile_args()


def test_profile_accepts_the_default_options():
    with get_context().override_server_args(
        kt_expert_profile="p.pt", kt_max_deferred_experts_per_token=0
    ):
        kt_ep_wrapper._check_expert_profile_args()


def test_gpu_runner_gets_no_routed_scale():
    """The CPU experts never apply routed_scaling_factor; a GPU runner handed it would
    weight the GPU experts by it twice once the model scales their sum."""
    wrapper = _wrapper()
    wrapper.override_num_local_experts = False
    seen = {}
    wrapper.gpu_method.create_moe_runner = lambda layer, config: seen.update(
        scale=config.routed_scaling_factor
    )
    wrapper.create_moe_runner(None, MoeRunnerConfig(routed_scaling_factor=2.5))
    assert seen == {"scale": None}


@pytest.mark.parametrize("tp_rank, scale", [(0, 2.5), (1, 2.5), (1, None)])
def test_merge_without_cpu_experts_scales_once(tp_rank, scale, monkeypatch):
    """Ranks that run no CPU experts (tp_rank > 0, or no KT wrapper) still apply the
    routed scale the model handed over, exactly once; None leaves the output as is."""
    wrapper = _wrapper()
    wrapper.tp_rank = tp_rank
    wrapper.wrapper = None
    wrapper.routed_output_scale = scale
    monkeypatch.setattr(KTEPWrapperMethod, "sync", lambda self, x: torch.zeros_like(x))
    gpu_output = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.bfloat16)
    expected = gpu_output.clone() if scale is None else gpu_output * scale
    merged = wrapper._merge_cpu_output(gpu_output.clone(), gpu_output.clone())
    assert torch.equal(merged, expected)


def test_draft_moe_gets_no_kt_config(monkeypatch):
    """A draft's MoE stays on the GPU: kt-kernel would load its CPU experts from the
    target layer with the same id."""
    from sglang.srt.layers.moe.utils import draft_model_build_scope

    hf_config = SimpleNamespace(num_hidden_layers=2, architectures=[], swiglu_limit=0.0)
    monkeypatch.setattr(
        kt_ep_wrapper, "model_config_of", lambda _: SimpleNamespace(hf_config=hf_config)
    )
    with get_context().override_server_args(kt_weight_path="/weights"):
        assert kt_ep_wrapper.create_kt_config_from_server_args(None, 0) is not None
        with draft_model_build_scope():
            assert kt_ep_wrapper.create_kt_config_from_server_args(None, 0) is None
