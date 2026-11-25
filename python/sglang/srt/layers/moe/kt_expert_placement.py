# SPDX-License-Identifier: Apache-2.0
"""KT expert placement: --kt-expert-profile's per-layer GPU experts, the
KT_ROUTING_DUMP histogram that records such profiles, and which checkpoint tensors
kt-kernel loads for the CPU."""

import concurrent.futures
import functools
import logging
import os
import re
import threading
import time
from typing import Optional

import torch

from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)

# KT_ROUTING_DUMP=<prefix>: the routing histogram's file prefix (unset: no dump).
ROUTING_DUMP_PREFIX = os.environ.get("KT_ROUTING_DUMP")
# Extra histogram rows past num_hidden_layers for MTP / NextN layers; arbitrary.
_ROUTING_DUMP_EXTRA_LAYERS = 8
_routing_state = {}

# Layers whose CPU experts kt-kernel loads, by layer index: [KTEPWrapperMethod, its CPU
# logical experts once computed].
_kt_layers = {}
# Background CPU-expert loads by layer index, from start_kt_cpu_loads.
_cpu_loads = {}
_cpu_loads_started = False
# "...layers.<layer>.[mlp.|ffn.|block_sparse_moe.]experts.<expert>.", as kt-kernel's
# naming variants spell routed-expert tensors.
_ROUTED_EXPERT_TENSOR = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:mlp\.|ffn\.|block_sparse_moe\.)?experts\.(\d+)\."
)


def register_kt_layer(layer_idx: int, method) -> None:
    """Record a layer whose CPU experts kt-kernel loads (method: its KTEPWrapperMethod)."""
    _kt_layers[layer_idx] = [method, None]


def kt_layers_registered() -> bool:
    return bool(_kt_layers)


def is_kt_cpu_expert_tensor(name: str) -> bool:
    """Whether a checkpoint tensor belongs to a routed expert that kt-kernel loads for the
    CPU, which the GPU weight loader skips."""
    match = _ROUTED_EXPERT_TENSOR.search(name)
    entry = _kt_layers.get(int(match.group(1))) if match else None
    if entry is None:
        return False
    if entry[1] is None:
        entry[1] = _cpu_logical_experts(layer_idx=int(match.group(1)), method=entry[0])
    return int(match.group(2)) in entry[1]


def start_kt_cpu_loads() -> None:
    """Load every registered layer's CPU experts on one background thread, in layer
    order, so they overlap the GPU weights' load. Once per process."""
    global _cpu_loads_started
    if _cpu_loads_started or not _kt_layers:
        return
    _cpu_loads_started = True
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="kt_cpu_load"
    )
    for layer_idx in sorted(_kt_layers):
        _cpu_loads[layer_idx] = pool.submit(_kt_layers[layer_idx][0].load_cpu_experts)
    pool.shutdown(wait=False)


def wait_kt_cpu_load(layer_idx: int, load) -> None:
    """Wait for this layer's background load, or call load() if none was started."""
    future = _cpu_loads.pop(layer_idx, None)
    if future is None:
        load()
    else:
        future.result()


def _cpu_logical_experts(layer_idx: int, method) -> frozenset:
    """Logical experts with no GPU replica, as kt-kernel's load_weights selects them."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    num_physical = method.global_num_experts
    if metadata is None:
        physical_to_logical = list(range(num_physical))
    else:
        physical_to_logical = metadata.physical_to_logical_map_cpu[layer_idx].tolist()
    on_gpu = {
        physical_to_logical[p]
        for p in range(num_physical)
        if method.gpu_expert_slot(p) >= 0
    }
    return frozenset(set(physical_to_logical) - on_gpu)


def check_expert_profile_args() -> None:
    moe = get_exec().moe
    # The profile places experts with logical ids on one rank; an expert-location
    # mapping would permute the ids it ranks, and deferral is unverified with it.
    if moe.init_expert_location != "trivial" or moe.enable_eplb:
        raise ValueError(
            "--kt-expert-profile cannot be combined with --init-expert-location or "
            "--enable-eplb"
        )
    if moe.ep_num_redundant_experts or get_parallel().moe_ep_size > 1:
        raise ValueError(
            "--kt-expert-profile needs expert parallel size 1 and no redundant experts"
        )
    if (moe.kt_max_deferred_experts_per_token or 0) > 0:
        raise ValueError(
            "--kt-expert-profile does not support --kt-max-deferred-experts-per-token > 0"
        )


@functools.lru_cache(maxsize=None)
def warn_profile_unused(path: str) -> None:
    logger.warning("--kt-expert-profile %s is unused: --kt-num-gpu-experts is 0", path)


def _load_expert_profile(path: str, num_layers: int, num_experts: int):
    """Return ([num_layers, num_experts] float64 routing counts, the key they came from)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        key = next((k for k in ("decode_count", "logical_count") if k in data), None)
        if key is None:
            raise ValueError(f"{path}: no decode_count or logical_count entry")
        counts = data[key]
    else:
        key, counts = "tensor", data
    if not isinstance(counts, torch.Tensor):
        raise ValueError(f"{path}: {key} is not a tensor")
    # [buffer, layers, experts], as sglang's expert distribution recorder saves it.
    if counts.dim() == 3:
        counts = counts.sum(dim=0)
    if counts.dim() != 2 or counts.shape[1] != num_experts:
        raise ValueError(
            f"{path}: {key} has shape {tuple(counts.shape)}, expected "
            f"[{num_layers}, {num_experts}]"
        )
    # Dumps saved before the histogram was trimmed keep its spare MTP rows.
    if counts.shape[0] == num_layers + _ROUTING_DUMP_EXTRA_LAYERS:
        counts = counts[:num_layers]
    if counts.shape[0] != num_layers:
        raise ValueError(
            f"{path}: {key} has {counts.shape[0]} layers, the model has {num_layers}"
        )
    counts = counts.to(torch.float64)
    if bool((counts < 0).any()):
        raise ValueError(f"{path}: {key} has negative counts")
    return counts, key


@functools.lru_cache(maxsize=None)
def _expert_profile_slots(
    path: str, num_layers: int, num_experts: int, num_gpu_experts: int
) -> torch.Tensor:
    """[num_layers, num_experts] int32 GPU slot of each expert, -1 for CPU experts.

    Each layer keeps its num_gpu_experts most-routed experts (ties to the lower
    id) in slots ordered by expert id; layers without routes get -1 rows.
    """
    counts, key = _load_expert_profile(path, num_layers, num_experts)
    top = torch.sort(counts, dim=1, descending=True, stable=True).indices
    # Explicit CPU: model construction runs under a CUDA default device.
    gpu = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    gpu.scatter_(1, top[:, :num_gpu_experts], True)
    routed = counts.sum(dim=1) > 0
    gpu &= routed.unsqueeze(1)
    slots = torch.where(gpu, gpu.cumsum(dim=1, dtype=torch.int32) - 1, -1).to(
        torch.int32
    )
    total = counts[routed].sum(dim=1)
    placed = (counts * gpu)[routed].sum(dim=1) / total
    today = counts[routed, :num_gpu_experts].sum(dim=1) / total
    logger.info(
        "KT expert profile %s (%s, %d routed layers): its GPU experts take %.1f%% of "
        "the profile's routes per layer on average, against %.1f%% for experts 0..%d",
        path,
        key,
        int(routed.sum()),
        100 * placed.mean().item() if total.numel() else 0.0,
        100 * today.mean().item() if total.numel() else 0.0,
        num_gpu_experts - 1,
    )
    return slots


def layer_gpu_slots(
    path: str, layer_idx: int, num_layers: int, num_experts: int, num_gpu_experts: int
) -> Optional[torch.Tensor]:
    """This layer's [num_experts] GPU slots, or None when they are experts 0..N-1."""
    slots = _expert_profile_slots(path, num_layers, num_experts, num_gpu_experts)
    row = slots[layer_idx]
    if bool((row < 0).all()):
        raise ValueError(f"{path}: no routes recorded for MoE layer {layer_idx}")
    identity = torch.arange(num_gpu_experts, dtype=row.dtype, device=row.device)
    if torch.equal(row[:num_gpu_experts], identity):
        return None
    return row.clone()


def _routing_snapshot(st, rank):
    """The dump file: [num_hidden_layers, experts] counts, plus MTP rows if any."""
    prefill, decode = st["prefill"].cpu(), st["decode"].cpu()
    n = st["num_layers"] or prefill.shape[0]
    snap = {
        "rank": rank,
        "logical_count": prefill[:n] + decode[:n],
        "prefill_count": prefill[:n],
        "decode_count": decode[:n],
    }
    extra = prefill[n:] + decode[n:]
    if bool(extra.any()):
        # Rows for layer ids num_hidden_layers.. (MTP / NextN draft layers).
        snap["extra_layer_count"] = extra
    return snap


def count_routes(device, layer_idx, topk_ids, rank, num_layers, num_experts):
    st = _routing_state.get(device.index)
    if st is None:
        shape = ((num_layers or 0) + _ROUTING_DUMP_EXTRA_LAYERS, num_experts)
        st = _routing_state[device.index] = {
            "prefill": torch.zeros(shape, dtype=torch.int32, device=device),
            "decode": torch.zeros(shape, dtype=torch.int32, device=device),
            "num_layers": num_layers,
        }

        def dump():
            while True:
                time.sleep(30)
                try:
                    torch.save(
                        _routing_snapshot(st, rank),
                        f"{ROUTING_DUMP_PREFIX}-rank{rank}.pt",
                    )
                except Exception as e:  # never take the scheduler down for a dump
                    logger.warning("KT_ROUTING_DUMP: saving failed: %s", e)

        threading.Thread(target=dump, daemon=True, name="kt-routing-dump").start()
    if layer_idx >= st["prefill"].shape[0]:
        raise ValueError(
            f"KT_ROUTING_DUMP: layer {layer_idx} exceeds the "
            f"{st['prefill'].shape[0]}-row routing histogram"
        )
    # One-row calls are bs=1 decode steps (and rare one-token extends);
    # batched decode lands in the prefill histogram.
    counts = st["decode" if topk_ids.shape[0] == 1 else "prefill"]
    ids = topk_ids.reshape(-1).to(torch.int64)
    # Padded rows route to -1; count them as zero at a valid index so the
    # shapes stay static under CUDA graph capture.
    counts[layer_idx].index_add_(0, ids.clamp(min=0), (ids >= 0).to(torch.int32))
