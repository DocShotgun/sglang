# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import functools
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        swiglu_limit: SwiGLU clamp applied before SiLU (0.0 disables).
            Required for GLM-5-Next checkpoints whose GPU path clamps
            gate/up to +-10 (config.swiglu_limit).
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None
    swiglu_limit: float = 0.0
    # True when the model's router hands KT logical expert ids even under a
    # non-trivial expert placement, so the wrapper must remap them itself.
    router_emits_logical_ids: bool = False
    # --kt-expert-profile: picks each layer's GPU experts (None: 0..N-1).
    expert_profile: Optional[str] = None


# Routers that never call topk_ids_logical_to_physical. Every other router remaps
# whenever --init-expert-location / EPLB set ep_dispatch_algorithm.
_LOGICAL_ID_ROUTER_ARCHS = frozenset(
    (
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForCausalLMNextN",
        "DeepseekV4ForCausalLMDSpark",
    )
)


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if get_exec().moe.kt_weight_path is None:
        return None

    num_layers = getattr(
        model_config_of(server_args).hf_config, "num_hidden_layers", None
    )
    # GLM-5-Next clamps gate/up inside SwiGLU on the GPU path (see
    # models/glm5_next.py swiglu_clamped); kt must clamp identically or CPU
    # and GPU experts diverge.
    swiglu_limit = float(
        getattr(model_config_of(server_args).hf_config, "swiglu_limit", 0.0) or 0.0
    )
    architectures = model_config_of(server_args).hf_config.architectures or []
    expert_profile = get_exec().moe.kt_expert_profile
    if expert_profile is not None:
        _check_expert_profile_args()
        if not get_exec().moe.kt_num_gpu_experts:
            _warn_profile_unused(expert_profile)

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=get_exec().moe.kt_num_gpu_experts,
        cpuinfer_threads=get_exec().moe.kt_cpuinfer,
        threadpool_count=get_exec().moe.kt_threadpool_count,
        weight_path=get_exec().moe.kt_weight_path,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        method=get_exec().moe.kt_method,
        max_deferred_experts_per_token=get_exec().moe.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        swiglu_limit=swiglu_limit,
        router_emits_logical_ids=bool(_LOGICAL_ID_ROUTER_ARCHS & set(architectures)),
        expert_profile=expert_profile,
    )


def _check_expert_profile_args() -> None:
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
def _warn_profile_unused(path: str) -> None:
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


def _layer_gpu_slots(
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


# Rows per fp32 scatter-add chunk in the compact GPU-route path; arbitrary.
_COMPACT_SCATTER_ROWS = 8192

_routing_dump_prefix = os.environ.get("KT_ROUTING_DUMP")
_routing_state = {}


# Extra histogram rows past num_hidden_layers for MTP / NextN layers; arbitrary.
_ROUTING_DUMP_EXTRA_LAYERS = 8


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


def _routing_count(device, layer_idx, topk_ids, rank, num_layers, num_experts):
    import threading
    import time

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
                        f"{_routing_dump_prefix}-rank{rank}.pt",
                    )
                except Exception as e:  # never take the scheduler down for a dump
                    print(f"[kt-routing] dump failed: {e}", flush=True)

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


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1, or each layer's most-routed experts
      under --kt-expert-profile) using any quantization method
    - CPU experts (all others) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank
        # [num_experts] int32 GPU weight slot of each logical expert, -1 for CPU
        # experts; None means experts 0..num_gpu_experts-1 in slots 0..N-1.
        self.gpu_expert_slots: Optional[torch.Tensor] = None
        self._gpu_slot_table: Optional[torch.Tensor] = None

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # Draft (MTP) layers load without expert-location metadata, so the
        # weight loader's slot remap would not run for them; they keep 0..N-1.
        if (
            self.kt_config.expert_profile is not None
            and layer.uses_expert_location
            and 0 < self.num_gpu_experts < num_experts
        ):
            if self.kt_config.num_layers is None:
                raise ValueError("--kt-expert-profile needs num_hidden_layers in the config")
            self.gpu_expert_slots = _layer_gpu_slots(
                self.kt_config.expert_profile,
                self.kt_config.layer_idx,
                self.kt_config.num_layers,
                num_experts,
                self.num_gpu_experts,
            )

        # 1. Create weights for GPU experts using the wrapped method
        # GPU weight slots 0..num_gpu_experts-1 (see gpu_expert_slot)
        # ModelOpt (NVFP4) methods size their tensors from
        # layer.num_local_experts instead of the num_experts argument, which
        # would allocate every routed expert on the GPU (~4.5 GiB/layer for
        # GLM-5.3-Flash) and OOM during weight creation. Mask the attribute
        # for the duration of the call so every method allocates the GPU
        # slice only; _weight_loader_physical additionally skips checkpoint
        # experts >= num_gpu_experts at load time.
        saved_num_local_experts = layer.num_local_experts
        layer.num_local_experts = self.num_gpu_experts
        try:
            self.gpu_method.create_weights(
                layer=layer,
                num_experts=self.num_gpu_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                **extra_weight_attrs,
            )
        finally:
            layer.num_local_experts = saved_num_local_experts

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: every expert without a GPU slot
        # An all-GPU placement has no CPU weights or CPU work to initialize.
        if self.tp_rank == 0 and self.num_gpu_experts < num_experts:
            # kt-kernel 0.7.0 inference API takes an explicit boolean mask of
            # GPU-resident experts instead of num_gpu_experts (SFT-only).
            if self.gpu_expert_slots is not None:
                gpu_mask = self.gpu_expert_slots >= 0
            else:
                gpu_mask = torch.zeros(num_experts, dtype=torch.bool)
                gpu_mask[: self.num_gpu_experts] = True
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=gpu_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
                swiglu_limit=self.kt_config.swiglu_limit,
                # KT_NUMA_NODES="1" pins a single pool to node 1, the node
                # that does not host the TP0 scheduler, for the prefill-decay
                # isolation; unset keeps kt-kernel's sequential default.
                numa_nodes=(
                    [int(x) for x in os.environ["KT_NUMA_NODES"].split(",")]
                    if os.environ.get("KT_NUMA_NODES")
                    else None
                ),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if (
            hasattr(self.gpu_method, "process_weights_after_loading")
            and self.num_gpu_experts > 0
        ):
            self.gpu_method.process_weights_after_loading(layer)
        # Built here, not on first use, so no host copy lands in graph capture.
        if self.gpu_expert_slots is not None:
            param = next(layer.parameters(), None)
            self._build_gpu_slot_table(
                param.device if param is not None else torch.device("cpu")
            )

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.gpu_method.get_moe_runner_backend(layer)

    def get_moe_routing_method_type(self, layer, requested_method):
        return self.gpu_method.get_moe_routing_method_type(layer, requested_method)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # KT_ROUTING_DUMP=<prefix>: accumulate a [layers, experts] histogram of
        # the routed expert ids on the GPU (a captured index_add_, so it keeps
        # counting under CUDA graph replay) and save it as
        # <prefix>-rank<r>.pt every 30 s. SGLang's expert distribution recorder
        # never sees the V4 router's ids (09-13); this does.
        if _routing_dump_prefix:
            _routing_count(
                x.device,
                self.kt_config.layer_idx,
                topk_ids,
                self.tp_rank,
                num_layers=self.kt_config.num_layers,
                num_experts=self.global_num_experts,
            )

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if TopKOutputChecker.format_is_bypassed(dispatch_output.topk_output):
            dispatch_output = dispatch_output._replace(
                topk_output=dispatch_output.topk_output.to_standard()
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        # --init-expert-location moves experts between the GPU slots (physical
        # 0..num_gpu_experts-1) and the CPU, and both weight loaders honour it.
        # Most routers remap to physical ids themselves, but the V4 router hands
        # this wrapper *logical* ids; only then remap here (see KTConfig).
        log2phy = self._logical_to_physical(x.device)
        if log2phy is not None:
            ids = topk_output.topk_ids
            remapped = torch.where(ids >= 0, log2phy[ids.clamp(min=0)], ids).to(ids.dtype)
            topk_output = topk_output._replace(topk_ids=remapped)
            dispatch_output = dispatch_output._replace(topk_output=topk_output)

        # Step 1: Submit CPU expert computation (non-blocking)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2/3: GPU expert computation on VALID GPU routes only.
        #
        # Feeding marlin -1 routes through its EP-capable specialization
        # garbled NVFP4 on SM120 ("17*23" -> "!ththth..."); its plain all-local
        # path is correct (the configuration Lsglang proves). Both strategies
        # below hand the GPU method only valid local expert ids, so marlin runs
        # its plain path (it never sees a -1 route), and contribute
        # exactly zero for CPU routes through the top-k reduction:
        #   * compact rows (prefill-sized batches): gather only the GPU-route
        #     rows, run marlin with topk=1 per row, scatter-add the weighted
        #     rows back. GPU MoE flops ~= gpu-route share of traffic instead of
        #     all M*topk rows.
        #   * dummy rows (small batches / CUDA-graph decode): remap CPU routes
        #     to expert id 0 with routing weight 0; marlin computes weighted-zero
        #     rows. Static shapes only -- decode graph capture (bs=1) must never
        #     take data-dependent branches, hence the token gate + capture check.
        if self.num_gpu_experts > 0:
            compact = (
                x.shape[0] >= 16  # decode-graph capture runs at bs=1
                and not torch.cuda.is_current_stream_capturing()
            )
            # The GPU method sees GPU weight slots; the CPU above got expert ids.
            topk_ids, gpu_route = self._gpu_routes(topk_output.topk_ids)
            if topk_ids is not topk_output.topk_ids:
                topk_output = topk_output._replace(topk_ids=topk_ids)
            if compact:
                output = self._gpu_output_compact(
                    layer, dispatch_output, topk_output, gpu_route
                )
            else:
                # Weight 0 zeroes a dummy row only if the GPU method applies
                # topk_weights (to its output, or to a bias-free expert's input).
                dummy_topk_ids = torch.where(
                    gpu_route,
                    topk_ids,
                    torch.zeros_like(topk_ids),
                )
                dummy_topk_weights = torch.where(
                    gpu_route,
                    topk_output.topk_weights,
                    torch.zeros_like(topk_output.topk_weights),
                )

                masked_topk_output = topk_output._replace(
                    topk_ids=dummy_topk_ids, topk_weights=dummy_topk_weights
                )
                masked_dispatch_output = dispatch_output._replace(
                    topk_output=masked_topk_output
                )
                gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
                output = gpu_combine_input.hidden_states
        else:
            gpu_combine_input = None
            output = torch.zeros_like(x)

        # Step 4: Synchronize CPU results and merge with GPU results
        if self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def _gpu_output_compact(self, layer, dispatch_output, topk_output, gpu_route):
        """Weighted sum of the GPU-route rows per token, accumulated in fp32."""
        x = dispatch_output.hidden_states
        rows, w_r, tok_idx = self._gpu_rows_compact(
            layer, dispatch_output, topk_output, gpu_route
        )
        if rows is None:
            return torch.zeros_like(x)
        if x.is_cuda:
            from sglang.kernels.ops.moe.kt_compact_route_sum import compact_route_sum

            # A fixed per-token order; the scatter-add below is atomic on CUDA,
            # which makes repeated prefills differ.
            return compact_route_sum(
                rows, w_r.view(-1).float(), gpu_route, out_dtype=x.dtype
            )
        acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # Chunked so the fp32 copy of the rows stays bounded (~200 MiB at 6144).
        for lo in range(0, rows.shape[0], _COMPACT_SCATTER_ROWS):
            hi = lo + _COMPACT_SCATTER_ROWS
            acc.index_add_(0, tok_idx[lo:hi], rows[lo:hi].float() * w_r[lo:hi].float())
        return acc.to(x.dtype)

    def _gpu_rows_compact(self, layer, dispatch_output, topk_output, gpu_route):
        """Run GPU experts on only the GPU-route rows.

        Gathers the rows selected by ``gpu_route`` into a contiguous
        [R, hidden] batch with topk=1 per row, runs the GPU quant method on
        it, and returns (rows, routing_weights, tok_idx) for the caller to
        weight and scatter-add per token. CPU-route rows never reach the GPU.

        The GPU method is called with unit routing weights, so whether its
        combine multiplies by topk_weights (marlin's mul_topk_weights) or not
        is irrelevant and the path works for every quantization method.
        Returns (None, None, None) when no route hits a GPU expert.
        """
        x = dispatch_output.hidden_states
        topk_ids = topk_output.topk_ids
        # The only host sync: the route count sizes every gather below.
        route_idx = gpu_route.reshape(-1).nonzero().squeeze(1)
        if route_idx.numel() == 0:
            return None, None, None
        tok_idx = route_idx // topk_ids.shape[1]

        ids_r = topk_ids.reshape(-1)[route_idx].view(-1, 1)
        w_r = topk_output.topk_weights.reshape(-1)[route_idx].view(-1, 1)
        ones = torch.ones_like(w_r)
        rl = getattr(topk_output, "router_logits", None)
        if rl is not None:
            rows_topk = topk_output._replace(
                topk_ids=ids_r, topk_weights=ones, router_logits=rl[tok_idx]
            )
        else:
            rows_topk = topk_output._replace(topk_ids=ids_r, topk_weights=ones)

        rows_dispatch = dispatch_output._replace(
            hidden_states=x.index_select(0, tok_idx), topk_output=rows_topk
        )
        combine_in = self.gpu_method.apply(layer, rows_dispatch)
        return combine_in.hidden_states, w_r, tok_idx

    def gpu_expert_slot(self, expert_id: int) -> int:
        """GPU weight slot of expert ``expert_id``, or -1 when it runs on the CPU."""
        if self.gpu_expert_slots is None:
            return expert_id if expert_id < self.num_gpu_experts else -1
        return int(self.gpu_expert_slots[expert_id])

    def _gpu_routes(self, topk_ids: torch.Tensor):
        """Return (GPU-method expert ids, GPU-route mask) for ``topk_ids``."""
        if self.gpu_expert_slots is None:
            # Padding uses -1, which is < num_gpu_experts but is not a GPU
            # route; keep it out of both branches.
            return topk_ids, (topk_ids >= 0) & (topk_ids < self.num_gpu_experts)
        table = self._gpu_slot_table
        if table is None or table.device != topk_ids.device:
            table = self._build_gpu_slot_table(topk_ids.device)
        slots = table[topk_ids]
        if slots.dtype != topk_ids.dtype:
            slots = slots.to(topk_ids.dtype)
        return slots, slots >= 0

    def _build_gpu_slot_table(self, device) -> torch.Tensor:
        # One extra -1 entry: padding ids (-1) index it, so they stay off the GPU.
        self._gpu_slot_table = torch.cat(
            [self.gpu_expert_slots, self.gpu_expert_slots.new_full((1,), -1)]
        ).to(device)
        return self._gpu_slot_table

    def _logical_to_physical(self, device):
        cached = self.__dict__.get("_log2phy_cache", "unset")
        if cached != "unset":
            return cached
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        meta = get_global_expert_location_metadata()
        table = None
        # Other routers already remapped; a second remap would misroute.
        if meta is not None and self.kt_config.router_emits_logical_ids:
            m = meta.logical_to_all_physical_map[self.kt_config.layer_idx, :, 0].to(device)
            if not torch.equal(m.cpu(), torch.arange(m.numel(), dtype=m.dtype)):
                table = m.to(torch.int64)
        self.__dict__["_log2phy_cache"] = table
        return table

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
