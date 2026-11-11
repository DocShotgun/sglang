# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

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
    )


# Rows per fp32 scatter-add chunk in the compact GPU-route path; arbitrary.
_COMPACT_SCATTER_ROWS = 8192

_routing_dump_prefix = os.environ.get("KT_ROUTING_DUMP")
_routing_state = {}


# Extra histogram rows past num_hidden_layers for MTP / NextN layers; arbitrary.
_ROUTING_DUMP_EXTRA_LAYERS = 8


def _routing_count(device, layer_idx, topk_ids, rank, num_layers, num_experts):
    import threading
    import time

    st = _routing_state.get(device.index)
    if st is None:
        shape = ((num_layers or 0) + _ROUTING_DUMP_EXTRA_LAYERS, num_experts)
        st = _routing_state[device.index] = {
            "prefill": torch.zeros(shape, dtype=torch.int32, device=device),
            "decode": torch.zeros(shape, dtype=torch.int32, device=device),
        }

        def dump():
            while True:
                time.sleep(30)
                try:
                    prefill, decode = st["prefill"].cpu(), st["decode"].cpu()
                    torch.save(
                        {
                            "rank": rank,
                            "logical_count": prefill + decode,
                            "prefill_count": prefill,
                            "decode_count": decode,
                        },
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
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

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

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
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
        # CPU experts: num_gpu_experts to num_experts-1
        # An all-GPU placement has no CPU weights or CPU work to initialize.
        if self.tp_rank == 0 and self.num_gpu_experts < num_experts:
            # kt-kernel 0.7.0 inference API takes an explicit boolean mask of
            # GPU-resident experts instead of num_gpu_experts (SFT-only).
            # GPU experts occupy ordinals 0..num_gpu_experts-1 (see above).
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
        topk_ids = topk_output.topk_ids

        if self.num_gpu_experts > 0:
            compact = (
                x.shape[0] >= 16  # decode-graph capture runs at bs=1
                and not torch.cuda.is_current_stream_capturing()
            )
            # Padding uses -1, which is < num_gpu_experts but is not a GPU
            # route; keep it out of both branches.
            gpu_route = (topk_ids >= 0) & (topk_ids < self.num_gpu_experts)
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
