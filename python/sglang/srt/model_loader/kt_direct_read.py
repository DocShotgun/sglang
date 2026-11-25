# SPDX-License-Identifier: Apache-2.0
"""Checkpoint shards read with parallel O_DIRECT for KT runs, instead of mmap page
faults on the loading thread. Skips the routed-expert tensors kt-kernel loads for the CPU,
which it starts loading in the background, and fills no page cache."""

import concurrent.futures
from typing import Callable, Generator, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.kt_expert_placement import (
    is_kt_cpu_expert_tensor,
    start_kt_cpu_loads,
)

# A shard is read ahead, while the previous one is consumed, only up to this size, so
# two of DSv4.1's ~100 GB engram shards are never held at once. Arbitrary.
_READ_AHEAD_MAX_BYTES = 16 << 30


def kt_direct_read_weights_iterator(
    hf_weights_files: List[str],
    keep: Optional[Callable[[str], bool]] = None,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """(name, tensor) per shard in name order, for the names keep accepts (all if
    None); small shards are read ahead."""
    from kt_kernel.utils.direct_read import read_tensors, shard_tensors

    start_kt_cpu_loads()

    def selected(path: str):
        return [
            t
            for t in shard_tensors(path)
            if not is_kt_cpu_expert_tensor(t.key) and (keep is None or keep(t.key))
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ahead:
        pending = None
        for i, path in enumerate(hf_weights_files):
            if pending is None:
                tensors = read_tensors(selected(path))
            else:
                tensors = pending.result()
            pending = None
            if i + 1 < len(hf_weights_files):
                following = selected(hf_weights_files[i + 1])
                if sum(t.nbytes for t in following) <= _READ_AHEAD_MAX_BYTES:
                    pending = ahead.submit(read_tensors, following)
            for name in sorted(tensors):
                yield name, tensors[name]
            del tensors
