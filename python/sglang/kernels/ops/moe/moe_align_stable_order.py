"""Reorder moe_align_block_size's output so each expert's rows ascend by id.

moe_align places an expert's rows in atomic arrival order. This keeps every
expert's run of blocks where moe_align put it and only permutes the rows inside
it, so the result depends on topk_ids alone. Four launches and no host sync,
so it also runs inside captured graphs.
"""

import torch
import triton
import triton.language as tl

_CHUNK = 128


@triton.jit
def _run_of_entry_kernel(
    SortedIds,
    ExpertIds,
    PostPadded,
    EntryRun,
    RunStart,
    numel,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Each placed row records its run's expert, and each run's first block
    records where the run starts."""
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(PostPadded)
    live = slots < total
    ids = tl.load(SortedIds + slots, live, numel)
    experts = tl.load(ExpertIds + slots // BLOCK_SIZE, live, 0)
    real = live & (ids < numel)
    tl.store(EntryRun + ids, experts, real)
    first = live & (slots % BLOCK_SIZE == 0)
    previous = tl.load(
        ExpertIds + slots // BLOCK_SIZE - 1, first & (slots >= BLOCK_SIZE), -2
    )
    tl.store(RunStart + experts + 1, slots, first & (previous != experts))


@triton.jit
def _chunk_histogram_kernel(
    EntryRun, Hist, numel, CHUNK: tl.constexpr, BINS: tl.constexpr
):
    entries = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    runs = tl.load(EntryRun + entries, entries < numel, -1)
    # Rows past numel land in bin 0 and are subtracted.
    counts = tl.histogram(runs + 1, BINS)
    pad = CHUNK - tl.minimum(numel - tl.program_id(0) * CHUNK, CHUNK)
    counts = tl.where(tl.arange(0, BINS) == 0, counts - pad, counts)
    tl.store(Hist + tl.program_id(0) * BINS + tl.arange(0, BINS), counts)


@triton.jit
def _exclusive_scan_kernel(Hist, chunks, BINS: tl.constexpr, TILE: tl.constexpr):
    """Hist[c, b] becomes the count of bin b over chunks before c."""
    bins = tl.program_id(0) * 16 + tl.arange(0, 16)
    carry = tl.zeros((16,), tl.int32)
    for first in range(0, chunks, TILE):
        rows = first + tl.arange(0, TILE)
        mask = rows[:, None] < chunks
        offsets = rows[:, None] * BINS + bins[None, :]
        counts = tl.load(Hist + offsets, mask, 0)
        tl.store(Hist + offsets, tl.cumsum(counts, 0) - counts + carry[None, :], mask)
        carry += tl.sum(counts, 0)


@triton.jit
def _place_kernel(
    EntryRun, Hist, RunStart, SortedIds, numel, CHUNK: tl.constexpr, BINS: tl.constexpr
):
    chunk = tl.program_id(0)
    entries = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = entries < numel
    runs = tl.load(EntryRun + entries, valid, -1)
    # Rows earlier in this chunk with the same run.
    earlier = tl.arange(0, CHUNK)[None, :] < tl.arange(0, CHUNK)[:, None]
    same = (runs[None, :] == runs[:, None]) & earlier
    rank = tl.sum(same.to(tl.int32), 1)
    before = tl.load(Hist + chunk * BINS + runs + 1, valid, 0)
    start = tl.load(RunStart + runs + 1, valid, 0)
    tl.store(SortedIds + start + before + rank, entries, valid)


def order_rows_within_experts(
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size: int,
    numel: int,
    num_experts: int,
) -> torch.Tensor:
    """Sort each expert's rows by id, in place, in the output of
    moe_align_block_size(topk_ids, block_size, num_experts) for a topk_ids of
    ``numel`` entries. Every entry must have been placed (the default +1-bucket
    convention, not ignore_invalid_expert)."""
    if numel == 0:
        return sorted_token_ids
    device = sorted_token_ids.device
    bins = max(16, triton.next_power_of_2(num_experts + 2))
    chunks = triton.cdiv(numel, _CHUNK)
    entry_run = torch.empty(numel, dtype=torch.int32, device=device)
    run_start = torch.empty(bins, dtype=torch.int32, device=device)
    hist = torch.empty((chunks, bins), dtype=torch.int32, device=device)
    slot_block = 1024
    _run_of_entry_kernel[(triton.cdiv(sorted_token_ids.numel(), slot_block),)](
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        entry_run,
        run_start,
        numel,
        BLOCK_SIZE=block_size,
        BLOCK=slot_block,
    )
    _chunk_histogram_kernel[(chunks,)](entry_run, hist, numel, CHUNK=_CHUNK, BINS=bins)
    _exclusive_scan_kernel[(bins // 16,)](hist, chunks, BINS=bins, TILE=256)
    _place_kernel[(chunks,)](
        entry_run, hist, run_start, sorted_token_ids, numel, CHUNK=_CHUNK, BINS=bins
    )
    return sorted_token_ids
