"""Engram history rows the scheduler builds for multimodal requests, and the
host tables' NUMA interleave policy. CPU only."""

import ctypes
import mmap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND
from sglang.srt.layers import engram
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 9
# Multimodal pad ids span 62 bits, past what the int32 history holds.
WIDE_PAD = MM_PAD_SHIFT_VALUE + (1 << 61)


def _make_manager():
    """A manager whose trigram hasher has a 16-token vocab, one layer and two
    heads, built without a tokenizer."""
    hasher = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(hasher)
    hasher.max_ngram_size = 3
    hasher.pad_id = 0
    hasher.image_token_id = IMAGE_TOKEN_ID
    hasher.token_map = torch.arange(16)
    hasher.multipliers = torch.tensor([[3, 5, 7]])
    hasher.primes = torch.tensor([[[11, 13], [17, 19]]])
    hasher.offsets = torch.zeros(1, 4, dtype=torch.int64)
    hasher.init_history(num_req_slots=2, device="cpu")
    return NgramEmbeddingManager(enabled=False, table=None, n=0, engram_hasher=hasher)


class TestEngramHistoryMultimodalPads(CustomTestCase):
    def test_extend_history_clamps_pads_and_hashes_them_as_images(self):
        manager = _make_manager()
        hasher = manager.engram_hasher
        # The prefix hit ends right after an image, so both predecessors are pads.
        req = SimpleNamespace(
            full_untruncated_fill_ids=[1, 2, WIDE_PAD, WIDE_PAD, 3, 4]
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, reqs=[req], prefix_lens=[4]
        )
        manager._prepare_engram_history(batch)
        self.assertEqual(batch.engram_history.tolist(), [[MM_PAD_SHIFT_VALUE] * 2])

        def hash_ids(history):
            return hasher._torch_hash_ids(
                torch.tensor([3, 4]),
                torch.tensor([4, 5]),
                MODE_EXTEND,
                history,
                num_real=2,
                block=1,
                row=torch.tensor([0, 0]),
                starts=torch.tensor([0]),
            )[0]

        image_history = torch.full((1, 2), IMAGE_TOKEN_ID, dtype=torch.int32)
        torch.testing.assert_close(
            hash_ids(batch.engram_history), hash_ids(image_history)
        )

    def test_pd_prebuilt_history_clamps_pads(self):
        manager = _make_manager()
        # The prompt ends with an image and prefill sampled token 7.
        req = SimpleNamespace(
            origin_input_ids=[1, WIDE_PAD, WIDE_PAD],
            output_ids=[7],
            kv=SimpleNamespace(req_pool_idx=1),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT, reqs=[req])
        manager._prepare_engram_history(batch)
        self.assertEqual(
            manager.engram_hasher.history[1].tolist(), [MM_PAD_SHIFT_VALUE] * 2
        )


class TestEngramHostTableNumaInterleave(CustomTestCase):
    def _mbind_nodes(self, online):
        """Nodes the kernel would interleave over: mbind reads only maxnode - 1
        bits of the mask (get_nodes() in mm/mempolicy.c)."""
        calls = []

        def fake_mbind(ptr, nbytes, mask, maxnode):
            calls.append(([int(word) for word in mask], maxnode))
            return 0

        with (
            patch.object(engram, "_online_numa_nodes", return_value=online),
            patch.object(engram, "_mbind_interleave", fake_mbind),
        ):
            engram._interleave_numa_pages(0, 4096, "test")
        (words, maxnode), = calls
        self.assertGreaterEqual(len(words) * 64, maxnode - 1)
        return [
            bit for bit in range(maxnode - 1) if words[bit // 64] >> (bit % 64) & 1
        ]

    def test_mask_covers_every_online_node(self):
        """The highest node id must stay inside the bits mbind reads, including
        node 63 on a 64-node host and ids past the first mask word."""
        for online in ([0, 1], [0, 1, 2, 3], list(range(64)), [0, 2, 64]):
            with self.subTest(online=online):
                self.assertEqual(self._mbind_nodes(online), online)

    def test_kernel_interleaves_over_every_online_node(self):
        online = engram._online_numa_nodes()
        if len(online) < 2:
            self.skipTest("needs at least two online NUMA nodes")
        pages = 4 * len(online)
        table = mmap.mmap(-1, pages * mmap.PAGESIZE)
        anchor = ctypes.c_char.from_buffer(table)
        try:
            ptr = ctypes.addressof(anchor)
            engram._interleave_numa_pages(ptr, pages * mmap.PAGESIZE, "test")
            for page in range(pages):
                table[page * mmap.PAGESIZE] = 1
            with open("/proc/self/numa_maps") as fh:
                policy = next(
                    line.split()[1] for line in fh if int(line.split()[0], 16) == ptr
                )
        finally:
            del anchor
            table.close()
        self.assertTrue(policy.startswith("interleave:"), policy)
        self.assertEqual(engram._parse_node_ranges(policy.split(":", 1)[1]), online)


if __name__ == "__main__":
    unittest.main()
