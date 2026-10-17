# SPDX-License-Identifier: Apache-2.0
"""Per-stream codec decoder state held by slot."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CodecStateSpec:
    """Static description of one stream's incremental codec state.

    conv_histories and transconv_overlaps are (key, channels, length) triples in the order the
    decoder visits them. Zero-length entries are kept so every consumer sees the same key set
    regardless of kernel size.
    """

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        elements = sum(channels * length for _, channels, length in self.conv_histories)
        elements += sum(
            channels * length for _, channels, length in self.transconv_overlaps
        )
        elements += (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.retained_context
            * self.head_dim
        )
        return int(elements * itemsize)
