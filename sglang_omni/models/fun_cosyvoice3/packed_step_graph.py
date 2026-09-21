# SPDX-License-Identifier: Apache-2.0
"""One Euler step of the packed CosyVoice3 DiT replayed from breakable CUDA
graphs, one per packed step size.

A step launches about a thousand kernels whatever it computes, so the host
paces it. Every per token op of the step is captured; the row attention and
the conv position embedding stay eager between the graph's segments, because
their shapes follow the rows of the call and not the step size.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left
from collections.abc import Callable

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRowAttention,
    PackedRows,
)

logger = logging.getLogger(__name__)

# note (ratish): the conditional rows and their unconditional twins share a step.
CFG_LANES = 2
STEP_SIZES_PER_DOUBLING = 8


def step_sizes(unit: int, limit: int) -> tuple[int, ...]:
    """Packed step sizes to capture, in frames: multiples of unit whose spacing
    doubles every STEP_SIZES_PER_DOUBLING sizes, then the largest multiple the
    limit holds."""
    top = limit // unit * unit
    sizes: list[int] = []
    size = spacing = unit
    while size < top:
        sizes.append(size)
        if size >= STEP_SIZES_PER_DOUBLING * spacing:
            spacing *= 2
        size += spacing
    return (*sizes, top)


def pad_frames(packed: torch.Tensor, frames: int) -> torch.Tensor:
    """(1, total, channels) -> (1, frames, channels), zeros past total."""
    if packed.shape[1] == frames:
        return packed
    return F.pad(packed, (0, 0, 0, frames - packed.shape[1]))


class StepGraphDiT(PackedDiT):
    """PackedDiT whose step replays at the smallest captured size that holds
    the call's frames; a call past the largest size runs eager. The rows, the
    attention and the rope of the call in flight live on the instance, where
    the eager segments read them: one thread runs the vocoder's Flow calls.
    """

    def __init__(
        self, dit: torch.nn.Module, *, device: str | torch.device, max_frames: int
    ) -> None:
        super().__init__(dit, device=device)
        self.sizes = step_sizes(CFG_LANES * self.chunk_size, max_frames)
        self.graphs = BaseBreakableCudaGraphRunner(self.step, torch.device(device))
        # note (ratish): startup alone decides the captured set; none is evicted.
        self.graphs.max_entries = 0
        # note (ratish): the size the startup warmup is capturing; serving
        # replays and never captures.
        self.capture_size: int | None = None
        self.rows: PackedRows | None = None
        self.attention: PackedRowAttention | None = None
        self.step_rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def capture_sizes(self, run_call: Callable[[], object], *, frames: int) -> None:
        """Captures every size that holds frames, the largest first so the
        smaller graphs fit the pool it sized. run_call is a Flow call of that
        many packed frames through the serving entry point, so the captured
        dtypes and autocast are the serving ones."""
        started = time.monotonic()
        free_bytes = torch.cuda.mem_get_info()[0]
        sizes = [size for size in reversed(self.sizes) if size >= frames]
        for size in sizes:
            self.capture_size = size
            run_call()
        self.capture_size = None
        logger.info(
            "Fun-CosyVoice3 Flow step graphs: %d sizes up to %d frames in %.1f s, "
            "%d MiB",
            len(sizes),
            self.sizes[-1],
            time.monotonic() - started,
            (free_bytes - torch.cuda.mem_get_info()[0]) >> 20,
        )

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PackedRowAttention,
    ) -> torch.Tensor:
        total = x.shape[1]
        self.rows, self.attention = rows, attention
        cos, sin = super().rope(rows)
        index = bisect_left(self.sizes, total)
        if self.capture_size is None and index == len(self.sizes):
            self.step_rope = cos, sin
            return super().forward(x, mu, spks, cond, t, rows, attention)
        frames = self.capture_size or self.sizes[index]
        inputs = {
            "x": pad_frames(x, frames),
            "mu": pad_frames(mu, frames),
            "spks": pad_frames(spks, frames),
            "cond": pad_frames(cond, frames),
            "t": t,
            "cos": pad_frames(cos, frames),
            "sin": pad_frames(sin, frames),
        }
        if self.capture_size is not None:
            # note (ratish): a weight cast that autocast caches dies with its
            # context, so the captured casts must be the graph's own.
            device_type = x.device.type
            with torch.autocast(
                device_type,
                dtype=torch.get_autocast_dtype(device_type),
                cache_enabled=False,
            ):
                self.graphs.capture(**inputs)
        return self.graphs(**inputs)[:, :total]

    def step(
        self,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """x, mu, spks, cond: (1, step frames, channels). A captured step keeps
        only tensor addresses, so the rows and the attention come from self."""
        self.step_rope = cos, sin
        return super().forward(x, mu, spks, cond, t, self.rows, self.attend_rows)

    def rope(self, rows: PackedRows) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step_rope

    # note (ratish): an eager segment replays with the arguments of the
    # capture, so both read this call's rows from self.
    @eager_on_graph(True)
    def attend_rows(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, step frames, heads * head_dim), the call's
        frames first. Returns the same shape."""
        total = self.rows.total
        out = self.attention(query[:, :total], key[:, :total], value[:, :total])
        return pad_frames(out, query.shape[1])

    @eager_on_graph(True)
    def conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        rows = self.rows
        out = PackedDiT.conv_pos_embed(self, h[:, : rows.total], rows)
        return pad_frames(out, h.shape[1])
