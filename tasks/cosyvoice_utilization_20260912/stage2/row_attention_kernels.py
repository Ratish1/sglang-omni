"""Row attention of a final: FA3 with a page table against FA3 varlen.

A final reads each row whole, one query segment per row whose keys are the row
itself, so the keys of different segments do not overlap and the plain varlen
call can express it without a page table. A hop cannot: its segments are
(row, chunk) pairs whose keys are overlapping prefixes of one row.

Times one attention call per kernel on the shapes a served final has, the rows
doubled for classifier free guidance as the solver doubles them, and reports
how far the two outputs are apart.

  python row_attention_kernels.py
"""

from __future__ import annotations

import json

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    RaggedRowAttention,
    RowAttention,
    pack_rows,
)

HEADS = 16
HEAD_DIM = 64
WARMUP = 10
ITERATIONS = 50

# Mel frames per row: prompt plus the whole generated history, two per token.
SHAPES: dict[str, tuple[int, ...]] = {
    "1 row, 400": (400,),
    "1 row, 1200": (1200,),
    "4 rows, 300 to 900": (300, 500, 700, 900),
    "8 rows, 300 to 1500": (300, 400, 500, 700, 900, 1100, 1300, 1500),
    "16 rows, 300 to 1500": tuple(300 + 80 * index for index in range(16)),
    "16 rows, one of 4000": (4000,) + tuple(300 + 40 * index for index in range(15)),
}


def timed(call) -> float:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERATIONS


def main() -> None:
    device = torch.device("cuda")
    report = []
    for name, lengths in SHAPES.items():
        torch.manual_seed(0)
        rows = pack_rows(lengths * 2, device)
        query, key, value = (
            torch.randn(
                1, rows.total, HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        paged = RaggedRowAttention(
            rows, chunk_size=None, heads=HEADS, head_dim=HEAD_DIM
        )
        padded = RowAttention(rows, chunk_size=None, heads=HEADS)
        starts = rows.starts_host.to(device)

        def varlen() -> torch.Tensor:
            return flash_attn_varlen_func(
                query[0].view(-1, HEADS, HEAD_DIM),
                key[0].view(-1, HEADS, HEAD_DIM),
                value[0].view(-1, HEADS, HEAD_DIM),
                cu_seqlens_q=starts,
                cu_seqlens_k=starts,
                max_seqlen_q=rows.width,
                max_seqlen_k=rows.width,
                causal=False,
            ).reshape(1, -1, HEADS * HEAD_DIM)

        paged_out = paged(query, key, value)
        varlen_out = varlen()
        padded_out = padded(query, key, value)
        report.append(
            {
                "shape": name,
                "total_frames": rows.total,
                "paged_ms": round(timed(lambda: paged(query, key, value)), 3),
                "varlen_ms": round(timed(varlen), 3),
                "padded_sdpa_ms": round(timed(lambda: padded(query, key, value)), 3),
                "paged_build_ms": round(
                    timed(
                        lambda: RaggedRowAttention(
                            rows, chunk_size=None, heads=HEADS, head_dim=HEAD_DIM
                        )
                    ),
                    3,
                ),
                "varlen_equals_paged": bool(torch.equal(varlen_out, paged_out)),
                "max_abs_varlen_minus_paged": float(
                    (varlen_out.float() - paged_out.float()).abs().max()
                ),
                "max_abs_varlen_minus_padded": float(
                    (varlen_out.float() - padded_out.float()).abs().max()
                ),
            }
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "rows": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
