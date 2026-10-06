# SPDX-License-Identifier: Apache-2.0
"""Two probes for the prefix CUDA graph design, one process, real module shapes.

conv: the causal positional conv of a prefix hop run per row on today's padded
(rows, 30 + widest) layout against one packed (1, channels, sum(30 + new)) sequence;
reports whether every row's new frame outputs and both next contexts are bit identical.

fa3: the prefix hop's segmented paged attention with the page table as wide as the
widest segment, as the model's frame ceiling, and in between; reports kernel time and
whether the output is bit identical.

    python prefix_graph_probes.py conv
    python prefix_graph_probes.py fa3
"""

from __future__ import annotations

import argparse
import statistics

import torch
from cosyvoice.flow.DiT.modules import CausalConvPositionEmbedding

from sglang_omni.models.fun_cosyvoice3.packed_dit import ragged_fa3

HIDDEN = 1024
CONTEXT = 30
CHUNK = 50
HEADS, HEAD_DIM = 16, 64
# served row mixes of a causal step: first hops carry the prompt, later hops 100 or 200
ROW_MIXES = {
    "one_hop2": [100],
    "one_first": [350],
    "four_mixed": [350, 200, 100, 200],
    "sixteen_steady": [200] * 16,
    "sixteen_mixed": [
        550,
        200,
        200,
        100,
        300,
        200,
        200,
        200,
        450,
        200,
        100,
        200,
        200,
        250,
        200,
        200,
    ],
}


def conv_probe() -> None:
    torch.manual_seed(0)
    module = CausalConvPositionEmbedding(HIDDEN).cuda().eval()
    with torch.no_grad():
        for conv in (module.conv1[0], module.conv2[0]):
            conv.bias.uniform_(-0.5, 0.5)
    module.to(torch.bfloat16)
    for name, new_frames in ROW_MIXES.items():
        for twin in (False, True):
            rows = new_frames * 2 if twin else new_frames
            hidden = [
                torch.randn(n, HIDDEN, device="cuda", dtype=torch.bfloat16)
                for n in rows
            ]
            first_context = torch.randn(
                len(rows), CONTEXT, HIDDEN, device="cuda", dtype=torch.bfloat16
            )
            second_context = torch.randn_like(first_context)
            committed = [n // CHUNK * CHUNK for n in rows]
            with torch.inference_mode():
                width = max(rows)
                padded = torch.zeros(
                    len(rows), width, HIDDEN, device="cuda", dtype=torch.bfloat16
                )
                for index, frames in enumerate(hidden):
                    padded[index, : frames.shape[0]] = frames
                first_input = torch.cat((first_context, padded), dim=1)
                first_output = module.conv1(first_input.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                second_input = torch.cat((second_context, first_output), dim=1)
                second_output = module.conv2(second_input.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                padded_out = [second_output[index, :n] for index, n in enumerate(rows)]
                padded_tail1 = [
                    first_input[index, c : c + CONTEXT]
                    for index, c in enumerate(committed)
                ]
                padded_tail2 = [
                    second_input[index, c : c + CONTEXT]
                    for index, c in enumerate(committed)
                ]

                extended1 = torch.cat(
                    [torch.cat((first_context[i], hidden[i])) for i in range(len(rows))]
                )
                out1 = module.conv1(extended1.T.unsqueeze(0))[0].T
                starts, position = [], 0
                for n in rows:
                    starts.append(position)
                    position += CONTEXT + n
                new1 = [out1[s : s + n] for s, n in zip(starts, rows)]
                extended2 = torch.cat(
                    [torch.cat((second_context[i], new1[i])) for i in range(len(rows))]
                )
                out2 = module.conv2(extended2.T.unsqueeze(0))[0].T
                packed_out = [out2[s : s + n] for s, n in zip(starts, rows)]
                packed_tail1 = [
                    extended1[s + c : s + c + CONTEXT]
                    for s, c in zip(starts, committed)
                ]
                packed_tail2 = [
                    extended2[s + c : s + c + CONTEXT]
                    for s, c in zip(starts, committed)
                ]
            same = all(torch.equal(a, b) for a, b in zip(padded_out, packed_out))
            same_tails = all(
                torch.equal(a, b)
                for a, b in zip(
                    padded_tail1 + padded_tail2, packed_tail1 + packed_tail2
                )
            )
            worst = max(
                float((a.float() - b.float()).abs().max())
                for a, b in zip(padded_out, packed_out)
            )
            print(
                f"conv {name:15s} twin={int(twin)} rows={len(rows):2d} outputs_equal={same} "
                f"tails_equal={same_tails} max_abs_diff={worst:.3g}",
                flush=True,
            )


def fa3_probe() -> None:
    torch.manual_seed(0)
    pool_pages = 23552
    keys = torch.randn(
        pool_pages, 1, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    for name, new_frames in ROW_MIXES.items():
        prefix = [400 if index % 2 else 0 for index in range(len(new_frames))]
        twin_new, twin_prefix = new_frames * 2, prefix * 2
        segment_rows, segment_ends, offsets = [], [], [0]
        for row, (start, count) in enumerate(zip(twin_prefix, twin_new)):
            end, segment_start = start + count, start
            while segment_start < end:
                segment_end = min((segment_start // CHUNK + 1) * CHUNK, end)
                segment_rows.append(row)
                segment_ends.append(segment_end)
                offsets.append(offsets[-1] + segment_end - segment_start)
                segment_start = segment_end
        row_pages = [
            torch.randperm(pool_pages, device="cuda", dtype=torch.int64)[: s + n].to(
                torch.int32
            )
            for s, n in zip(twin_prefix, twin_new)
        ]
        cache_seqlens = torch.tensor(segment_ends, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        max_seqlen_q = max(b - a for a, b in zip(offsets, offsets[1:]))
        query = torch.randn(
            offsets[-1], HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        outputs = {}
        for width_name, width in (
            ("tight", max(segment_ends)),
            ("8192", 8192),
            ("15000", 15000),
        ):
            table = torch.zeros(
                len(segment_ends), width, dtype=torch.int32, device="cuda"
            )
            for segment, (row, end) in enumerate(zip(segment_rows, segment_ends)):
                table[segment, :end] = row_pages[row][:end]
            times = []
            with torch.inference_mode():
                for _ in range(10):
                    ragged_fa3(
                        query,
                        keys,
                        values,
                        cache_seqlens,
                        table,
                        cu_seqlens_q,
                        max_seqlen_q,
                    )
                for _ in range(50):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    out = ragged_fa3(
                        query,
                        keys,
                        values,
                        cache_seqlens,
                        table,
                        cu_seqlens_q,
                        max_seqlen_q,
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event) * 1e3)
            outputs[width_name] = out
            print(
                f"fa3 {name:15s} segments={len(segment_ends):3d} width={width_name:5s} "
                f"median_us={statistics.median(times):7.1f} "
                f"equal_to_tight={torch.equal(out, outputs['tight'])}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("conv", "fa3"))
    args = parser.parse_args()
    if args.probe == "conv":
        conv_probe()
    else:
        fa3_probe()


if __name__ == "__main__":
    main()
