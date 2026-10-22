"""What the talker's forward actually received at each decode step, from traces of talker_trace/sitecustomize.py.

Run on the box, CPU only: python3 talker_trace_audit.py --trace-dir <dir> [--steps 120:130]
Per request and decode step: the rope position against the one HF's generate uses (the prompt's largest position,
truncated, plus one plus the step), the sequence length against the prompt length plus the step, whether the feedback
buffer row the model reads equals the recorded feedback plus text row and is masked in, and the KV slot. Across
requests, every KV slot two requests held over overlapping forwards (each holds its slots from the forward that wrote
them to its last decode forward). Prints the steps that break any of these and a summary; --steps also prints every
record in that range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROW_TOLERANCE = 1e-2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--steps", help="start:end decode steps to print in full")
    args = parser.parse_args()
    shown = range(*(int(bound) for bound in args.steps.split(":"))) if args.steps else range(0)
    # slot: (request, first forward, last forward) of each request whose KV held it
    owners: dict[int, list[tuple[str, int, int]]] = {}
    for path in sorted(args.trace_dir.glob("*.pt")):
        trace = torch.load(path, weights_only=False)
        if not trace.get("decode_steps") or trace.get("mrope_positions") is None:
            continue
        else:
            pass
        request = trace["request_id"][:8]
        length = len(trace["input_ids"])
        next_position = int(trace["mrope_positions"].long().max()) + 1
        last_forward = trace["decode_steps"][-1]["forward"]
        for slot in trace["prefill_slots"].tolist():
            owners.setdefault(slot, []).append((request, trace["prefill_forward"], last_forward))
        breaks = []
        for step, record in enumerate(trace["decode_steps"]):
            owners.setdefault(record["slot"], []).append((request, record["forward"], last_forward))
            expected_row = trace["feedback_rows"][step] + trace["text_rows"][step]
            row_error = float((record["buffer_row"] - expected_row).norm() / expected_row.norm())
            problems = []
            if set(record["positions"]) != {next_position + step}:
                problems.append(f"positions {record['positions']} expected {next_position + step}")
            else:
                pass
            if record["seq_len"] != length + step + 1:
                problems.append(f"seq_len {record['seq_len']} expected {length + step + 1}")
            else:
                pass
            if not record["mask"] or row_error > ROW_TOLERANCE:
                problems.append(f"mask {record['mask']} buffer row error {row_error:.4f}")
            else:
                pass
            if problems or step in shown:
                breaks.append(step)
                print(f"{request} step {step} bs {record['batch_size']} row {record['row']} slot {record['slot']} "
                      f"positions {record['positions']} seq_len {record['seq_len']} mask {record['mask']} "
                      f"row error {row_error:.4f} {'; '.join(problems)}")
            else:
                pass
        print(f"{request}: prompt {length} rows, next position {next_position}, {len(trace['decode_steps'])} decode "
              f"steps, {len(breaks)} printed")
    collisions = {}
    for slot, holders in owners.items():
        for index, (request, start, end) in enumerate(holders):
            for other, other_start, other_end in holders[index + 1:]:
                if other != request and start <= other_end and other_start <= end:
                    collisions.setdefault(slot, []).append((request, start, end, other, other_start, other_end))
                else:
                    pass
    print(f"KV slots two live requests held at once: {len(collisions)} of {len(owners)}")
    for slot, pairs in list(collisions.items())[:20]:
        print(f"  slot {slot}: {pairs[:3]}")


if __name__ == "__main__":
    main()
