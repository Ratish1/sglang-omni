# SPDX-License-Identifier: Apache-2.0
"""Where in the final step omni's Flow error sits (acc_flow_steps_c20261011.py: the final, over the whole
history graph, 30.3 dB median against float64 where CosyVoice's own Flow under bf16 autocast is 38.7 and
omni's hops 37.9 to 39.4). Per final step, the new frames split into the hop region already covered by the
previous hops' tokens, the frames of the last hop's lookahead tokens and the last PRE_LOOKAHEAD tokens'
frames (whose pre lookahead runs without right context at the final): SNR against float64 per region for omni
and for CosyVoice's bf16 autocast, and the largest per frame error's position. Then over every final, the mean per frame error of omni and of
CosyVoice's bf16 autocast by the frame's position in its 50 frame chunk (absolute index over the padded prompt
and the generated frames) and by whether the frame is in the sequence's last chunk.

    python3 acc_final_frames_c20261011.py --records <probe dir>
"""

from __future__ import annotations

import argparse
import copy
import glob
import os

import torch

from acc_reference_c20261011 import MODEL, PRE_LOOKAHEAD, flow_mel, snr_db
from sglang_omni.models.fun_cosyvoice3 import stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cosy = CosyVoice3(stages.resolve_checkpoint(MODEL))
    flow32 = cosy.model.flow
    flow64 = copy.deepcopy(flow32).double()
    by_position: dict[tuple[str, int], list[float]] = {}
    for path in sorted(glob.glob(os.path.join(args.records, "voc_*.pt"))):
        record = torch.load(path, weights_only=False)
        tokens = record["tokens"]
        final = next(step for step in record["steps"] if step["is_final"])
        offset = final["token_offset"]
        inputs = (tokens, record["prompt_token"], record["prompt_feat"], record["embedding"], True, "cuda")
        mel64 = flow_mel(flow64, *inputs, torch.float64, False)[:, :, offset * 2 :].cpu()
        mel16 = flow_mel(flow32, *inputs, torch.float32, True)[:, :, offset * 2 :].cpu()
        omni = final["mel"][:, :, offset * 2 :]
        frames = omni.shape[-1]
        tail = PRE_LOOKAHEAD * 2
        regions = {"body": slice(0, frames - tail), "last 3 tokens": slice(frames - tail, frames)}
        cells = []
        for name, region in regions.items():
            cells.append(
                f"{name} omni {snr_db(mel64[:, :, region], omni[:, :, region]):5.1f}"
                f" cosyvoice bf16 {snr_db(mel64[:, :, region], mel16[:, :, region]):5.1f}"
            )
        per_frame = (omni.double() - mel64.double()).abs().mean(dim=1)[0]
        per_frame16 = (mel16.double() - mel64.double()).abs().mean(dim=1)[0]
        start = record["prompt_token"].shape[1] * 2 + offset * 2
        total = record["prompt_token"].shape[1] * 2 + len(tokens) * 2
        last_chunk_start = (total - 1) // 50 * 50
        for index in range(frames):
            absolute = start + index
            where = "last chunk" if absolute >= last_chunk_start else "earlier chunk"
            bucket = absolute % 50 // 10
            by_position.setdefault((where, bucket), []).append((float(per_frame[index]), float(per_frame16[index])))
        worst = int(per_frame.argmax())
        print(
            f"{os.path.basename(path)[4:16]} tokens {len(tokens):3d} offset {offset:3d} frames {frames:3d} | "
            + " | ".join(cells)
            + f" | worst frame {worst} of {frames} (mean abs {float(per_frame[worst]):.3f},"
            f" frame median {float(per_frame.median()):.3f})",
            flush=True,
        )
    for (where, bucket), values in sorted(by_position.items()):
        omni_mean = sum(value[0] for value in values) / len(values)
        cosy_mean = sum(value[1] for value in values) / len(values)
        print(
            f"{where:13s} frames {bucket * 10:2d}-{bucket * 10 + 9:2d} of the chunk: n {len(values):4d}"
            f" omni {omni_mean:.4f} cosyvoice bf16 {cosy_mean:.4f} ratio {omni_mean / cosy_mean:.2f}"
        )


if __name__ == "__main__":
    main()
