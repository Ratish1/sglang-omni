# SPDX-License-Identifier: Apache-2.0
"""Where omni's Flow loses to CosyVoice's own Flow under bf16 autocast (acc_reference_c20261011.py: median
37.2 against 38.7 dB, worst step 22 against 27 to 30 dB, both against float64). Per served step of the
accuracy records, by kind (the first hop at offset 0, later hops over the prefix cache, the final over the
whole history graph): omni's and CosyVoice's bf16 autocast SNR against float64, and omni's against
CosyVoice's bf16 autocast.

    python3 acc_flow_steps_c20261011.py --records <probe dir>
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import statistics

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
    kinds: dict[str, list[tuple[float, float, float]]] = {}
    for path in sorted(glob.glob(os.path.join(args.records, "voc_*.pt"))):
        record = torch.load(path, weights_only=False)
        tokens = record["tokens"]
        for step in record["steps"]:
            offset, final = step["token_offset"], step["is_final"]
            end = len(tokens) if final else offset + step["hop_len"] + PRE_LOOKAHEAD
            frames = slice(offset * 2, None)
            args64 = (tokens[:end], record["prompt_token"], record["prompt_feat"], record["embedding"], final, "cuda")
            mel64 = flow_mel(flow64, *args64, torch.float64, False)[:, :, frames]
            mel16 = flow_mel(flow32, *args64, torch.float32, True)[:, :, frames]
            omni = step["mel"][:, :, frames]
            kind = "final" if final else ("first hop" if offset == 0 else "later hop")
            kinds.setdefault(kind, []).append((snr_db(mel64, omni), snr_db(mel64, mel16), snr_db(mel16, omni)))
            print(
                f"{os.path.basename(path)[4:16]} {kind:9s} offset {offset:3d} frames {omni.shape[-1]:4d}"
                f" omni {kinds[kind][-1][0]:5.1f} cosyvoice bf16 {kinds[kind][-1][1]:5.1f}"
                f" omni vs cosyvoice bf16 {kinds[kind][-1][2]:5.1f} dB",
                flush=True,
            )
    for kind, rows in kinds.items():
        omni, cosy16, between = zip(*rows)
        print(
            f"{kind:9s} n {len(rows):3d} | omni vs fp64 median {statistics.median(omni):5.1f} min {min(omni):5.1f}"
            f" | cosyvoice bf16 vs fp64 median {statistics.median(cosy16):5.1f} min {min(cosy16):5.1f}"
            f" | omni vs cosyvoice bf16 median {statistics.median(between):5.1f} min {min(between):5.1f} dB"
        )


if __name__ == "__main__":
    main()
