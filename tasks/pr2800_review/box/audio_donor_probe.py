"""Whose audio a departed request was served: its served rows against every same-length clip's encoder output.

Run on the box from the arm's tree after audio_encoder_probe.py: python3 audio_donor_probe.py --run <run> --arm fle
For every request audio_probe.json marks departed, every request whose processor call had the same feature length
goes through the tree's Qwen3OmniAudioEncoder (bf16, layer graphs, card 0); the departed request's served rows are
compared with each output. Prints the closest clip, its distance, and both requests' places in the processing order
(the processor dump's call index), with the departed request's own encoder output distance for scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def relative_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    return float((value.float() - truth.float()).norm() / truth.float().norm().clamp_min(1e-30))


def main() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", default="fle")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(MODEL).thinker_config
    vocab_size = config.text_config.vocab_size
    probe = json.loads((args.run / "audio_probe.json").read_text())
    departed = [record["rid"] for record in probe if record["departed"]]

    calls = []
    for path in (args.run / args.arm / "dumps").glob("*.pt"):
        dump = torch.load(path, weights_only=False)
        pid, index = path.stem.split("_")
        calls.append(dict(
            order=int(index), prompt=tuple(dump["input_ids"][0].tolist()), features=dump["input_features"],
            mask=dump["feature_attention_mask"], frames=int(dump["feature_attention_mask"].sum()),
        ))
    calls.sort(key=lambda call: call["order"])
    by_prompt = {call["prompt"]: call for call in calls}

    encoder = Qwen3OmniAudioEncoder(
        snapshot_download(MODEL), device="cuda:0", dtype=torch.bfloat16, enable_layer_cuda_graph=True
    )
    outputs: dict[int, torch.Tensor] = {}

    def encode(call: dict) -> torch.Tensor:
        if call["order"] not in outputs:
            with torch.no_grad():
                outputs[call["order"]] = encoder(
                    input_features=call["features"].to("cuda:0"), feature_attention_mask=call["mask"].to("cuda:0")
                )["audio_embeds"].cpu()
        else:
            pass
        return outputs[call["order"]]

    for rid in departed:
        served = torch.load(args.run / args.arm / "layers" / f"{rid}_inputs.pt", weights_only=False)
        prompt = tuple(config.audio_token_id if token_id >= vocab_size else token_id for token_id in served["origin_input_ids"])
        own = by_prompt[prompt]
        audio_rows = torch.tensor([index for index, token_id in enumerate(served["origin_input_ids"]) if token_id >= vocab_size])
        served_audio = served["input_embeds"][audio_rows - served["prefix"]]
        candidates = [call for call in calls if call["frames"] == own["frames"]]
        scored = sorted((relative_l2(served_audio, encode(call)), call["order"]) for call in candidates)
        best_distance, best_order = scored[0]
        print(
            f"{rid[:8]} order {own['order']:5d} frames {own['frames']:5d}: own output {relative_l2(served_audio, encode(own)):.4f}; "
            f"closest clip order {best_order:5d} at {best_distance:.4f} (offset {best_order - own['order']:+d}); "
            f"{len(candidates)} clips of this length",
            flush=True,
        )


if __name__ == "__main__":
    main()
