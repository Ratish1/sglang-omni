"""The served audio encoder on the requests whose audio rows departed in mmsu_layer_parity.py, graphed and eager.

Run on the box from the arm's tree: python3 audio_encoder_probe.py --run /data/pr2800/<layer parity run> --arm fle
Reads the run's parity.jsonl, takes every request whose served audio rows sit more than THRESHOLD relative L2 from HF
fp32 while HF bf16 sits within it, and as many clean requests of the same audio lengths. For each, the request's own
processor features go through Qwen3OmniAudioEncoder (the tree's class, bf16, on card 0) with the layer CUDA graphs and
without them; each output is compared with HF fp32's audio tower output and with the served rows. Prints, per request,
the token count, the attention segments, the graph bucket taken, and the four distances; writes <run>/audio_probe.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
THRESHOLD = 0.3


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
    rows = [json.loads(line) for line in (args.run / "parity.jsonl").read_text().splitlines()]
    measured = [row for row in rows if row["audio_served"] is not None]
    departed = [row for row in measured if row["audio_served"] > THRESHOLD and row["audio_floor"] <= THRESHOLD]
    lengths = {row["audio_rows"] for row in departed}
    clean = [row for row in measured if row["audio_served"] <= THRESHOLD and row["audio_rows"] in lengths]
    chosen = departed + clean[: len(departed)]

    vocab_size = AutoConfig.from_pretrained(MODEL).thinker_config.text_config.vocab_size
    audio_token_id = AutoConfig.from_pretrained(MODEL).thinker_config.audio_token_id
    features = {}
    for path in (args.run / args.arm / "dumps").glob("*.pt"):
        dump = torch.load(path, weights_only=False)
        features[tuple(dump["input_ids"][0].tolist())] = (dump["input_features"], dump["feature_attention_mask"])

    model_path = snapshot_download(MODEL)
    graphed = Qwen3OmniAudioEncoder(model_path, device="cuda:0", dtype=torch.bfloat16, enable_layer_cuda_graph=True)
    eager = Qwen3OmniAudioEncoder(model_path, device="cuda:0", dtype=torch.bfloat16, enable_layer_cuda_graph=False)
    runner = graphed.layer_graph_runner
    print(f"graph window {runner.window if runner else None}, buckets {list(runner.graphs) if runner else None}")

    results = []
    for row in chosen:
        served = torch.load(args.run / args.arm / "layers" / f"{row['rid']}_inputs.pt", weights_only=False)
        truth = torch.load(args.run / "hf_fp32" / f"{row['rid']}.pt", weights_only=False)["audio"]
        prompt_ids = [audio_token_id if token_id >= vocab_size else token_id for token_id in served["origin_input_ids"]]
        input_features, feature_mask = features[tuple(prompt_ids)]
        audio_rows = torch.tensor(
            [index for index, token_id in enumerate(served["origin_input_ids"]) if token_id >= vocab_size]
        )
        served_audio = served["input_embeds"][audio_rows - served["prefix"]]
        with torch.no_grad():
            graphed_audio = graphed(
                input_features=input_features.to("cuda:0"), feature_attention_mask=feature_mask.to("cuda:0")
            )["audio_embeds"].cpu()
            eager_audio = eager(
                input_features=input_features.to("cuda:0"), feature_attention_mask=feature_mask.to("cuda:0")
            )["audio_embeds"].cpu()
        frames = int(feature_mask.sum())
        record = dict(
            rid=row["rid"],
            departed=row in departed,
            frames=frames,
            audio_rows=row["audio_rows"],
            served_to_truth=relative_l2(served_audio, truth),
            graphed_to_truth=relative_l2(graphed_audio, truth),
            eager_to_truth=relative_l2(eager_audio, truth),
            graphed_to_served=relative_l2(graphed_audio, served_audio),
        )
        results.append(record)
        print(
            f"{record['rid'][:8]} departed {record['departed']!s:5} frames {frames:5d} rows {record['audio_rows']:4d} "
            f"served {record['served_to_truth']:.4f} graphed {record['graphed_to_truth']:.4f} "
            f"eager {record['eager_to_truth']:.4f} graphed-vs-served {record['graphed_to_served']:.4f}",
            flush=True,
        )
    (args.run / "audio_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
