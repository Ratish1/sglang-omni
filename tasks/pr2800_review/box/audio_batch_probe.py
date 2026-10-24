"""Whether a clip's Qwen3-Omni audio encoder rows depend on the clips batched with it, and at which boundary.

Run on the box from the arm's tree (cwd and PYTHONPATH the tree):
    python3 audio_batch_probe.py --prepare --out <dir>          (the first 64 mmsu-ci-2000 clips through the processor)
    python3 audio_batch_probe.py --encode <tag> --out <dir>     (one process, card 0, the checkpoint's dtype)
    python3 audio_batch_probe.py --compare <tag> [<tag>] --out <dir>
--encode builds Qwen3OmniAudioEncoder as the served stage does (layer CUDA graphs unless AUDIO_PROBE_GRAPHS=0) and
encodes every clip alone, then in consecutive groups of 16 padded as batch_audio_encoder_payloads pads a batch.
AUDIO_PROBE_BATCH_INVARIANT=before turns on SGLang's batch-invariant mode before the encoder is built, so its graphs are
captured with it; =after turns it on once the encoder is built, the served text pipeline's order (the thinker's model
runner turns it on after the encoder stages are built). Each call records every clip's rows entering the layer stack,
leaving it, and leaving the tower, and whether the layer stack replayed a graph. Writes <dir>/<tag>.pt.
--compare with one tag: each clip alone against the same clip batched, at each boundary; with two tags: each call of
the first process against the same call of the second.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPO_ID = "zhaochenyang20/mmsu-ci-2000"
AUDIO_PROMPT = "<|audio_start|><|audio_pad|><|audio_end|>"
CLIP_COUNT = 64
GROUP_SIZE = 16
BOUNDARIES = ("layers_in", "layers_out", "embeds")


def prepare(out: Path) -> None:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from transformers import Qwen3OmniMoeProcessor

    from sglang_omni.preprocessing.audio import load_audio_path

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    inputs = []
    for sample in load_mmsu_samples(repo_id=REPO_ID)[:CLIP_COUNT]:
        array = load_audio_path(sample.audio_path, target_sr=16000)
        processed = processor(text=AUDIO_PROMPT, audio=[array], add_special_tokens=False, return_tensors="pt")
        inputs.append((processed["input_features"], processed["feature_attention_mask"]))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, out / "inputs.pt")
    print(f"{len(inputs)} clips prepared, feature frames {[int(mask.sum()) for _, mask in inputs]}")


def encode(out: Path, tag: str) -> None:
    from huggingface_hub import snapshot_download
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
    from sglang_omni.models.qwen3_omni.stages import pad_audio_features, pad_audio_mask

    batch_invariant = os.environ.get("AUDIO_PROBE_BATCH_INVARIANT", "")
    graphs = os.environ.get("AUDIO_PROBE_GRAPHS", "1") == "1"
    if batch_invariant == "before":
        enable_batch_invariant_mode()
    else:
        pass
    encoder = Qwen3OmniAudioEncoder(
        snapshot_download(MODEL), device="cuda:0", dtype=None, enable_layer_cuda_graph=graphs
    )
    if batch_invariant == "after":
        enable_batch_invariant_mode()
    else:
        pass
    tower = encoder.audio_tower
    runner = encoder.layer_graph_runner
    first_layer, last_layer = (tower.layers, tower.layers) if runner is not None else (tower.layers[0], tower.layers[-1])
    call: dict[str, object] = {}
    first_layer.register_forward_pre_hook(
        lambda module, args, kwargs: call.update(
            layers_in=(args[0] if args else kwargs["hidden_states"]).detach().cpu().clone()
        ),
        with_kwargs=True,
    )
    last_layer.register_forward_hook(
        lambda module, args, output: call.update(
            layers_out=(output[0] if isinstance(output, tuple) else output).detach().cpu().clone()
        )
    )
    if runner is not None:
        replay = runner.maybe_replay

        def recorded_replay(hidden_states, cu_seqlens, segments):
            result = replay(hidden_states, cu_seqlens, segments)
            call.update(replayed=result is not None, bucket=runner.select(hidden_states.shape[0], segments))
            return result

        runner.maybe_replay = recorded_replay
    else:
        pass
    print(f"{tag}: graphs {runner is not None}, batch-invariant {batch_invariant or 'off'}")

    def run(features: torch.Tensor, mask: torch.Tensor) -> list[dict[str, object]]:
        call.clear()
        call.update(replayed=False, bucket=None)
        with torch.no_grad():
            output = encoder(input_features=features, feature_attention_mask=mask, audio_feature_lengths=mask.sum(-1))
        lengths = output["audio_output_lengths"].cpu().tolist()
        rows = {"layers_in": call["layers_in"], "layers_out": call["layers_out"], "embeds": output["audio_embeds"].cpu()}
        clips = []
        start = 0
        for length in lengths:
            clips.append(
                {name: rows[name][start : start + length].clone() for name in BOUNDARIES}
                | {"replayed": call["replayed"], "bucket": call["bucket"], "tokens": sum(lengths)}
            )
            start += length
        return clips

    inputs = torch.load(out / "inputs.pt")
    alone = [run(features, mask)[0] for features, mask in inputs]
    batched = []
    for group_start in range(0, len(inputs), GROUP_SIZE):
        group = inputs[group_start : group_start + GROUP_SIZE]
        max_time = max(int(features.shape[-1]) for features, _ in group)
        batched.extend(
            run(
                torch.cat([pad_audio_features(features, max_time) for features, _ in group]),
                torch.cat([pad_audio_mask(mask.bool(), max_time) for _, mask in group]),
            )
        )
    torch.save({"alone": alone, "batched": batched}, out / f"{tag}.pt")
    print(
        f"{tag}: {len(alone)} clips; alone replayed {sum(clip['replayed'] for clip in alone)}, buckets "
        f"{sorted({clip['bucket'] for clip in alone if clip['bucket'] is not None})}; batched replayed "
        f"{sum(clip['replayed'] for clip in batched)}, buckets "
        f"{sorted({clip['bucket'] for clip in batched if clip['bucket'] is not None})}"
    )


def equal_counts(first: list[dict], second: list[dict]) -> str:
    counts = []
    for name in BOUNDARIES:
        equal = sum(torch.equal(a[name], b[name]) for a, b in zip(first, second))
        largest = max(float((a[name].float() - b[name].float()).abs().max()) for a, b in zip(first, second))
        counts.append(f"{name} {equal}/{len(first)} (largest {largest:.4g})")
    return ", ".join(counts)


def compare(out: Path, tags: list[str]) -> None:
    first = torch.load(out / f"{tags[0]}.pt")
    if len(tags) == 1:
        print(f"{tags[0]} alone vs batched: {equal_counts(first['alone'], first['batched'])}")
    else:
        second = torch.load(out / f"{tags[1]}.pt")
        for call in ("alone", "batched"):
            print(f"{tags[0]} vs {tags[1]}, {call}: {equal_counts(first[call], second[call])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--encode")
    parser.add_argument("--compare", nargs="+")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out)
    elif args.encode:
        encode(args.out, args.encode)
    else:
        compare(args.out, args.compare)


if __name__ == "__main__":
    main()
