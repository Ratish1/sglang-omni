"""The Qwen3-Omni audio encoder's attention, main against a branch, over mmsu-ci-2000's 2000 clips.

Run on the box from the arm's tree (cwd and PYTHONPATH the tree):
    python3 audio_attention_probe.py --prepare --out <dir>      (every clip through the processor)
    python3 audio_attention_probe.py --reference --out <dir>    (main's tree only: fp32, math attention, TF32 off)
    python3 audio_attention_probe.py --encode <tag> --out <dir>
    python3 audio_attention_probe.py --compare <tag> <tag> --out <dir>
--reference encodes each clip alone with the encoder in fp32 and its layer graphs off, so every layer runs the
Transformers attention under math sdpa; writes reference.pt.
--encode builds Qwen3OmniAudioEncoder as the served stage does (checkpoint dtype, layer CUDA graphs on) and makes four
calls over the clips: "graph", each clip alone; "eager", each clip alone with the runner declining, so the layers run
eagerly; "batch16" and "batch32", consecutive groups of 16 and 32 (the stage's batch limit) padded as
batch_audio_encoder_payloads pads a batch, where a group over the largest bucket declines on its own. Per clip and
call: the relative L2 of its embeddings to the reference, whether the call replayed, and a hash of its rows; per call
kind the device time of each of TIMED_PASSES further sweeps; per clip the largest difference of eager against graph.
Writes <tag>.json.
--compare prints, per call kind, each tag's median and largest distance, replays and times, and the clips whose rows
hash equal across the two tags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPO_ID = "zhaochenyang20/mmsu-ci-2000"
AUDIO_PROMPT = "<|audio_start|><|audio_pad|><|audio_end|>"
CALLS = ("graph", "eager", "batch16", "batch32")
TIMED_PASSES = 2


def prepare(out: Path) -> None:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from transformers import Qwen3OmniMoeProcessor

    from sglang_omni.preprocessing.audio import load_audio_path

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    inputs = []
    for sample in load_mmsu_samples(repo_id=REPO_ID):
        array = load_audio_path(sample.audio_path, target_sr=16000)
        processed = processor(text=AUDIO_PROMPT, audio=[array], add_special_tokens=False, return_tensors="pt")
        inputs.append((processed["input_features"], processed["feature_attention_mask"]))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, out / "inputs.pt")
    frames = [int(mask.sum()) for _, mask in inputs]
    print(f"{len(inputs)} clips prepared, feature frames median {statistics.median(frames)} largest {max(frames)}")


def build_encoder(dtype: torch.dtype | None, graphs: bool):
    from huggingface_hub import snapshot_download

    from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder

    return Qwen3OmniAudioEncoder(
        snapshot_download(MODEL), device="cuda:0", dtype=dtype, enable_layer_cuda_graph=graphs
    )


def split_rows(output: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    lengths = output["audio_output_lengths"].cpu().tolist()
    embeds = output["audio_embeds"].cpu()
    return list(embeds.split(lengths))


def reference(out: Path) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = build_encoder(torch.float32, graphs=False)
    rows = []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for features, mask in torch.load(out / "inputs.pt"):
            rows.append(
                split_rows(encoder(input_features=features, feature_attention_mask=mask,
                                   audio_feature_lengths=mask.sum(-1)))[0].float()
            )
    torch.save(rows, out / "reference.pt")
    print(f"reference: {len(rows)} clips in fp32")


def groups(inputs: list, size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from sglang_omni.models.qwen3_omni.stages import pad_audio_features, pad_audio_mask

    batches = []
    for start in range(0, len(inputs), size):
        group = inputs[start : start + size]
        max_time = max(int(features.shape[-1]) for features, _ in group)
        batches.append((
            torch.cat([pad_audio_features(features, max_time) for features, _ in group]),
            torch.cat([pad_audio_mask(mask.bool(), max_time) for _, mask in group]),
        ))
    return batches


def relative_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    return float((value.float() - truth).norm() / truth.norm().clamp_min(1e-30))


def row_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def encode(out: Path, tag: str) -> None:
    encoder = build_encoder(None, graphs=True)
    runner = encoder.layer_graph_runner
    replays: list[bool] = []
    replay = runner.maybe_replay

    def recorded_replay(hidden_states, cu_seqlens, segments):
        result = replay(hidden_states, cu_seqlens, segments)
        replays.append(result is not None)
        return result

    runner.maybe_replay = recorded_replay
    inputs = torch.load(out / "inputs.pt")
    truth = torch.load(out / "reference.pt")
    calls = {
        "graph": [(features, mask, False) for features, mask in inputs],
        "eager": [(features, mask, True) for features, mask in inputs],
        "batch16": [(features, mask, False) for features, mask in groups(inputs, 16)],
        "batch32": [(features, mask, False) for features, mask in groups(inputs, 32)],
    }

    def run(features: torch.Tensor, mask: torch.Tensor, eager: bool) -> dict[str, torch.Tensor]:
        runner.disabled_reason = "probe eager" if eager else None
        try:
            return encoder(input_features=features, feature_attention_mask=mask, audio_feature_lengths=mask.sum(-1))
        finally:
            runner.disabled_reason = None

    record: dict[str, dict] = {}
    graph_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for call, arguments in calls.items():
            replays.clear()
            rows = [clip for features, mask, eager in arguments for clip in split_rows(run(features, mask, eager))]
            replayed = list(replays)
            times = []
            for _ in range(TIMED_PASSES):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for features, mask, eager in arguments:
                    run(features, mask, eager)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            record[call] = {
                "distance": [relative_l2(clip, expected) for clip, expected in zip(rows, truth)],
                "hash": [row_hash(clip) for clip in rows],
                "replayed_calls": sum(replayed),
                "calls": len(arguments),
                "times_ms": times,
            }
            if call == "graph":
                graph_rows = rows
            elif call == "eager":
                record["eager_vs_graph"] = [
                    float((clip.float() - graph.float()).abs().max()) for clip, graph in zip(rows, graph_rows)
                ]
            else:
                pass
            print(f"{tag} {call}: {len(arguments)} calls, replayed {sum(replayed)}, passes "
                  f"{', '.join(f'{time:.1f}' for time in times)} ms", flush=True)
    (out / f"{tag}.json").write_text(json.dumps(record))


def compare(out: Path, tags: list[str]) -> None:
    records = {tag: json.loads((out / f"{tag}.json").read_text()) for tag in tags}
    for call in CALLS:
        for tag, record in records.items():
            distances = record[call]["distance"]
            print(f"{call} {tag}: relative L2 median {statistics.median(distances):.4f} largest {max(distances):.4f}, "
                  f"replayed {record[call]['replayed_calls']} of {record[call]['calls']} calls, passes "
                  f"{', '.join(f'{time:.1f}' for time in record[call]['times_ms'])} ms")
        first, second = (records[tag][call]["hash"] for tag in tags)
        print(f"{call}: rows equal across {tags[0]} and {tags[1]} on {sum(a == b for a, b in zip(first, second))} "
              f"of {len(first)} clips")
    for tag, record in records.items():
        differences = record["eager_vs_graph"]
        print(f"eager vs graph {tag}: equal on {sum(value == 0 for value in differences)} of {len(differences)} "
              f"clips, largest difference {max(differences):.4g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--encode")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out)
    elif args.reference:
        reference(args.out)
    elif args.encode:
        encode(args.out, args.encode)
    else:
        compare(args.out, args.compare)


if __name__ == "__main__":
    main()
