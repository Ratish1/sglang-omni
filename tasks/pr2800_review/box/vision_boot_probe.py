"""Whether omni's image encoder returns the same embeddings in two processes, how far they sit from fp32, and its time.

Run on the box from the arm's tree:
    python3 vision_boot_probe.py --prepare --out <dir>                (MMMU-50's images through HF's image processor)
    python3 vision_boot_probe.py --encode <tag> --out <dir>           (one process: every image alone, card 0)
    python3 vision_boot_probe.py --compare <tag> <tag> --out <dir>
    python3 vision_boot_probe.py --distance <reference tag> <tag> [<tag> ...] --out <dir>
--encode honours VISION_PROBE_SDPA=math|flash|efficient|cudnn (forces PyTorch's attention backend),
VISION_PROBE_DETERMINISTIC=1 (torch.use_deterministic_algorithms) and VISION_PROBE_NATIVE=fp32|bf16 (the tree's
Transformers-attention encoder, the one it builds off its platform, in that dtype, in place of the served one);
CUBLAS_WORKSPACE_CONFIG passes through the environment. Each encode writes <dir>/<tag>.pt with every image's pooled
embeddings and deepstack features, then times three more passes over every image (device time, events on card 0).
--distance prints each tag's relative L2 to the reference tag, per image, as median and largest.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import statistics
from pathlib import Path

import torch

MODEL = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
PROCESSOR_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TIMED_PASSES = 3


def prepare(out: Path) -> None:
    from benchmarks.dataset.mmmu import load_mmmu_samples
    from benchmarks.dataset.prepare import DATASETS
    from PIL import Image
    from transformers import Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(PROCESSOR_MODEL)
    inputs = []
    for sample in load_mmmu_samples(repo_id=DATASETS["mmmu-ci-50"]):
        for data_uri in sample.image_data_uris:
            image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).convert("RGB")
            processed = processor.image_processor(images=[image], return_tensors="pt")
            inputs.append((processed["pixel_values"], processed["image_grid_thw"]))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, out / "inputs.pt")
    print(f"{len(inputs)} images prepared")


def encode(out: Path, tag: str) -> None:
    from huggingface_hub import snapshot_download
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from sglang_omni.models.qwen3_omni.components import vision_encoder
    from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
    from sglang_omni.utils.hf import instantiate_module

    if os.environ.get("VISION_PROBE_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        pass
    backends = {
        "math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    forced = os.environ.get("VISION_PROBE_SDPA")
    native = os.environ.get("VISION_PROBE_NATIVE")
    if native:
        vision_encoder.instantiate_visual = lambda vision_config, *, device, native_visual_class: instantiate_module(
            native_visual_class, vision_config
        )
    else:
        pass
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, None: None}[native]
    encoder = Qwen3OmniImageEncoder(snapshot_download(MODEL), device="cuda:0", dtype=dtype)
    attention = encoder.visual.blocks[0].attn
    print(
        f"{tag}: {type(attention).__name__}, packed {type(getattr(attention, 'packed_attention', None)).__name__}, "
        f"attention implementation {encoder.visual.config._attn_implementation}, dtype {encoder.visual.dtype}"
    )
    inputs = torch.load(out / "inputs.pt")
    results = []
    with torch.no_grad(), sdpa_kernel([backends[forced]]) if forced else torch.no_grad():
        for pixel_values, grid_thw in inputs:
            output = encoder(pixel_values=pixel_values, image_grid_thw=grid_thw)
            results.append((
                output["image_embeds"].cpu(),
                [features.cpu() for features in output["deepstack_visual_embeds_image"]],
            ))
        times = []
        for _ in range(TIMED_PASSES):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for pixel_values, grid_thw in inputs:
                encoder(pixel_values=pixel_values, image_grid_thw=grid_thw)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    torch.save(results, out / f"{tag}.pt")
    print(f"{tag}: {len(results)} images encoded; passes {', '.join(f'{time:.1f}' for time in times)} ms")


def compare(out: Path, first_tag: str, second_tag: str) -> None:
    first, second = torch.load(out / f"{first_tag}.pt"), torch.load(out / f"{second_tag}.pt")
    embeds_equal = sum(torch.equal(a[0], b[0]) for a, b in zip(first, second))
    deepstack_equal = sum(all(torch.equal(x, y) for x, y in zip(a[1], b[1])) for a, b in zip(first, second))
    largest = max(float((a[0].float() - b[0].float()).abs().max()) for a, b in zip(first, second))
    print(
        f"{first_tag} vs {second_tag}: {len(first)} images, embeddings equal {embeds_equal}, deepstack equal "
        f"{deepstack_equal}, largest difference {largest:.4g}"
    )


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-30))


def distance(out: Path, reference_tag: str, tags: list[str]) -> None:
    reference = torch.load(out / f"{reference_tag}.pt")
    for tag in tags:
        results = torch.load(out / f"{tag}.pt")
        embeds = [relative_l2(result[0], truth[0]) for result, truth in zip(results, reference)]
        deepstack = [
            max(relative_l2(value, truth) for value, truth in zip(result[1], expected[1]))
            for result, expected in zip(results, reference)
        ]
        print(
            f"{tag} vs {reference_tag}: embeddings median {statistics.median(embeds):.4g} largest {max(embeds):.4g}; "
            f"deepstack median {statistics.median(deepstack):.4g} largest {max(deepstack):.4g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--encode")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--distance", nargs="+")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out)
    elif args.encode:
        encode(args.out, args.encode)
    elif args.compare:
        compare(args.out, *args.compare)
    else:
        distance(args.out, args.distance[0], args.distance[1:])


if __name__ == "__main__":
    main()
