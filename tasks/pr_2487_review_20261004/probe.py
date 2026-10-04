"""Exercise the PR implementation against the checkpoint processor on CUDA."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from PIL import Image
from PIL import __version__ as pillow_version
from transformers import AutoProcessor

from sglang_omni.models.minicpm_o.components.image_processing import process_images


def image_fixture(width: int, height: int, pattern: str) -> Image.Image:
    coordinates_y, coordinates_x = np.indices((height, width))
    if pattern == "noise":
        pixels = np.random.default_rng(2487).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
    elif pattern == "edges":
        pixels = np.stack(
            [
                (coordinates_x % 7 < 3) * 255,
                (coordinates_y % 11 < 5) * 255,
                ((coordinates_x + coordinates_y) % 13 < 6) * 255,
            ],
            axis=-1,
        ).astype(np.uint8)
    else:
        pixels = np.stack(
            [
                coordinates_x % 256,
                coordinates_y % 256,
                (coordinates_x + coordinates_y) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint = snapshot_download(
        arguments.model,
        revision="503e754207c94da6bb26850b4469f367c9ea3582",
        allow_patterns=["*.py", "*.json", "*.txt", "*.jinja", "*.model"],
    )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    geometry = processor.image_processor
    reference = processor.process_image
    report = {
        "checkpoint": checkpoint,
        "torch": torch.__version__,
        "pillow": pillow_version,
        "gpu": torch.cuda.get_device_name(),
        "threads": torch.get_num_threads(),
        "geometry": {
            name: getattr(geometry, name)
            for name in [
                "max_slice_nums",
                "scale_resolution",
                "patch_size",
                "slice_mode",
            ]
        },
    }
    fixture = image_fixture(641, 479, "noise")
    processor.process_image = partial(process_images, geometry)
    try:
        processor(
            "<image>./</image> Describe this.", images=[[fixture]], return_tensors="pt"
        )
    except Exception as error:
        report["production_call"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    else:
        report["production_call"] = "passed"
    processor.process_image = reference
    baseline = processor(
        "<image>./</image> Describe this.", images=[[fixture]], return_tensors="pt"
    )
    report["baseline_call"] = {
        "slices": len(baseline["pixel_values"][0]),
        "bounds": len(baseline["image_bound"][0]),
    }
    print(json.dumps(report), flush=True)

    measurements = []
    for width, height, pattern, slice_limit in [
        (28, 42, "ramp", 1),
        (448, 448, "ramp", 1),
        (641, 479, "noise", 9),
        (641, 479, "edges", 9),
        (479, 641, "noise", 9),
        (1920, 1080, "noise", 9),
        (1920, 1080, "edges", 1),
        (4096, 128, "ramp", 9),
        (3840, 2160, "noise", 9),
        (8000, 6000, "ramp", 9),
    ]:
        image = image_fixture(width, height, pattern)
        expected = reference([[image]], max_slice_nums=slice_limit, return_tensors="pt")
        torch.cuda.reset_peak_memory_stats()
        before_bytes = torch.cuda.memory_allocated()
        actual = process_images(geometry, [[image]], max_slice_nums=slice_limit)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated() - before_bytes
        torch.testing.assert_close(
            actual["tgt_sizes"][0].long(), expected["tgt_sizes"][0].long()
        )
        errors = torch.cat(
            [
                (left.cpu() - right).flatten()
                for left, right in zip(
                    actual["pixel_values"][0], expected["pixel_values"][0], strict=True
                )
            ]
        )
        timings = {"cpu": [], "cpu_plus_h2d": [], "gpu": []}
        for iteration in range(12):
            for implementation in (
                list(timings) if iteration % 2 else list(timings)[::-1]
            ):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                if implementation == "gpu":
                    output = process_images(
                        geometry, [[image]], max_slice_nums=slice_limit
                    )
                else:
                    output = reference(
                        [[image]], max_slice_nums=slice_limit, return_tensors="pt"
                    )
                    if implementation == "cpu_plus_h2d":
                        output["pixel_values"] = [
                            [value.cuda() for value in output["pixel_values"][0]]
                        ]
                    else:
                        pass
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                if iteration >= 2:
                    timings[implementation].append(elapsed)
                else:
                    pass
                del output
        measurement = {
            "size": [width, height],
            "pattern": pattern,
            "slice_limit": slice_limit,
            "slices": len(actual["pixel_values"][0]),
            "peak_mib": peak_bytes / 2**20,
            "max_error": errors.abs().max().item(),
            "mean_error": errors.abs().mean().item(),
            "changed_fraction": (errors != 0).float().mean().item(),
            "latency_ms": {
                name: statistics.median(values) for name, values in timings.items()
            },
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    report["measurements"] = measurements

    geometry.slice_mode = False
    packing_image = image_fixture(42, 70, "noise")
    expected = reference([[packing_image]], max_slice_nums=1, return_tensors="pt")
    actual = process_images(geometry, [[packing_image]], max_slice_nums=1)
    torch.testing.assert_close(
        actual["pixel_values"][0][0].cpu(),
        expected["pixel_values"][0][0],
        rtol=0,
        atol=2e-7,
    )
    report["packing_normalization_atol"] = 2e-7
    unaligned = image_fixture(43, 71, "noise")
    report["unaligned_baseline_shape"] = list(
        reference([[unaligned]])["pixel_values"][0][0].shape
    )
    try:
        process_images(geometry, [[unaligned]])
    except RuntimeError as error:
        report["unaligned_pr_error"] = str(error)
    else:
        report["unaligned_pr_error"] = None
    geometry.slice_mode = True

    pixels = (
        torch.from_numpy(np.array(fixture)).permute(2, 0, 1).unsqueeze(0).cuda().float()
    )
    width, height = geometry.find_best_resize(fixture.size, 448, 14)
    pil = (
        torch.from_numpy(
            np.array(fixture.resize((width, height), Image.Resampling.BICUBIC))
        )
        .permute(2, 0, 1)
        .cuda()
    )
    full = (
        functional.interpolate(
            pixels,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)[0]
    )
    horizontal = (
        functional.interpolate(
            pixels,
            size=(pixels.shape[2], width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
    )
    separable = (
        functional.interpolate(
            horizontal,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)[0]
    )
    report["resize_bisection"] = {
        name: {
            "max_uint8_error": (value - pil).abs().max().item(),
            "changed_fraction": (value != pil).float().mean().item(),
        }
        for name, value in [("full_float", full), ("quantize_each_axis", separable)]
    }
    mixed = [
        [image_fixture(448, 448, "ramp"), fixture],
        [],
        [image_fixture(479, 641, "edges")],
    ]
    expected = reference(mixed, max_slice_nums=9)
    actual = process_images(geometry, mixed, max_slice_nums=9)
    for expected_batch, actual_batch in zip(
        expected["pixel_values"], actual["pixel_values"], strict=True
    ):
        assert [tuple(value.shape) for value in expected_batch] == [
            tuple(value.shape) for value in actual_batch
        ]
    report["mixed_batch_shapes"] = "passed"
    report["current_device_outputs"] = []
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            output = process_images(geometry, [[fixture]])
            report["current_device_outputs"].append(
                str(output["pixel_values"][0][0].device)
            )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        process_images(geometry, [[fixture]], max_slice_nums=9)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(arguments.output / "profile.json"))
    (arguments.output / "operators.txt").write_text(
        profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    )
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "measurements"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
