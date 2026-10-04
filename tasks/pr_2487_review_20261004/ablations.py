"""Isolate packing, input layout, placement, and real processor integration."""

import argparse
import asyncio
import inspect
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

from sglang_omni.comm.router import CommRouter
from sglang_omni.models.minicpm_o.components import image_processing
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.config import text_stages
from sglang_omni.proto import OmniRequest, StagePayload
from tasks.pr_2487_review_20261004.probe import image_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = snapshot_download(
        "openbmb/MiniCPM-o-4_5",
        revision="503e754207c94da6bb26850b4469f367c9ea3582",
        allow_patterns=["*.py", "*.json", "*.txt", "*.jinja", "*.model"],
    )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    geometry = processor.image_processor
    reference_pack = geometry.reshape_by_patch

    def reshape_pack(pixels: np.ndarray) -> np.ndarray:
        channels, height, width = pixels.shape
        patch_size = geometry.patch_size
        return (
            pixels.reshape(
                channels,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(channels, patch_size, -1)
        )

    source = inspect.getsource(image_processing)
    assert source.count(".unsqueeze(0).float()") == 1
    namespace = {}
    exec(
        compile(
            source.replace(
                ".unsqueeze(0).float()", ".unsqueeze(0).float().contiguous()"
            ),
            "contiguous_ablation.py",
            "exec",
        ),
        namespace,
    )
    contiguous_process = namespace["process_images"]
    report = {}
    fixture = image_fixture(641, 479, "noise")
    preprocessor = MiniCPMOPreprocessor(checkpoint)
    payload = StagePayload(
        request_id="review-image",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "Describe this."}],
                "images": [fixture],
            }
        ),
        data=None,
    )
    try:
        asyncio.run(preprocessor(payload))
    except TypeError as error:
        report["actual_preprocessor_error"] = str(error)
    else:
        report["actual_preprocessor_error"] = None

    torch.cuda.set_device(1)
    report["main_thread_device"] = torch.cuda.current_device()

    def worker() -> None:
        output = image_processing.process_images(geometry, [[fixture]])
        report["new_scheduler_thread_device"] = str(output["pixel_values"][0][0].device)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    torch.cuda.set_device(0)
    report["stage_gpu_assignments"] = {stage.name: stage.gpu for stage in text_stages()}
    router = CommRouter(
        stage_name="preprocessing",
        gpu_id=None,
        same_process_targets=set(),
        gpu_stage_names={"image_encoder"},
        stage_gpu_ids={"image_encoder": (1,)},
    )
    cuda_output = image_processing.process_images(geometry, [[fixture]])
    report["separate_process_transport"] = router.outbound_payload(
        "image_encoder", cuda_output
    ).value
    router.close()

    report["ablations"] = []
    for width, height in [
        (448, 448),
        (641, 479),
        (1920, 1080),
        (3840, 2160),
        (8000, 6000),
    ]:
        image = image_fixture(width, height, "noise")
        timings = {
            "cpu_unfold": [],
            "cpu_reshape": [],
            "gpu_pr": [],
            "gpu_contiguous": [],
        }
        baseline = processor.process_image([[image]], max_slice_nums=9)
        geometry.reshape_by_patch = reshape_pack
        packed = processor.process_image([[image]], max_slice_nums=9)
        for expected, actual in zip(
            baseline["pixel_values"][0], packed["pixel_values"][0], strict=True
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        geometry.reshape_by_patch = reference_pack
        baseline_gpu = image_processing.process_images(geometry, [[image]])
        contiguous_gpu = contiguous_process(geometry, [[image]])
        for expected, actual in zip(
            baseline_gpu["pixel_values"][0],
            contiguous_gpu["pixel_values"][0],
            strict=True,
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for iteration in range(12):
            for name in list(timings) if iteration % 2 else list(timings)[::-1]:
                geometry.reshape_by_patch = (
                    reshape_pack if name == "cpu_reshape" else reference_pack
                )
                torch.cuda.synchronize()
                begin = time.perf_counter()
                if name == "gpu_pr":
                    output = image_processing.process_images(geometry, [[image]])
                elif name == "gpu_contiguous":
                    output = contiguous_process(geometry, [[image]])
                else:
                    output = processor.process_image([[image]], max_slice_nums=9)
                torch.cuda.synchronize()
                if iteration >= 2:
                    timings[name].append((time.perf_counter() - begin) * 1000)
                else:
                    pass
                del output
        geometry.reshape_by_patch = reference_pack
        row = {
            "size": [width, height],
            "cpu_packing_bitwise_equal": True,
            "gpu_layout_bitwise_equal": True,
            "latency_ms": {
                name: statistics.median(values) for name, values in timings.items()
            },
        }
        report["ablations"].append(row)
        print(json.dumps(row), flush=True)
    (arguments.output / "ablations.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
