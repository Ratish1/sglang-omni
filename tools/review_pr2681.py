"""Exercise image-pipeline configuration and real Coordinator requests."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
from pathlib import Path

from PIL import Image

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.llada2_uni.config import (
    LLaDA2ImageDecoderFactoryArgs,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.proto import OmniRequest


def configuration_boundaries() -> None:
    configuration = LLaDA2UniOmniPipelineConfig(model_path="unused")
    try:
        preparation = prepare_pipeline_runtime(configuration)
    except ValueError as error:
        print(json.dumps({"default_placement_error": str(error)}), flush=True)
    else:
        preparation.runtime_dir.close()
        print(json.dumps({"default_placement_error": None}), flush=True)

    configuration.stage_named("image_decode").factory = LLaDA2ImageDecoderFactoryArgs(
        backend="sglang", num_steps=7
    )
    before = resolve_stage_typed_kwargs(configuration.stage_named("image_decode"))
    resolved = ConfigResolver(configuration).resolve(ConfigPatchSet()).config
    after = resolve_stage_typed_kwargs(resolved.stage_named("image_decode"))
    print(json.dumps({"before_resolution": before, "after_resolution": after}))
    for backend in ("sglang", "invalid"):
        merged = ConfigManager(configuration).merge_config(
            {"image_decode.factory.backend": backend}
        )
        print(
            json.dumps(
                {
                    "written_backend": backend,
                    "resolved_factory_type": type(
                        merged.stage_named("image_decode").factory
                    ).__name__,
                    "resolved_kwargs": resolve_stage_typed_kwargs(
                        merged.stage_named("image_decode")
                    ),
                }
            )
        )


async def model_requests(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.new("RGB", (640, 480))
    source.putdata(
        [(x % 256, y % 256, (x + y) % 256) for y in range(480) for x in range(640)]
    )
    source_path = directory / "source.png"
    source.save(source_path)
    configuration = LLaDA2UniOmniPipelineConfig(model_path="inclusionAI/LLaDA2.0-Uni")
    # note (ratish): bypass the confirmed placement defect only for diagnosis.
    configuration.placement.require_memory_fraction_for_colocation = False
    configuration.stage_named("image_decode").factory = FactoryArgs(
        backend="sglang", decode_mode="decoder-turbo", num_steps=2
    )
    runner = MultiProcessPipelineRunner(configuration)
    records: list[dict[str, str | int | bool]] = []

    async def generate(request_id: str, request: OmniRequest) -> None:
        result = await asyncio.wait_for(
            runner.coordinator.submit(request_id, request), timeout=600
        )
        assert isinstance(result, dict)
        assert set(result) == {"decode", "image_decode"}
        decoded = result["image_decode"]
        image_bytes = base64.b64decode(decoded["image"])
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            assert image.mode == "RGB"
            record = {
                "request_id": request_id,
                "width": image.width,
                "height": image.height,
                "png_bytes": len(image_bytes),
                "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "event_type": decoded["events"][0]["type"],
            }
        records.append(record)
        print(json.dumps(record), flush=True)

    def text_to_image(width: int, height: int) -> OmniRequest:
        return OmniRequest(
            inputs={"messages": [{"role": "user", "content": "A red apple."}]},
            metadata={
                "output_modalities": ["image"],
                "image_generation": {
                    "image_h": height,
                    "image_w": width,
                    "cfg_scale": 4.0,
                    "cfg_rescale": 0.7,
                    "dllm_steps": 8,
                    "seed": 42,
                },
            },
        )

    try:
        await runner.start(timeout=1200)
        print("PIPELINE_READY_WITH_DIAGNOSTIC_PLACEMENT_OVERRIDE", flush=True)
        await generate("t2i-rectangular", text_to_image(128, 96))
        await generate("t2i-repeat", text_to_image(128, 96))
        assert records[0]["sha256"] == records[1]["sha256"]
        await generate(
            "edit-three-branches",
            OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": "Make it green."}],
                    "images": [str(source_path)],
                },
                metadata={
                    "output_modalities": ["image"],
                    "image_generation": {
                        "cfg_text_scale": 4.0,
                        "cfg_image_scale": 1.5,
                        "dllm_steps": 8,
                        "seed": 42,
                    },
                },
            ),
        )
        chat = await asyncio.wait_for(
            runner.coordinator.submit(
                "text-after-image",
                OmniRequest(
                    inputs={"messages": [{"role": "user", "content": "Say hi."}]},
                    params={"max_new_tokens": 32},
                ),
            ),
            timeout=120,
        )
        assert chat["image_decode"]["skipped"]
        assert chat["decode"]["text"]
        print(json.dumps({"text_after_image": chat["decode"]["text"]}), flush=True)
        await asyncio.gather(
            generate("concurrent-a", text_to_image(96, 96)),
            generate("concurrent-b", text_to_image(128, 96)),
        )
    finally:
        await runner.stop()
        (directory / "model-results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("boundaries", "model"))
    parser.add_argument("--directory", type=Path, default=Path("/data/pr2681-review"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if arguments.mode == "boundaries":
        configuration_boundaries()
    else:
        asyncio.run(model_requests(arguments.directory))
