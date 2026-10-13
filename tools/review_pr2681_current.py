"""Verify revised configuration and the exact example's Coordinator lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import ValidationError

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.llada2_uni.config import (
    LLaDA2ImageDecoderFactoryArgs,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.proto import OmniRequest


def check_configuration() -> None:
    configuration = ConfigManager.from_file("examples/configs/llada2_uni.yaml").config
    preparation = prepare_pipeline_runtime(configuration)
    try:
        print(
            json.dumps(
                {
                    "check": "example_placement",
                    "stages": preparation.process_plan.stage_to_process,
                    "budgets": {
                        stage.name: stage.gpu_memory_fraction
                        for stage in configuration.stages
                        if stage.gpu is not None
                    },
                }
            ),
            flush=True,
        )
    finally:
        preparation.runtime_dir.close()
    configuration = LLaDA2UniOmniPipelineConfig(model_path="unused")
    configuration.stage_named("image_decode").factory = LLaDA2ImageDecoderFactoryArgs(
        backend="sglang",
        decode_mode="decoder-turbo",
        num_steps=7,
        resolution_multiplier=3,
        attention_backend="torch_sdpa",
    )
    expected = resolve_stage_typed_kwargs(configuration.stage_named("image_decode"))
    resolved = ConfigResolver(configuration).resolve(ConfigPatchSet()).config
    actual = resolve_stage_typed_kwargs(resolved.stage_named("image_decode"))
    assert actual == expected
    try:
        ConfigManager(resolved).merge_config(
            {"image_decode.factory.backend": "invalid"}
        )
    except ValidationError:
        print(
            json.dumps(
                {
                    "check": "typed_resolution",
                    "preserved": actual,
                    "invalid_backend_rejected": True,
                }
            ),
            flush=True,
        )
    else:
        raise AssertionError(
            "Invalid decoder backend survived configuration resolution"
        )


async def check_model(directory: Path, backend: Literal["diffusers", "sglang"]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    configuration = ConfigManager.from_file("examples/configs/llada2_uni.yaml").config
    if backend == "sglang":
        configuration = ConfigManager(configuration).merge_config(
            {"image_decode.factory.backend": backend}
        )
    else:
        pass
    runner = MultiProcessPipelineRunner(configuration)
    fingerprints: dict[str, str] = {}

    def t2i(width: int, height: int) -> OmniRequest:
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

    async def generate(
        request_id: str, request: OmniRequest, expected_size: tuple[int, int]
    ) -> None:
        outcome = await asyncio.wait_for(
            runner.coordinator.submit(request_id, request), timeout=600
        )
        assert isinstance(outcome, dict) and set(outcome) == {"decode", "image_decode"}
        decoded = outcome["image_decode"]
        with Image.open(io.BytesIO(base64.b64decode(decoded["image"]))) as image:
            image.load()
            assert image.mode == "RGB" and image.size == expected_size
            fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
            fingerprints[request_id] = fingerprint
            print(
                json.dumps(
                    {
                        "check": "model_image",
                        "request_id": request_id,
                        "size": image.size,
                        "sha256": fingerprint,
                        "event_type": decoded["events"][0]["type"],
                    }
                ),
                flush=True,
            )
        assert decoded["events"][0]["type"] == "image_final"

    try:
        await runner.start(timeout=1200)
        print(
            json.dumps(
                {
                    "check": "model_ready",
                    "placement_override": False,
                    "decoder_backend": backend,
                    "decode_mode": "normal",
                    "decoder_steps": 50,
                }
            ),
            flush=True,
        )
        await generate("rectangular", t2i(128, 96), (128, 96))
        await generate("repeat", t2i(128, 96), (128, 96))
        assert fingerprints["rectangular"] == fingerprints["repeat"]
        await asyncio.gather(
            generate("concurrent-square", t2i(96, 96), (96, 96)),
            generate("concurrent-rectangle", t2i(128, 96), (128, 96)),
        )
        assert fingerprints["rectangular"] == fingerprints["concurrent-rectangle"]
        source = Image.new("RGB", (640, 480))
        source.putdata(
            [(x % 256, y % 256, (x + y) % 256) for y in range(480) for x in range(640)]
        )
        source_path = directory / "source.png"
        source.save(source_path)
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
                        "cfg_text_scale": 3.5,
                        "cfg_image_scale": 1.5,
                        "cfg_rescale": 0.6,
                        "dllm_steps": 8,
                        "decoder_steps": 2,
                        "seed": 9,
                    },
                },
            ),
            (1152, 864),
        )
        outcome = await asyncio.wait_for(
            runner.coordinator.submit(
                "text-after-image",
                OmniRequest(
                    inputs={"messages": [{"role": "user", "content": "Say hi."}]},
                    params={"max_new_tokens": 32},
                ),
            ),
            timeout=120,
        )
        assert outcome["decode"]["text"] and outcome["image_decode"]["skipped"]
        print(
            json.dumps(
                {
                    "check": "model_chat",
                    "text": outcome["decode"]["text"],
                    "image_skipped": True,
                }
            ),
            flush=True,
        )
    finally:
        await runner.stop()
        print(json.dumps({"check": "model_stopped"}), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true")
    parser.add_argument(
        "--backend", choices=("diffusers", "sglang"), default="diffusers"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/data/pr2681-rereview-0d6b")
    )
    arguments = parser.parse_args()
    check_configuration()
    if arguments.model:
        asyncio.run(check_model(arguments.output, arguments.backend))
    else:
        pass
