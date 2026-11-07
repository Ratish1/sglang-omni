# SPDX-License-Identifier: Apache-2.0
"""Compare a real speech request before and after worker cache reuse."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import uuid
from pathlib import Path

import aiohttp
import torch
import xxhash

from sglang_omni.sampling.seed import SAMPLING_SEED_MASK
from sglang_omni.serve.protocol import ChatCompletionRequest


async def run(
    base_url: str,
    request_file: Path,
    distractor_file: Path,
    audio_directory: Path,
    capture_directory: Path,
    output_directory: Path,
) -> None:
    target = ChatCompletionRequest.model_validate_json(request_file.read_bytes())
    distractor = ChatCompletionRequest.model_validate_json(distractor_file.read_bytes())
    assert target.request_id is not None and distractor.request_id is not None
    assert target.temperature == distractor.temperature == 0.0
    assert target.stream is distractor.stream is False
    output_directory.mkdir(exist_ok=False)
    observations = []
    target_captures = []
    async with aiohttp.ClientSession() as session:
        for label, request in (
            ("target-original", target),
            ("target-first", target),
            ("unrelated", distractor),
            ("target-after-unrelated", target),
            ("target-repeat", target),
        ):
            assert request.request_id is not None and request.audios
            sampling_seed = request.seed
            if sampling_seed is None:
                sampling_seed = (
                    xxhash.xxh64_intdigest(request.request_id.encode())
                    & SAMPLING_SEED_MASK
                )
            else:
                pass
            if label == "target-original":
                request_id = request.request_id
                submitted_seed = request.seed
            else:
                request_id = str(uuid.uuid5(uuid.UUID(request.request_id), label))
                submitted_seed = sampling_seed
            audio_paths = [audio_directory / Path(path).name for path in request.audios]
            assert all(path.is_file() for path in audio_paths)
            submitted = request.model_copy(
                update={
                    "request_id": request_id,
                    "seed": submitted_seed,
                    "audios": [str(path) for path in audio_paths],
                }
            )
            request_bytes = submitted.model_dump_json(exclude_none=True).encode()
            (output_directory / f"{label}.request.json").write_bytes(request_bytes)
            async with session.post(
                f"{base_url}/v1/chat/completions",
                data=request_bytes,
                headers={"Content-Type": "application/json"},
            ) as response:
                response_bytes = await response.read()
                (output_directory / f"{label}.response.json").write_bytes(
                    response_bytes
                )
                response.raise_for_status()
            response_payload = json.loads(response_bytes)
            choice = response_payload["choices"][0]
            message = choice["message"]
            waveform = base64.b64decode(message["audio"]["data"], validate=True)
            (output_directory / f"{label}.wav").write_bytes(waveform)
            observation = {
                "label": label,
                "request_id": request_id,
                "source_request_id": request.request_id,
                "sampling_seed": sampling_seed,
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "audio_input_sha256": [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in audio_paths
                ],
                "text": message["content"],
                "finish_reason": choice["finish_reason"],
                "usage": response_payload["usage"],
                "waveform_sha256": hashlib.sha256(waveform).hexdigest(),
                "capture_directory": str(capture_directory / request_id),
            }
            observations.append(observation)
            (output_directory / "observations.json").write_text(
                json.dumps(observations, indent=2)
            )
            print(json.dumps(observation), flush=True)
            if label == "unrelated":
                continue
            else:
                pass
            captured = capture_directory / request_id
            decode_file = next(captured.glob("*-before_decode.pt"))
            decode = torch.load(decode_file, map_location="cpu", weights_only=True)
            assert decode["sampling_seed"].tolist() == [sampling_seed]
            target_captures.append(captured)
    comparisons = []
    baseline_directory = target_captures[0]
    for captured in target_captures[1:]:
        fields = {}
        for baseline_file in sorted(baseline_directory.glob("*.pt")):
            compared_file = captured / baseline_file.name
            if not compared_file.is_file():
                fields[baseline_file.name] = {"matching_file": False}
                continue
            else:
                pass
            baseline = torch.load(baseline_file, map_location="cpu", weights_only=True)
            compared = torch.load(compared_file, map_location="cpu", weights_only=True)
            compared_fields = baseline.keys() & compared.keys()
            if baseline_file.name.endswith("-before_prefill.pt"):
                compared_fields &= {
                    "prefill_embeds",
                    "trailing_text_embeds",
                    "positions",
                    "input_ids",
                }
            elif baseline_file.name.endswith("-post_prefill.pt"):
                compared_fields -= {
                    "sampling_seed",
                    "feedback_input",
                    "repetition_mask",
                }
            else:
                pass
            fields[baseline_file.name] = {
                name: baseline[name].shape == compared[name].shape
                and torch.equal(baseline[name], compared[name])
                for name in sorted(compared_fields)
            }
        comparisons.append({"capture_directory": str(captured), "fields": fields})
    (output_directory / "tensor-comparison.json").write_text(
        json.dumps(comparisons, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--distractor-file", type=Path, required=True)
    parser.add_argument("--audio-directory", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(
        run(
            arguments.base_url,
            arguments.request_file,
            arguments.distractor_file,
            arguments.audio_directory,
            arguments.capture_directory,
            arguments.output_directory,
        )
    )


if __name__ == "__main__":
    main()
