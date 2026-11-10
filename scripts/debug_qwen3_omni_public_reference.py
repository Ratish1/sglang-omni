# SPDX-License-Identifier: Apache-2.0
"""Run complete checkpoint generation on saved audio-and-text server requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import soundfile
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, TypeAdapter
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.serve.protocol import ChatCompletionRequest

AUDIO_SAMPLE_RATE = 24000


class ReferenceCase(BaseModel):
    sample_id: str
    target_text: str
    request_file: Path


@dataclass(kw_only=True)
class LayerObservation:
    family: Literal["audio", "vision", "thinker", "talker", "predictor", "vocoder"]
    layer_index: int
    calls: int = 0
    last_output: torch.Tensor | None = None

    def __call__(
        self,
        module: torch.nn.Module,
        arguments: tuple[torch.Tensor | None, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        self.calls += 1
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.last_output = hidden_states.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--observe-layers", action="store_true")
    arguments = parser.parse_args()
    cases = TypeAdapter(list[ReferenceCase]).validate_json(
        arguments.case_file.read_text()
    )
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    model, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    assert not loading["unexpected_keys"], loading["unexpected_keys"]
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    layer_observations: list[LayerObservation] = []
    if arguments.observe_layers:
        for family, layers in (
            ("audio", model.thinker.audio_tower.layers),
            ("vision", model.thinker.visual.blocks),
            ("thinker", model.thinker.model.layers),
            ("talker", model.talker.model.layers),
            ("predictor", model.talker.code_predictor.model.layers),
            ("vocoder", model.code2wav.pre_transformer.layers),
        ):
            for layer_index, layer in enumerate(layers):
                observation = LayerObservation(family=family, layer_index=layer_index)
                layer_observations.append(observation)
                layer.register_forward_hook(observation)
    else:
        pass
    reports = []
    generated = []
    for case in cases:
        request_bytes = case.request_file.read_bytes()
        request = ChatCompletionRequest.model_validate_json(request_bytes)
        assert request.request_id is not None
        assert request.audios and request.max_tokens is not None
        assert request.temperature is not None and request.temperature > 0
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        assert messages[-1]["role"] == "user"
        assert isinstance(messages[-1]["content"], str)
        messages[-1]["content"] = [
            *({"type": "audio", "path": audio} for audio in request.audios),
            {"type": "text", "text": messages[-1]["content"]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        seed = derive_sampling_seed("sglang-omni-unseeded-row", request.request_id)
        torch.manual_seed(seed)
        for observation in layer_observations:
            observation.calls = 0
            observation.last_output = None
        started = time.monotonic()
        with torch.inference_mode():
            text_ids, waveform = model.generate(
                **inputs,
                thinker_max_new_tokens=request.max_tokens,
                thinker_do_sample=True,
                thinker_temperature=request.temperature,
                thinker_top_p=1.0,
                thinker_top_k=0,
            )
        assert torch.isfinite(waveform).all() and waveform.numel() > 0
        waveform = waveform.reshape(-1).float().cpu().numpy()
        elapsed_seconds = time.monotonic() - started
        audio_file = arguments.output_directory / f"{case.sample_id}.wav"
        soundfile.write(audio_file, waveform, AUDIO_SAMPLE_RATE, subtype="PCM_16")
        layer_reports = []
        for observation in layer_observations:
            if observation.last_output is not None:
                assert torch.isfinite(observation.last_output).all()
                shape = list(observation.last_output.shape)
            else:
                assert observation.family == "vision"
                shape = None
            layer_reports.append(
                {
                    "family": observation.family,
                    "layer": observation.layer_index,
                    "calls": observation.calls,
                    "last_output_shape": shape,
                }
            )
        generated.append(
            {
                "sample_id": case.sample_id,
                "target_text": case.target_text,
                "wav_path": str(audio_file),
                "is_success": True,
                "audio_duration_s": len(waveform) / AUDIO_SAMPLE_RATE,
                "latency_s": elapsed_seconds,
            }
        )
        report = {
            **generated[-1],
            "request_id": request.request_id,
            "request_file_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "torch_seed": seed,
            "thinker_text": processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
            "layers": layer_reports,
        }
        reports.append(report)
        (arguments.output_directory / f"{case.sample_id}.json").write_text(
            json.dumps(report, indent=2)
        )
        (arguments.output_directory / "generated.json").write_text(
            json.dumps(generated, indent=2)
        )
        print(json.dumps(generated[-1]), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "model_path": arguments.model_path,
                "model_revision": arguments.model_revision,
                "observe_layers": arguments.observe_layers,
                "transformers": version("transformers"),
                "torch": version("torch"),
                "parameter_dtypes": sorted(
                    {str(parameter.dtype) for parameter in model.parameters()}
                ),
                "cases": reports,
                "scope": "Full public checkpoint generation with real preprocessing and no frozen hidden states. Torch RNG and the sampled residual-code predictor differ from the native counter-based main-code RNG and greedy residual codes. These generations are not a same-noise numerical comparison or a speed benchmark. Optional layer observations count execution and check each last output, not every intermediate of every frame. Vision is not exercised by these audio/text cases.",
            },
            indent=2,
        )
    )
    checkpoint_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    audit_checkpoint_weights(
        model,
        checkpoint_directory,
        arguments.output_directory / "loaded-weight-audit.json",
    )


if __name__ == "__main__":
    main()
