# SPDX-License-Identifier: Apache-2.0
"""Run complete checkpoint generation on saved audio-and-text server requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import soundfile
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, TypeAdapter
from torch.utils.hooks import RemovableHandle
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from scripts.qwen3_omni_checkpoint_weights import (
    audit_checkpoint_buffers,
    audit_checkpoint_weights,
)
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.serve.protocol import ChatCompletionRequest

AUDIO_SAMPLE_RATE = 24000


class ReferenceCase(BaseModel):
    sample_id: str
    target_text: str
    request_file: Path
    capture_generation: bool = True


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


@dataclass(kw_only=True)
class CodecInputObservation:
    codec_chunks: list[torch.Tensor] = field(default_factory=list)

    def __call__(
        self,
        module: Qwen3OmniMoeCode2Wav,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        codec_ids = inputs[0]
        assert codec_ids.shape[1] == module.config.num_quantizers
        self.codec_chunks.append(codec_ids.detach().cpu())


@dataclass(kw_only=True)
class TensorCallObservation:
    operation_name: str
    tensor_calls: list[dict[str, torch.Tensor]] = field(default_factory=list)

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        assert not module.training
        if isinstance(output, torch.Tensor):
            output_tensors = (output,)
        else:
            output_tensors = output
        self.tensor_calls.append(
            {
                **{
                    f"input_{index}": tensor.detach().cpu()
                    for index, tensor in enumerate(inputs)
                },
                **{
                    f"output_{index}": tensor.detach().cpu()
                    for index, tensor in enumerate(output_tensors)
                },
                "cuda_rng_state": torch.cuda.get_rng_state(inputs[0].device),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--observe-layers", action="store_true")
    parser.add_argument("--save-codec-inputs", action="store_true")
    parser.add_argument("--save-generation-trace", action="store_true")
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
    audit_checkpoint_buffers(model, arguments.output_directory / "loaded-model-buffers")
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
    codec_observation = CodecInputObservation()
    if arguments.save_codec_inputs:
        model.code2wav.register_forward_pre_hook(codec_observation)
    else:
        pass
    generation_observations: list[TensorCallObservation] = []
    generation_modules: dict[str, torch.nn.Module] = {}
    generation_hook_handles: list[RemovableHandle] = []
    if arguments.save_generation_trace:
        generation_modules = {
            "audio_projection": model.thinker.audio_tower.proj2,
            "text_embedding": model.thinker.model.embed_tokens,
            "thinker_rotary": model.thinker.model.rotary_emb,
            "text_projection": model.talker.text_projection,
            "hidden_projection": model.talker.hidden_projection,
            "talker_rotary": model.talker.model.rotary_emb,
            "codec_head": model.talker.codec_head,
            "predictor_rotary": model.talker.code_predictor.model.rotary_emb,
            "vocoder_rotary": model.code2wav.pre_transformer.rotary_emb,
            **{
                f"residual_head_{index}": residual_head
                for index, residual_head in enumerate(
                    model.talker.code_predictor.lm_head
                )
            },
        }
        for operation_name in generation_modules:
            generation_observation = TensorCallObservation(
                operation_name=operation_name
            )
            generation_observations.append(generation_observation)
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
        codec_observation.codec_chunks.clear()
        for hook_handle in generation_hook_handles:
            hook_handle.remove()
        generation_hook_handles.clear()
        for generation_observation in generation_observations:
            generation_observation.tensor_calls.clear()
            if case.capture_generation:
                module = generation_modules[generation_observation.operation_name]
                generation_hook_handles.append(
                    module.register_forward_hook(generation_observation)
                )
            else:
                pass
        if arguments.save_generation_trace:
            processor_inputs = {
                name: tensor.detach().cpu() for name, tensor in inputs.items()
            }
            initial_cpu_rng_state = torch.get_rng_state()
            initial_cuda_rng_state = torch.cuda.get_rng_state(model.device)
        else:
            pass
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
        if arguments.save_codec_inputs:
            assert codec_observation.codec_chunks
            torch.save(
                {"codec_chunks": codec_observation.codec_chunks},
                arguments.output_directory / f"{case.sample_id}.codec-inputs.pt",
            )
        else:
            pass
        if arguments.save_generation_trace:
            torch.save(
                {
                    "processor_inputs": processor_inputs,
                    "initial_cpu_rng_state": initial_cpu_rng_state,
                    "initial_cuda_rng_state": initial_cuda_rng_state,
                    "operations": {
                        observation.operation_name: observation.tensor_calls
                        for observation in generation_observations
                    },
                },
                arguments.output_directory / f"{case.sample_id}.generation.pt",
            )
        else:
            pass
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
            "waveform_sha256": hashlib.sha256(audio_file.read_bytes()).hexdigest(),
            "codec_chunk_shapes": [
                list(chunk.shape) for chunk in codec_observation.codec_chunks
            ],
            "capture_generation": arguments.save_generation_trace
            and case.capture_generation,
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
                "save_codec_inputs": arguments.save_codec_inputs,
                "save_generation_trace": arguments.save_generation_trace,
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
    audit_checkpoint_buffers(
        model, arguments.output_directory / "generated-model-buffers"
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
