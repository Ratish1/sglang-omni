# SPDX-License-Identifier: Apache-2.0
"""Replay captured codec histories through the native waveform scheduler."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavTransformerLayer,
)

from sglang_omni.client.audio import encode_wav
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_omni.components import incremental_codec
from sglang_omni.models.qwen3_omni.components.code2wav import FusedCode2WavTransformer
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    create_code2wav_scheduler,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import (
    CodecStatePool,
    RingAddresses,
)


def capture_cold_window(
    scheduler: Code2WavScheduler, codes: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Capture real cold-window boundaries without changing the decoder's operands."""
    codec = scheduler.incremental_codec
    pool = scheduler.codec_pool
    assert codec is not None and pool is not None
    transformer = codec.model.pre_transformer
    assert isinstance(transformer, FusedCode2WavTransformer)
    row_count = codes.size(0)
    slot_indices = pool.stage_index(list(range(row_count)))
    unobserved_waveform = codec.decode_slots(
        codes, pool, slot_indices, first_decode=True
    )
    boundaries: dict[str, torch.Tensor] = {}
    layer_indices = {
        layer: index for index, layer in enumerate(transformer.transformer.layers)
    }
    module_names = {module: name for name, module in codec.model.named_modules()}
    project = transformer.projected_qkv
    attend = transformer.attend
    layer_output = transformer.layer_output
    slot_forward = transformer.forward_slot_frames
    convolution = incremental_codec.channels_last_conv1d
    transposed_convolution = incremental_codec.channels_last_conv_transpose1d

    def record(name: str, tensor: torch.Tensor) -> None:
        assert name not in boundaries, name
        assert tensor.size(0) % row_count == 0, (name, tensor.shape)
        boundaries[name] = tensor[: tensor.size(0) // row_count].detach().cpu()

    def project_and_record(
        layer: Qwen3OmniMoeCode2WavTransformerLayer, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_name = f"transformer.{layer_indices[layer]}"
        record(f"{layer_name}.input", hidden_states)
        query, key, value = project(layer, hidden_states)
        for operand_name, tensor in zip(
            ("query", "key", "value"), (query, key, value), strict=True
        ):
            record(f"{layer_name}.{operand_name}", tensor)
        return query, key, value

    def attend_and_record(
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        window_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        layer_name = f"transformer.{layer_indices[layer]}"
        record(f"{layer_name}.rotary_query", query)
        record(f"{layer_name}.rotary_key", keys)
        attended = attend(layer, query, keys, values, window_mask)
        record(f"{layer_name}.attention", attended)
        return attended

    def layer_output_and_record(
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        residual: torch.Tensor,
        attended: torch.Tensor,
    ) -> torch.Tensor:
        output = layer_output(layer, residual, attended)
        record(f"transformer.{layer_indices[layer]}.output", output)
        return output

    def slot_forward_and_record(
        inputs_embeds: torch.Tensor,
        pool: CodecStatePool,
        addresses: RingAddresses,
        *,
        first_decode: bool,
        is_paged_attention: bool,
    ) -> torch.Tensor:
        record("codec_embedding_mean", inputs_embeds)
        output = slot_forward(
            inputs_embeds,
            pool,
            addresses,
            first_decode=first_decode,
            is_paged_attention=is_paged_attention,
        )
        record("transformer.final_norm", output)
        return output

    def convolution_and_record(
        hidden_states: torch.Tensor,
        conv: torch.nn.Conv1d,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        length: int,
    ) -> torch.Tensor:
        name = module_names[conv]
        record(f"{name}.input", hidden_states)
        output = convolution(hidden_states, conv, weight, bias, length)
        record(f"{name}.output", output)
        return output

    def transposed_convolution_and_record(
        hidden_states: torch.Tensor,
        conv: torch.nn.ConvTranspose1d,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        name = module_names[conv]
        record(f"{name}.input", hidden_states)
        output = transposed_convolution(hidden_states, conv, weight, bias)
        record(f"{name}.output", output)
        return output

    with (
        patch.object(transformer, "projected_qkv", new=project_and_record),
        patch.object(transformer, "attend", new=attend_and_record),
        patch.object(transformer, "layer_output", new=layer_output_and_record),
        patch.object(transformer, "forward_slot_frames", new=slot_forward_and_record),
        patch.object(
            incremental_codec, "channels_last_conv1d", new=convolution_and_record
        ),
        patch.object(
            incremental_codec,
            "channels_last_conv_transpose1d",
            new=transposed_convolution_and_record,
        ),
    ):
        waveform = codec.decode_slots(
            codes,
            pool,
            slot_indices,
            first_decode=True,
        )
    assert torch.equal(unobserved_waveform, waveform)
    record("waveform", waveform)
    return boundaries


def replay(
    scheduler: Code2WavScheduler,
    histories: dict[str, torch.Tensor],
    *,
    batch_windows: bool,
) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, ...]]]:
    states = {}
    for request_id, frames in histories.items():
        state = scheduler.get_or_create_stream_state(request_id)
        assert state is not None
        state.stream_enabled = False
        for frame in frames.unbind(0):
            scheduler.ingest(request_id, state, frame)
        assert state.codec_slot == len(states), (request_id, state.codec_slot)
        states[request_id] = state
    window_groups: list[tuple[str, ...]] = []
    while participants := scheduler.ready_window_group():
        group = participants if batch_windows else participants[:1]
        scheduler.decode_windows(group)
        window_groups.append(tuple(request_id for request_id, _ in group))
    for request_id, state in reversed(states.items()):
        scheduler.flush_pending(request_id, state)
        scheduler.decode_delta(request_id, state, is_final=True)
        assert state.emitted == len(histories[request_id]) - 1, request_id
        assert state.codec_slot is None
        scheduler.clear_stream_state(request_id)
    return (
        {request_id: state.audio_parts for request_id, state in states.items()},
        window_groups,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--cold-window-boundaries", action="store_true")
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu_id)
    config = ConfigManager.from_file(
        str(arguments.server_config), model_path_override=arguments.model_path
    ).config
    assert config is not None
    stage = config.stage_named("code2wav")
    factory_parameters = resolve_stage_factory_args(
        stage, config, gpu_id=arguments.gpu_id
    )
    scheduler = create_code2wav_scheduler(**factory_parameters)
    graph_runner = scheduler.cuda_graph_runner
    assert graph_runner is not None and graph_runner.enabled
    assert scheduler.incremental_codec is not None
    scheduler.on_serving_start()
    directory = arguments.witness_directory
    manifests = json.loads((directory / "witnesses.json").read_bytes())
    captured = json.loads((directory / "codec-histories.json").read_bytes())
    histories = {
        row["directory"]: torch.tensor(
            captured[row["directory"]], device=scheduler.device, dtype=torch.long
        )
        for row in manifests
        if row["request_count"] == 1
    }
    for frames in histories.values():
        assert frames.shape[1] == scheduler.model.config.num_quantizers
        assert (
            frames[:, 0] == scheduler.codec_eos_token_id
        ).nonzero().flatten().tolist() == [len(frames) - 1]
    if arguments.cold_window_boundaries:
        request_id = next(iter(histories))
        first_frame_count = (
            scheduler.initial_codec_chunk_frames or scheduler.stream_chunk_size
        )
        codes = torch.stack(
            [frames[:first_frame_count].T for frames in histories.values()]
        ).contiguous()
        solo = capture_cold_window(scheduler, codes[:1])
        peers = capture_cold_window(scheduler, codes)
        repeat = capture_cold_window(scheduler, codes[:1])
        assert list(solo) == list(peers) == list(repeat)
        assert all(torch.equal(solo[name], repeat[name]) for name in solo)
        report = []
        for name in solo:
            delta = solo[name].float() - peers[name].float()
            report.append(
                {
                    "boundary": name,
                    "shape": list(delta.shape),
                    "different_elements": int(delta.count_nonzero()),
                    "max_abs_difference": float(delta.abs().max()),
                }
            )
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "peer_request_ids": list(histories),
                    "code_shape": list(codes.shape),
                    "boundaries": report,
                }
            )
        )
        return
    else:
        pass
    waveforms: dict[str, dict[str, list[np.ndarray]]] = {}
    runs = {}
    for name, graph_enabled, batch_windows in (
        ("graph_solo", True, False),
        ("graph_peers", True, True),
        ("eager_solo", False, False),
        ("eager_peers", False, True),
        ("graph_solo_repeat", True, False),
    ):
        graph_runner.enabled = graph_enabled
        before = graph_runner.graph_replays
        fallback_counts = graph_runner.fallback_counts.copy()
        waveforms[name], groups = replay(
            scheduler, histories, batch_windows=batch_windows
        )
        graph_replays = graph_runner.graph_replays - before
        if graph_enabled:
            assert graph_runner.fallback_counts == fallback_counts
            assert graph_replays == len(groups) + sum(
                len(parts) > sum(request_id in group for group in groups)
                for request_id, parts in waveforms[name].items()
            )
        else:
            assert graph_replays == 0
        runs[name] = {"window_groups": groups, "graph_replays": graph_replays}
    assert any(len(group) > 1 for group in runs["graph_peers"]["window_groups"])
    assert runs["graph_peers"]["window_groups"] == runs["eager_peers"]["window_groups"]
    scheduler.on_serving_stop()
    comparisons = {}
    for request_id in histories:
        windows = {
            name: request_waveforms[request_id]
            for name, request_waveforms in waveforms.items()
        }
        baseline = windows["graph_solo"]
        assert all(
            np.array_equal(left, right)
            for left, right in zip(baseline, windows["graph_solo_repeat"], strict=True)
        ), request_id
        sample_id = next(
            row["sample_id"] for row in manifests if row["directory"] == request_id
        )
        native_waveforms = {
            row["request_count"]: row["waveform_sha256"]
            for row in manifests
            if row["sample_id"] == sample_id
            and captured[row["directory"]] == captured[request_id]
        }
        hashes = {
            name: hashlib.sha256(
                encode_wav(np.concatenate(parts), scheduler.sample_rate)
            ).hexdigest()
            for name, parts in windows.items()
        }
        differences = {}
        for reference_name, alternative_name in (
            ("graph_solo", "graph_peers"),
            ("eager_solo", "eager_peers"),
            ("graph_solo", "eager_solo"),
            ("graph_peers", "eager_peers"),
        ):
            window_differences = []
            for reference_audio, alternative_audio in zip(
                windows[reference_name], windows[alternative_name], strict=True
            ):
                assert reference_audio.shape == alternative_audio.shape
                difference = reference_audio - alternative_audio
                window_differences.append(
                    {
                        "samples": int(difference.size),
                        "changed_samples": int(np.count_nonzero(difference)),
                        "max_abs_difference": float(np.abs(difference).max()),
                        "sum_squared_difference": float(
                            np.square(difference, dtype=np.float64).sum()
                        ),
                    }
                )
            differences[f"{reference_name}_vs_{alternative_name}"] = window_differences
        comparisons[request_id] = {
            "native_sha256_by_concurrency": native_waveforms,
            "replay_sha256": hashes,
            "windows": differences,
        }
    print(
        json.dumps(
            {
                "code2wav_factory": factory_parameters,
                "dtype": str(scheduler.model.dtype),
                "uses_paged_attention": scheduler.incremental_codec.is_paged_attention,
                "runs": runs,
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
