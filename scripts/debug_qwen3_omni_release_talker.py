# SPDX-License-Identifier: Apache-2.0
"""Continue the checkpoint from the recorded first frame and text rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_held_talker import filtered_distribution
from sglang_omni.client.audio import encode_wav

MAX_FRAMES = 128


def continue_codec_history(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    trace_directory: Path,
    forced_second_code: int | None,
) -> tuple[torch.Tensor, int]:
    device = reference.device
    prefill = torch.load(
        trace_directory / "backbone-prefill-000.pt",
        map_location=device,
        weights_only=True,
    )
    finish = torch.load(
        trace_directory / "backbone-finish-000.pt",
        map_location=device,
        weights_only=True,
    )
    sampling = torch.load(
        trace_directory / "decode-sampling-001.pt",
        map_location=device,
        weights_only=True,
    )
    opening = torch.load(
        trace_directory / "codec-000.pt", map_location=device, weights_only=True
    )
    assert finish["thinker_done"]
    positions = prefill["positions"]
    assert prefill["rotary_positions"] is None or torch.equal(
        prefill["rotary_positions"], positions.view(1, -1).expand(3, -1)
    )
    assert torch.equal(positions, torch.arange(len(positions), device=device))
    decode_inputs = [
        torch.load(path, map_location=device, weights_only=True)
        for path in sorted(trace_directory.glob("backbone-decode-*.pt"))
    ]
    for offset, recorded in enumerate(decode_inputs):
        assert int(recorded["position"]) == len(positions) + offset
        rotary = recorded["rotary_positions"]
        assert rotary is None or (rotary == recorded["position"]).all()
    future_text = [
        recorded["text_row"].reshape(-1)
        for recorded in decode_inputs
        if not recorded["uses_pad"]
    ]
    future_text.extend(finish["pending_rows"])
    pad_embedding = finish["pad_embedding"].reshape(-1)
    cache = reference.model(
        inputs_embeds=prefill["embeddings"].unsqueeze(0),
        position_ids=positions.reshape(1, 1, -1).expand(4, 1, -1),
        use_cache=True,
    ).past_key_values
    frames = [opening["codec_ids"].reshape(-1).long()]
    feedback = opening["feedback_embedding"].reshape(-1)
    repeated = torch.zeros_like(sampling["repetition_mask"])
    repeated[frames[0][0]] = True
    eos_token = reference.config.codec_eos_token_id
    natural_second_code = None
    for decode_index in range(min(MAX_FRAMES, prefill["max_new_tokens"]) - 1):
        text_row = (
            future_text[decode_index]
            if decode_index < len(future_text)
            else pad_embedding
        )
        position = sampling["position"].new_tensor(len(positions) + decode_index)
        assert cache.get_seq_length() == int(position)
        output = reference.model(
            inputs_embeds=(feedback + text_row).view(1, 1, -1),
            position_ids=position.reshape(1, 1, 1).expand(4, 1, 1),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        query = output.last_hidden_state[:, -1:]
        logits = reference.codec_head(query).float().reshape(-1)
        sampling["repetition_mask"] = repeated
        sampling["position"] = position
        _, sampled_code = filtered_distribution(logits, sampling)
        if decode_index == 0:
            natural_second_code = sampled_code
            main_code = (
                sampled_code if forced_second_code is None else forced_second_code
            )
        else:
            main_code = sampled_code
        repeated[main_code] = True
        if main_code == eos_token:
            assert natural_second_code is not None
            terminal = frames[0].new_zeros(frames[0].shape)
            terminal[0] = eos_token
            return torch.stack([*frames, terminal]).cpu(), natural_second_code
        else:
            pass
        main_token = frames[0].new_tensor([[main_code]])
        main_embedding = reference.model.codec_embedding(main_token)
        residual_codes = reference.code_predictor.generate(
            inputs_embeds=torch.cat((query, main_embedding), dim=1),
            max_new_tokens=reference.config.num_code_groups - 1,
            do_sample=False,
            return_dict_in_generate=True,
        ).sequences
        assert residual_codes.shape == (1, reference.config.num_code_groups - 1)
        frames.append(torch.cat((main_token, residual_codes), dim=1).reshape(-1))
        feedback = main_embedding
        for group_index, embedding in enumerate(
            reference.code_predictor.model.codec_embedding
        ):
            feedback = feedback + embedding(
                residual_codes[:, group_index : group_index + 1]
            )
        feedback = feedback.reshape(-1)
    raise RuntimeError(f"Reference did not reach codec EOS in {len(frames)} frames")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir()
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    talker_config = config.talker_config
    register_checkpoint_conversion_mapping(
        talker_config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=talker_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    reference.eval()
    code_histories = {}
    reports = {}
    for arm in ("bad", "good"):
        trace_directory = arguments.witness_directory / "held-native" / arm
        native_codes = torch.stack(
            [
                torch.load(path, weights_only=True)["codec_ids"].reshape(-1)
                for path in sorted(trace_directory.glob("codec-*.pt"))
            ]
        )
        held_second_code = int(native_codes[1, 0])
        natural_codes, natural_second_code = continue_codec_history(
            reference, trace_directory, None
        )
        code_histories[f"{arm}-native"] = native_codes
        code_histories[f"{arm}-reference-natural"] = natural_codes
        if natural_second_code == held_second_code:
            forced_codes = natural_codes
        else:
            forced_codes, repeated_second_code = continue_codec_history(
                reference, trace_directory, held_second_code
            )
            assert repeated_second_code == natural_second_code
        code_histories[f"{arm}-reference-forced"] = forced_codes
        reports[arm] = {
            "native_second_code": held_second_code,
            "reference_natural_second_code": natural_second_code,
            "forced_arm_reuses_natural_codes": natural_second_code == held_second_code,
        }
    del reference
    vocoder, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    vocoder.eval()
    with wave.open(
        str(arguments.witness_directory / "held-native-bad-response" / "held.wav")
    ) as native_waveform:
        sample_rate = native_waveform.getframerate()
    waveforms = {}
    for label, codec_ids in code_histories.items():
        assert codec_ids[-1, 0] == talker_config.codec_eos_token_id
        assert not (codec_ids[:-1, 0] == talker_config.codec_eos_token_id).any()
        codec_ids_without_eos = codec_ids[:-1].T.unsqueeze(0).to(vocoder.device)
        waveform = (
            vocoder.chunked_decode(codec_ids_without_eos).float().reshape(-1).cpu()
        )
        assert waveform.isfinite().all()
        wav_bytes = encode_wav(waveform.numpy(), sample_rate)
        (arguments.output_directory / f"{label}.wav").write_bytes(wav_bytes)
        waveforms[label] = {
            "codec_frames_without_eos": codec_ids_without_eos.shape[-1],
            "duration_seconds": waveform.numel() / sample_rate,
            "waveform_sha256": hashlib.sha256(wav_bytes).hexdigest(),
        }
    torch.save(code_histories, arguments.output_directory / "code-histories.pt")
    (arguments.output_directory / "render.json").write_text(
        json.dumps({"continuations": reports, "waveforms": waveforms}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
else:
    pass
