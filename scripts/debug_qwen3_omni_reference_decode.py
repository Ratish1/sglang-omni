# SPDX-License-Identifier: Apache-2.0
"""Replay a captured Talker input with the checkpoint's decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile
import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    register_checkpoint_conversion_mapping(
        config.talker_config.model_type,
        get_checkpoint_conversion_mapping("qwen3_omni_moe"),
    )
    talker, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config.talker_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    assert not any(name.startswith("model.") for name in loading["unexpected_keys"])
    talker.eval()
    before = torch.load(
        arguments.trace_directory / "0000-before_prefill.pt", weights_only=True
    )
    first = torch.load(
        arguments.trace_directory / "0000-post_prefill.pt", weights_only=True
    )
    decode = torch.load(
        arguments.trace_directory / "0001-before_decode.pt", weights_only=True
    )
    embeddings = before["prefill_embeds"].to(device).unsqueeze(0)
    trailing_text = before["trailing_text_embeds"].to(device)
    padding = before["tts_pad_embed"].to(device)
    first_codec = first["codec_ids"].to(device)
    sampling_seed = decode["sampling_seed"].to(device)
    repetition_mask = torch.zeros(1, 3072, dtype=torch.bool, device=device)
    suppression = torch.arange(3072, device=device) >= 2048
    suppression[config.talker_config.codec_eos_token_id] = False
    cache = None
    codec_frames: list[torch.Tensor] = []
    decisions: list[dict[str, int | list[int]]] = []
    with torch.inference_mode():
        for step in range(4096):
            first_position = 0 if step == 0 else before["positions"].numel() + step - 1
            positions = torch.arange(
                first_position, first_position + embeddings.shape[1], device=device
            )
            outputs = talker.model(
                inputs_embeds=embeddings,
                position_ids=positions.reshape(1, 1, -1).expand(3, 1, -1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[:, -1:]
            logits = talker.codec_head(hidden_states[:, -1]).float()
            penalized = torch.where(logits > 0, logits / 1.05, logits * 1.05)
            logits = torch.where(repetition_mask, penalized, logits)
            logits = logits.masked_fill(suppression, float("-inf"))
            probabilities = (logits / 0.9).softmax(-1)
            sampled = top_k_top_p_min_p_sampling_from_probs_torch(
                probabilities,
                torch.tensor([50], device=device),
                torch.tensor([1.0], device=device),
                torch.tensor([0.0], device=device),
                False,
                sampling_seed,
                positions[-1:],
            ).long()
            if step == 0:
                assert sampled.item() == first_codec[0, 0].item()
            else:
                pass
            decisions.append(dict(step=step, first_code=sampled.item()))
            if sampled.item() == config.talker_config.codec_eos_token_id:
                break
            else:
                pass
            first_embedding = talker.model.codec_embedding(sampled[:, None])
            residual = talker.code_predictor.generate(
                inputs_embeds=torch.cat((hidden_states, first_embedding), dim=1),
                max_new_tokens=config.talker_config.num_code_groups - 1,
                do_sample=False,
            )
            codes = torch.cat((sampled[:, None], residual), dim=1)
            codec_frames.append(codes)
            embedding_rows = [first_embedding]
            for index, embedding in enumerate(
                talker.code_predictor.get_input_embeddings()
            ):
                embedding_rows.append(embedding(residual[:, index : index + 1]))
            feedback = torch.cat(embedding_rows, dim=1).sum(1)
            text = trailing_text[step] if step < len(trailing_text) else padding
            embeddings = (feedback + text).unsqueeze(1)
            repetition_mask[0, sampled.item()] = True
            if step % 25 == 0:
                print(json.dumps(decisions[-1]), flush=True)
            else:
                pass
        else:
            raise RuntimeError("Reference Talker exhausted the production codec budget")
        codes = torch.cat(codec_frames, dim=0).T.unsqueeze(0)
        arguments.output_directory.mkdir(exist_ok=False)
        torch.save(codes.cpu(), arguments.output_directory / "codes.pt")
        (arguments.output_directory / "decisions.json").write_text(
            json.dumps(decisions, indent=2)
        )
        vocoder, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
            arguments.model_path,
            config=config.code2wav_config,
            dtype=torch.bfloat16,
            device_map={"": arguments.gpu},
            key_mapping={r"^code2wav\.": ""},
            output_loading_info=True,
        )
        assert not loading["missing_keys"], loading["missing_keys"]
        assert not loading["mismatched_keys"], loading["mismatched_keys"]
        waveform = vocoder.eval()(codes).float().flatten().cpu().numpy()
        soundfile.write(arguments.output_directory / "reference.wav", waveform, 24000)
        print(
            json.dumps(
                dict(codec_frames=codes.shape[-1], duration_s=len(waveform) / 24000)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
