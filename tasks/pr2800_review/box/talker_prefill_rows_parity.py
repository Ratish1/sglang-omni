"""#2861's talker prefill against the one HF's generate builds, with HF's own thinker rows in the first chunk.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_prefill_rows_parity.py --arm phu
    --out /data/pr2800/<run>
The component accuracy check of 10-11 (test_talker_prefill.py on test/qwen3-omni-component-accuracy) in one driver:
HF's Qwen3OmniMoeForConditionalGeneration.generate runs in bf16 on card 0 up to its talker call, which is captured
instead of run (modeling_qwen3_omni_moe.py:3981-4064), and the thinker's accept_hidden_layer rows at the prompt come
from its generate (hidden_states[0][accept_hidden_layer]). The arm's TalkerPrefillBuilder then builds from the same
prompt ids, thinker tokens and talker projection modules, its first chunk carrying those rows at the prompt's
multimodal positions as the thinker's hand-off does. Every row must match HF's talker input: multimodal, text and the
9-row assistant tail. Prompts: the first MMSU_SAMPLES mmsu-ci-2000 questions with stage 6's talker prompt, and the
first SEEDTTS_SAMPLES SeedTTS EN voice-clone requests as the benchmark builds them (reference audio first). Writes
<out>/talker_prefill_rows.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import torch

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SPEAKER = "Ethan"
MMSU_SAMPLES = 10
SEEDTTS_SAMPLES = 10
MAX_TOKENS = 64
MATCH_RELATIVE_L2 = 1e-3
MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)


class TalkerInputsCaptured(Exception):
    pass


class TalkerProjections(torch.nn.Module):
    """The attributes TalkerPrefillBuilder reads from the talker, served by the HF talker's own modules."""

    def __init__(self, talker: torch.nn.Module) -> None:
        super().__init__()
        self.model = talker.model
        self.text_projection = talker.text_projection
        self.hidden_projection = talker.hidden_projection
        self.activation_dtype = talker.model.codec_embedding.weight.dtype

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    if not reference.numel():
        return 0.0
    else:
        pass
    return float((value.double() - reference.double()).norm() / reference.double().norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[args.arm][0])
    sys.path.insert(0, str(arm_tree))
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.dataset.seedtts import load_seedtts_samples
    from sglang_omni.models.qwen3_omni.components.talker_prefill import (
        TalkerPrefillBuilder,
    )
    from sglang_omni.pipeline.stage.stream_queue import StreamItem
    from sglang_omni.proto.request import OmniRequest, StagePayload

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    config = model.config
    accept_layer = config.talker_config.accept_hidden_layer
    builder = TalkerPrefillBuilder(
        model=TalkerProjections(model.talker),
        model_path=MODEL,
        audio_token_id=config.thinker_config.audio_token_id,
        image_token_id=config.thinker_config.image_token_id,
        video_token_id=config.thinker_config.video_token_id,
        tts_bos_token_id=config.tts_bos_token_id,
        tts_eos_token_id=config.tts_eos_token_id,
        tts_pad_token_id=config.tts_pad_token_id,
        im_start_token_id=config.im_start_token_id,
        im_end_token_id=config.im_end_token_id,
        system_token_id=config.system_token_id,
        user_token_id=config.user_token_id,
        assistant_token_id=config.assistant_token_id,
        codec_bos_id=config.talker_config.codec_bos_id,
        codec_nothink_id=config.talker_config.codec_nothink_id,
        codec_think_bos_id=config.talker_config.codec_think_bos_id,
        codec_think_eos_id=config.talker_config.codec_think_eos_id,
        codec_pad_id=config.talker_config.codec_pad_id,
        speaker_map=config.talker_config.speaker_id,
    )
    multimodal_ids = torch.tensor(
        [
            config.thinker_config.audio_token_id,
            config.thinker_config.image_token_id,
            config.thinker_config.video_token_id,
        ]
    )
    requests = [
        (f"mmsu {sample.sample_id}", sample.audio_path, MMSU_TTS_PROMPT)
        for sample in load_mmsu_samples(max_samples=MMSU_SAMPLES, repo_id="zhaochenyang20/mmsu-ci-2000")
    ] + [
        (
            f"seedtts {sample.sample_id}",
            sample.ref_audio,
            f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
            f"Now please read the following text out loud in the same voice and style: "
            f"{sample.target_text}",
        )
        for sample in load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", SEEDTTS_SAMPLES, split="en")
    ]

    records = []
    for request_id, audio_path, text in requests:
        waveform, _ = librosa.load(audio_path, sr=16000)
        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=prompt_text, audio=[waveform], return_tensors="pt", padding=True)
        inputs = {
            name: tensor.to("cuda:0", dtype=torch.bfloat16) if tensor.is_floating_point() else tensor.to("cuda:0")
            for name, tensor in inputs.items()
        }
        captured: dict[str, torch.Tensor] = {}
        thinker_generate = model.thinker.generate

        def record_thinker(*thinker_args, **thinker_kwargs):
            result = thinker_generate(*thinker_args, **thinker_kwargs)
            captured["sequences"] = result.sequences
            captured["prompt_hidden"] = result.hidden_states[0][accept_layer][0]
            return result

        def capture_talker(**talker_kwargs):
            captured.update(talker_kwargs)
            raise TalkerInputsCaptured

        model.thinker.generate = record_thinker
        model.talker.generate = capture_talker
        try:
            with torch.inference_mode():
                model.generate(**inputs, speaker=SPEAKER, return_audio=True, thinker_max_new_tokens=MAX_TOKENS)
        except TalkerInputsCaptured:
            pass
        finally:
            del model.thinker.generate
            del model.talker.generate
        prompt_ids = inputs["input_ids"][0].cpu()
        positions = torch.isin(prompt_ids, multimodal_ids).nonzero(as_tuple=True)[0]
        generated = captured["sequences"][0, prompt_ids.shape[0]:].tolist()
        chunks = [
            StreamItem(chunk_id=index, data=None, from_stage="thinker", metadata={"token_id": token})
            for index, token in enumerate(generated)
        ]
        chunks[0].data = captured["prompt_hidden"][positions.to("cuda:0")].cpu()
        chunks[0].metadata["prompt_hidden_positions"] = positions.tolist()
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs={}, params={"speaker": SPEAKER}),
            data={"prompt": {"input_ids": prompt_ids}},
        )
        with torch.inference_mode():
            whole = builder.build_prompt_prefill(payload, chunks, thinker_done=True)
        reference_embeds = captured["inputs_embeds"][0]
        reference_ids = captured["talker_input_ids"][0].cpu()
        multimodal = torch.isin(reference_ids, multimodal_ids)
        tail = torch.zeros_like(multimodal)
        tail[-9:] = True
        text_rows = ~multimodal & ~tail
        omni_embeds = whole["input_embeds"]
        record = dict(
            request=request_id,
            rows=int(reference_ids.shape[0]),
            multimodal_rows=int(multimodal.sum()),
            input_ids_equal=torch.equal(whole["input_ids"].cpu(), reference_ids),
            multimodal_rows_relative_l2=relative_l2(omni_embeds[multimodal], reference_embeds[multimodal]),
            text_rows_relative_l2=relative_l2(omni_embeds[text_rows], reference_embeds[text_rows]),
            tail_rows_relative_l2=relative_l2(omni_embeds[tail], reference_embeds[tail]),
        )
        records.append(record)
        print(record, flush=True)
    failures = [
        record["request"]
        for record in records
        if not record["input_ids_equal"]
        or max(record["multimodal_rows_relative_l2"], record["text_rows_relative_l2"], record["tail_rows_relative_l2"])
        > MATCH_RELATIVE_L2
    ]
    summary = dict(arm=args.arm, requests=len(records), failures=failures)
    print(f"summary {summary}", flush=True)
    (args.out / "talker_prefill_rows.json").write_text(json.dumps(dict(summary=summary, records=records), indent=1))


if __name__ == "__main__":
    main()
