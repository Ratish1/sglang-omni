"""Omni's talker, request by request and step by step, against HF's on the inputs a served request had.

Runs only in the isolated reference environment, on one card:
    PYTHONPATH=<4.57.1 env>:<box dir> CUDA_VISIBLE_DEVICES=<card> python3 talker_parity.py --trace-dir <dir>
        --out <file> [--fp8-talker <repo>] [--zero-multimodal-hidden] [--shard i --shards n]
Reads the traces talker_trace/sitecustomize.py wrote and checks each component against HF 4.57.1's own modules
(the bf16 Instruct checkpoint, or with --fp8-talker the FP8 checkpoint's talker as hf_standard_reference runs it):
1. prompt rows: HF's user segment from the request's talker input ids (zeros at the multimodal rows with
   --zero-multimodal-hidden, as main builds them; otherwise only the text rows are compared) and assistant tail;
2. text rows: HF's trailing text from the thinker tokens the request received (the first token in the prefill, then
   one a step, tts_eos once the thinker finished, tts_pad after);
3. feedback rows: HF's sum of the step's 16 code embeddings (talker codec embedding, then the code predictor's);
4. talker logits: HF's talker teacher-forced on the served prompt rows and decode rows, at the served positions; per
   step the KL divergence of the served distribution (temperature 0.9, suppressed ids masked) from HF's, the rank of
   the served code under HF's and, where the trace has it, the served talker hidden against HF's;
5. code predictor: HF's 15 residual codes by argmax, from HF's talker hidden and, where the trace has it, from the
   served hidden, with the served first code, against the served codes.
Writes one JSON row a request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from hf_standard_reference import MODEL, use_fp8_talker  # noqa: E402

TEMPERATURE = 0.9
ROW_TOLERANCE = 2e-2
# "\n" closes the assistant turn's opening "<|im_start|>assistant\n"; the talker's ids carry tts_pad there.
NEWLINE_TOKEN_ID = 198


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.float() - reference.float()).norm() / reference.float().norm().clamp(min=1e-12))


def build_reference(fp8_talker: str | None):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    if fp8_talker:
        print(f"FP8 talker: {use_fp8_talker(model, fp8_talker)} linears", flush=True)
    else:
        pass
    return model.eval()


@torch.no_grad()
def check_request(model, trace: dict, is_zero_multimodal: bool) -> dict:
    config = model.config
    talker = model.talker
    device = talker.device
    thinker_embed = model.thinker.get_input_embeddings()

    def project_text(token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, device=device).view(1, -1)
        return talker.text_projection(thinker_embed(ids))[0]

    tts_bos, tts_eos, tts_pad = project_text([config.tts_bos_token_id, config.tts_eos_token_id,
                                              config.tts_pad_token_id]).split(1)
    # 1. prompt rows: the user segment as HF's generate builds it, then the 9-row assistant tail
    input_ids = torch.tensor(trace["input_ids"], device=device)
    prompt_rows = trace["prompt_rows"].to(device)
    talker_config = config.talker_config
    thinker_config = config.thinker_config
    multimodal = ((input_ids == thinker_config.audio_token_id) | (input_ids == thinker_config.image_token_id)
                  | (input_ids == thinker_config.video_token_id))
    user_rows = input_ids.shape[0] - 9
    reference_user = torch.empty_like(prompt_rows[:user_rows])
    text_positions = ~multimodal[:user_rows]
    reference_user[text_positions] = project_text(input_ids[:user_rows][text_positions].tolist()).float()
    reference_user[~text_positions] = talker.hidden_projection(
        torch.zeros(int((~text_positions).sum()), thinker_config.text_config.hidden_size, device=device,
                    dtype=talker.dtype)).float()
    compared = slice(None) if is_zero_multimodal else text_positions
    user_error = relative_l2(prompt_rows[:user_rows][compared], reference_user[compared])
    tokens = [token for token in trace["prefill_tokens"] + trace["appended_tokens"]
              if token is not None and token != config.im_end_token_id]
    assistant_text = project_text([config.im_start_token_id, config.assistant_token_id, NEWLINE_TOKEN_ID, tokens[0]])
    codec_special = talker.get_input_embeddings()(torch.tensor(
        [talker_config.codec_nothink_id, talker_config.codec_think_bos_id, talker_config.codec_think_eos_id,
         talker_config.speaker_id["ethan"], talker_config.codec_pad_id, talker_config.codec_bos_id], device=device))
    reference_tail = torch.cat([assistant_text[:3], tts_pad.expand(4, -1), tts_bos, assistant_text[3:4]]) + torch.cat(
        [torch.zeros(3, codec_special.shape[-1], device=device, dtype=codec_special.dtype), codec_special])
    tail_error = relative_l2(prompt_rows[user_rows:], reference_tail)
    # 2. text rows: tokens after the first, tts_eos once the thinker finished, then pads
    expected_text = project_text(tokens[1:]) if len(tokens) > 1 else tts_eos[:0]
    if trace["thinker_done"]:
        expected_text = torch.cat([expected_text, tts_eos])
    else:
        pass
    served_text = torch.stack(trace["text_rows"]).to(device) if trace["text_rows"] else expected_text[:0]
    text_errors = []
    for step, row in enumerate(served_text):
        reference = expected_text[step] if step < expected_text.shape[0] else tts_pad[0]
        text_errors.append(relative_l2(row, reference))
    bad_text_steps = [step for step, error in enumerate(text_errors) if error > ROW_TOLERANCE]
    pad_from = next((step for step, is_pad in enumerate(trace["text_is_pad"]) if is_pad), None)
    # 3. feedback rows: the 16 code embeddings of the step before
    codes = torch.stack(trace["codes"]).to(device)
    predictor_embeddings = talker.code_predictor.get_input_embeddings()
    feedback_errors = []
    for step, row in enumerate(trace["feedback_rows"]):
        step_codes = codes[step]
        reference = talker.get_input_embeddings()(step_codes[:1]) + sum(
            predictor_embeddings[index](step_codes[index + 1:index + 2]) for index in range(15))
        feedback_errors.append(relative_l2(row.to(device), reference[0]))
    bad_feedback_steps = [step for step, error in enumerate(feedback_errors) if error > ROW_TOLERANCE]
    # 4. and 5. the talker teacher-forced on the served rows, and the code predictor on its hidden
    from transformers import DynamicCache

    vocab = talker_config.text_config.vocab_size
    suppressed = torch.zeros(vocab, dtype=torch.bool, device=device)
    suppressed[vocab - 1024:] = True
    suppressed[talker_config.codec_eos_token_id] = False
    length = prompt_rows.shape[0]
    if trace["mrope_positions"] is None:
        positions = torch.arange(length).view(1, -1).expand(3, -1)
    else:
        # note (ratish): both sides truncate the float positions, HF into its int64 position ids and the
        # server's forward batch by casting to int64; a decode step continues from the prompt's largest.
        positions = trace["mrope_positions"].long()
    next_position = int(positions.max()) + 1
    cache = DynamicCache()
    output = talker.model(inputs_embeds=prompt_rows[None].to(talker.dtype), position_ids=positions[:, None].to(device),
                          past_key_values=cache, use_cache=True)
    hidden = output.last_hidden_state[:, -1:]
    decode_rows = [feedback + text for feedback, text in zip(trace["feedback_rows"], trace["text_rows"])]
    served_hidden = trace.get("hidden") or [None] * len(trace["logits"])
    # note (ratish): a trace without hidden states predates the kept prefill logits; its step 0 holds SGLang's sampled
    # probabilities.
    is_step0_probabilities = "hidden" not in trace
    divergences, ranks, predictor_matches, served_predictor_matches, hidden_errors, top_equal = [], [], [], [], [], []

    def predictor_codes(talker_hidden: torch.Tensor, code0: int) -> torch.Tensor:
        code0_embed = talker.get_input_embeddings()(torch.tensor([[code0]], device=device))
        return talker.code_predictor.generate(
            inputs_embeds=torch.cat([talker_hidden, code0_embed], 1), max_new_tokens=15, do_sample=False,
            top_k=None, top_p=None)[0, -15:]

    for step, served_logits in enumerate(trace["logits"]):
        reference_logits = talker.codec_head(hidden)[0, 0].float().masked_fill(suppressed, float("-inf"))
        code0 = int(trace["sampled"][step])
        if not (step == 0 and is_step0_probabilities):
            served = served_logits.to(device).float().masked_fill(suppressed, float("-inf"))
            served_log = torch.log_softmax(served / TEMPERATURE, dim=-1)
            reference_log = torch.log_softmax(reference_logits / TEMPERATURE, dim=-1)
            support = torch.isfinite(served_log)
            divergences.append(float((served_log.exp() * (served_log - reference_log))[support].sum()))
            top_equal.append(int(served.argmax()) == int(reference_logits.argmax()))
        else:
            pass
        ranks.append(int((reference_logits > reference_logits[code0]).sum()))
        predictor_matches.append(int((predictor_codes(hidden, code0) == codes[step, 1:]).sum()))
        if served_hidden[step] is not None:
            step_hidden = served_hidden[step].to(device, talker.dtype).view(1, 1, -1)
            hidden_errors.append(relative_l2(step_hidden[0, 0], hidden[0, 0]))
            served_predictor_matches.append(int((predictor_codes(step_hidden, code0) == codes[step, 1:]).sum()))
        else:
            pass
        if step >= len(decode_rows):
            break
        else:
            pass
        position = torch.full((3, 1, 1), next_position + step, device=device)
        output = talker.model(inputs_embeds=decode_rows[step].to(device, talker.dtype).view(1, 1, -1),
                              position_ids=position, past_key_values=cache, use_cache=True,
                              cache_position=torch.tensor([length + step], device=device))
        hidden = output.last_hidden_state[:, -1:]
    steps = len(divergences)
    late = divergences[steps // 2:] or [0.0]
    frames = len(trace["logits"])
    return dict(
        request_id=trace["request_id"], finish_reason=trace["finish_reason"], frames=len(trace["codes"]),
        thinker_tokens=len(tokens), thinker_done_at_prefill=trace["thinker_done_at_prefill"],
        prompt_user_rows_error=user_error, prompt_tail_error=tail_error,
        text_rows=len(text_errors), text_rows_max_error=max(text_errors, default=0.0), bad_text_steps=bad_text_steps[:20],
        first_pad_step=pad_from, expected_text_rows=int(expected_text.shape[0]),
        feedback_max_error=max(feedback_errors, default=0.0), bad_feedback_steps=bad_feedback_steps[:20],
        kl_mean=sum(divergences) / steps, kl_max=max(divergences), kl_first_quarter=sum(divergences[:steps // 4 or 1])
        / (steps // 4 or 1), kl_second_half=sum(late) / len(late), top1_agreement=sum(top_equal) / steps,
        served_code_rank_median=sorted(ranks)[frames // 2], served_code_rank_over_50=sum(rank >= 50 for rank in ranks),
        predictor_codes_equal=sum(predictor_matches) / (15 * frames),
        predictor_codes_equal_on_served_hidden=(
            sum(served_predictor_matches) / (15 * len(served_predictor_matches)) if served_predictor_matches else None),
        hidden_error_mean=sum(hidden_errors) / len(hidden_errors) if hidden_errors else None,
        hidden_error_max=max(hidden_errors, default=None), step0_excluded=is_step0_probabilities,
        kl_steps=[round(divergence, 4) for divergence in divergences], top1_steps=[int(equal) for equal in top_equal],
        rank_steps=ranks, predictor_steps=predictor_matches, served_predictor_steps=served_predictor_matches,
        hidden_error_steps=[round(error, 4) for error in hidden_errors])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp8-talker")
    parser.add_argument("--zero-multimodal-hidden", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    model = build_reference(args.fp8_talker)
    rows = []
    for path in sorted(args.trace_dir.glob("*.pt"))[args.shard::args.shards]:
        trace = torch.load(path, weights_only=False)
        if not trace.get("codes") or trace.get("prompt_rows") is None:
            continue
        else:
            pass
        rows.append(check_request(model, trace, args.zero_multimodal_hidden))
        print(json.dumps(rows[-1]), flush=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
