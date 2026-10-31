"""Teacher-force HF's Qwen3-Omni talker on queue104's captures, step by step.

For each capture: HF FP32 (truth) and HF BF16 (floor) talkers run over the capture's own prefill and decode rows; per
frame, omni's captured hidden and the floor's are compared to the truth's (relative L2); the first code's served
distribution (repetition penalty on the frame's earlier first codes, the top 1,024 ids but EOS suppressed, temperature,
top-k) is built from omni's hidden and from the truth's, giving per frame P(EOS), the KL from truth to omni, top-1
agreement and the truth's probability of the code omni sampled. The decode rows are rebuilt with HF's modules from the
codes and the reply token ids (codec embeddings of the previous frame plus the next text row: reply tokens after the
first, tts_eos, then tts_pad), and the code predictor's argmax departures from HF FP32 are counted against HF BF16's.

usage: python3 teacher_force.py OUT.json CAPTURE.pt [CAPTURE.pt ...]
"""

import copy
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda:0"
OUT = sys.argv[1]
PATHS = sys.argv[2:]

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    dtype=torch.float32,
    device_map="cpu",
    enable_audio_output=True,
    experts_implementation="eager",
).eval()
config = model.config
truth = model.talker.to(DEVICE, torch.float32)
embedding = model.thinker.get_input_embeddings()
codec_eos = config.talker_config.codec_eos_token_id
vocab = config.talker_config.text_config.vocab_size
suppressed = torch.tensor(
    [i for i in range(vocab - 1024, vocab) if i != codec_eos], device=DEVICE
)
special = torch.tensor([config.tts_eos_token_id, config.tts_pad_token_id])


def relative_l2_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1).clamp_min(
        1e-12
    )


def served_probs(
    logits: torch.Tensor, first_codes: torch.Tensor, sampling: dict
) -> torch.Tensor:
    """The first code's sampling distribution per frame as the talker serves it."""
    logits = logits.float().clone()
    penalty = float(sampling["repetition_penalty"])
    for frame in range(1, logits.shape[0]):
        seen = first_codes[:frame].unique()
        row = logits[frame, seen]
        logits[frame, seen] = torch.where(row > 0, row / penalty, row * penalty)
    logits[:, suppressed] = float("-inf")
    logits = logits / float(sampling["temperature"])
    probs = torch.softmax(logits, dim=-1)
    top_k = int(sampling["top_k"])
    kept = probs.topk(top_k, dim=-1)
    filtered = torch.zeros_like(probs).scatter_(-1, kept.indices, kept.values)
    return filtered / filtered.sum(-1, keepdim=True)


def codec_feedback(talker: torch.nn.Module, codes: torch.Tensor) -> torch.Tensor:
    rows = talker.model.get_input_embeddings()(codes[:, 0])
    for group, table in enumerate(talker.code_predictor.get_input_embeddings()):
        rows = rows + table(codes[:, group + 1])
    return rows


def predictor_logits(
    talker: torch.nn.Module, hidden: torch.Tensor, codes: torch.Tensor
) -> torch.Tensor:
    tables = talker.code_predictor.get_input_embeddings()
    inputs = torch.stack(
        [hidden, talker.model.get_input_embeddings()(codes[:, 0])]
        + [tables[group](codes[:, group + 1]) for group in range(len(tables) - 1)],
        dim=1,
    )
    states = talker.code_predictor.model(inputs_embeds=inputs).last_hidden_state
    return torch.stack(
        [
            head(states[:, group + 1])
            for group, head in enumerate(talker.code_predictor.lm_head)
        ],
        dim=1,
    ).float()


floor = copy.deepcopy(truth).to(torch.bfloat16)
# note (ratish): a Transformers load in bf16 keeps its rotary tables in float32; the cast copy would round them.
for name, buffer in truth.named_buffers():
    if name.endswith("inv_freq"):
        owner, _, attribute = name.rpartition(".")
        setattr(floor.get_submodule(owner), attribute, buffer.detach().clone())
    else:
        pass

results = []
with torch.inference_mode():
    for path in PATHS:
        record = torch.load(path, map_location="cpu")
        codes = record["codes"].to(DEVICE)
        hidden = record["hidden"].to(DEVICE)
        prefill = record["prefill_input_embeds"].to(DEVICE)
        decode = record["decode_input_embeds"].to(DEVICE)
        frames = codes.shape[0]
        reply = [t for t in record["thinker_token_ids"] if t != config.im_end_token_id]
        text_rows_count = max(len(reply) - 1, 0)

        rows = truth.text_projection(
            embedding(torch.tensor(reply[1:], dtype=torch.long)).to(DEVICE)
        )
        eos_row, pad_row = truth.text_projection(embedding(special).to(DEVICE))
        padding = pad_row.expand(max(decode.shape[0] - rows.shape[0] - 1, 0), -1)
        text_rows = torch.cat([rows, eos_row[None], padding])[: decode.shape[0]]
        expected = codec_feedback(truth, codes[: decode.shape[0]]) + text_rows
        decode_dev = relative_l2_rows(decode, expected)

        sequence = torch.cat([prefill, decode])
        steps = torch.arange(
            prefill.shape[0] - 1, prefill.shape[0] - 1 + frames, device=DEVICE
        )
        truth_hidden = truth.model(
            inputs_embeds=sequence[None].float()
        ).last_hidden_state[0, steps]
        floor_hidden = floor.model(
            inputs_embeds=sequence[None].bfloat16()
        ).last_hidden_state[0, steps]
        omni_dev = relative_l2_rows(hidden, truth_hidden)
        floor_dev = relative_l2_rows(floor_hidden, truth_hidden)

        truth_logits = truth.codec_head(truth_hidden)
        omni_logits = truth.codec_head(hidden.float())
        first_codes = codes[:, 0]
        p_truth = served_probs(truth_logits, first_codes, record["sampling"])
        p_omni = served_probs(omni_logits, first_codes, record["sampling"])
        kl = (
            p_truth * (p_truth.clamp_min(1e-30).log() - p_omni.clamp_min(1e-30).log())
        ).sum(-1)
        sampled_truth_p = p_truth.gather(1, first_codes[:, None])[:, 0]
        sampled_omni_p = p_omni.gather(1, first_codes[:, None])[:, 0]

        truth_pred = predictor_logits(truth, hidden.float(), codes)
        floor_pred = predictor_logits(floor, hidden.bfloat16(), codes)
        choice = truth_pred.argmax(-1)
        omni_departures = int((choice != codes[:, 1:]).sum())
        floor_departures = int((choice != floor_pred.argmax(-1)).sum())

        # note (ratish): frame f's code is sampled from the hidden after decode row f - 1; the tts_eos row is decode
        # row text_rows_count, so frames from text_rows_count + 1 on see the text's end.
        end = min(text_rows_count + 1, frames)
        after = slice(end, frames)
        result = {
            "path": path,
            "request_text": (
                record["request_text"][-1][-120:] if record["request_text"] else ""
            ),
            "frames": frames,
            "reply_tokens": len(reply),
            "finished_reason": record["finished_reason"],
            "batch_size_median": int(record["batch_sizes"].float().median()),
            "decode_input_dev_max": (
                decode_dev.max().item() if decode_dev.numel() else 0.0
            ),
            "hidden_dev_omni_mean": omni_dev.mean().item(),
            "hidden_dev_floor_mean": floor_dev.mean().item(),
            "hidden_dev_ratio_max": (omni_dev / floor_dev.clamp_min(1e-6)).max().item(),
            "hidden_dev_omni_max": omni_dev.max().item(),
            "hidden_dev_omni_argmax_frame": int(omni_dev.argmax()),
            "code0_top1_agreement": (omni_logits.argmax(-1) == truth_logits.argmax(-1))
            .float()
            .mean()
            .item(),
            "kl_mean": kl.mean().item(),
            "kl_max": kl.max().item(),
            "kl_argmax_frame": int(kl.argmax()),
            "sampled_logp_truth": sampled_truth_p.clamp_min(1e-30).log().sum().item(),
            "sampled_logp_omni": sampled_omni_p.clamp_min(1e-30).log().sum().item(),
            "eos_frame_text_end": end,
            "p_eos_truth_after_end_mean": (
                p_truth[after, codec_eos].mean().item() if frames > end else None
            ),
            "p_eos_omni_after_end_mean": (
                p_omni[after, codec_eos].mean().item() if frames > end else None
            ),
            "p_eos_truth_after_end_max": (
                p_truth[after, codec_eos].max().item() if frames > end else None
            ),
            "p_eos_omni_after_end_max": (
                p_omni[after, codec_eos].max().item() if frames > end else None
            ),
            "p_eos_truth_first_frames_after_end": p_truth[
                end : end + 12, codec_eos
            ].tolist(),
            "p_eos_omni_first_frames_after_end": p_omni[
                end : end + 12, codec_eos
            ].tolist(),
            "predictor_departures_omni": omni_departures,
            "predictor_departures_floor": floor_departures,
            "predictor_codes": int(choice.numel()),
        }
        results.append(result)
        print(
            json.dumps(
                {
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in result.items()
                    if not k.startswith("p_eos_") or "mean" in k or "max" in k
                }
            ),
            flush=True,
        )
        Path(OUT).write_text(json.dumps(results, indent=1))
