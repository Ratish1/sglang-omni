"""What BF16 arithmetic itself costs per Qwen3-Omni sub-component: HF BF16 against HF FP32, local and accumulated.

The truth is HF's FP32 model on the host (no TF32). Hooks record every target module's inputs and output over one
forward: audio tower, thinker (prompt plus the target text teacher-forced), talker (HF's prefill built from the FP32
thinker, a seeded FP32 draw of the codes teacher-forced), code predictor, code2wav.

Local: each target module, copied to the card in BF16 as a Transformers BF16 load holds it, gets the FP32 inputs (cast
to BF16) and its output is compared to the FP32 output. This is the error the sub-component adds by itself.
Accumulated: the BF16 chain from the BF16 audio tower through every thinker layer, then the BF16 talker over the BF16
prefill and decode rows, with the FP32 codes teacher-forced, compared to FP32 at every layer. Rows are grouped (audio,
prompt text, reply; prefill, decode). Output distributions: thinker logits at temperature 0.7; talker first code as
served (repetition penalty, suppressed ids, temperature, top-k); residual codes with HF's top-k 50, top-p 0.8 and
argmax agreement. Match is 1 - relative L2 for tensors, 1 - total variation for distributions.

Each module is also replayed in FP32 on the card, the check that the host truth's own platform error sits far below
what BF16 costs. usage: python3 hf_numerics_ledger.py OUT.json SAMPLE_IDS_FILE COUNT
"""

import collections
import copy
import json
import re
import sys
from pathlib import Path

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
DEVICE = "cuda:0"
THINKER_TEMPERATURE = 0.7
SEED = 20261102

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    return None


def cast(value, device, dtype):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(cast(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [cast(item, device, dtype) for item in value]
    if isinstance(value, dict):
        return {key: cast(item, device, dtype) for key, item in value.items()}
    return value


def detached(value):
    if torch.is_tensor(value):
        return value.detach().clone().cpu()
    if isinstance(value, tuple):
        return tuple(detached(item) for item in value)
    if isinstance(value, list):
        return [detached(item) for item in value]
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    return value


class Recorder:
    """Records (args, kwargs, output) of every matching module over the forwards run inside it."""

    def __init__(self, root: torch.nn.Module, pattern: str) -> None:
        self.records = collections.defaultdict(list)
        self.handles = []
        for name, module in root.named_modules():
            if re.fullmatch(pattern, name):
                self.handles.append(
                    module.register_forward_hook(self.hook(name), with_kwargs=True)
                )

    def hook(self, name):
        def record(module, args, kwargs, output):
            self.records[name].append(
                (
                    detached(args),
                    {
                        key: detached(value)
                        for key, value in kwargs.items()
                        if key != "past_key_values"
                    },
                    detached(first_tensor(output)),
                )
            )

        return record

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def bf16_copy(root: torch.nn.Module, module: torch.nn.Module, dtype: torch.dtype):
    """The module in dtype as a Transformers load in dtype holds it: rotary tables stay float32."""
    copied = copy.deepcopy(module).to(device=DEVICE, dtype=dtype).eval()
    for name, buffer in module.named_buffers():
        if name.endswith("inv_freq") or name.endswith("original_inv_freq"):
            owner, _, attribute = name.rpartition(".")
            target = copied.get_submodule(owner) if owner else copied
            setattr(
                target, attribute, buffer.detach().to(DEVICE, dtype=torch.float32)
            )
    return copied


def row_errors(candidate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    candidate = candidate.float()
    truth = truth.float()
    if truth.dim() <= 1:
        return (candidate - truth).norm().unsqueeze(0) / truth.norm().clamp_min(1e-20)
    candidate = candidate.flatten(0, -2)
    truth = truth.flatten(0, -2)
    return (candidate - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1e-20)


def whole_error(candidate: torch.Tensor, truth: torch.Tensor) -> float:
    return (
        (candidate.float() - truth.float()).norm() / truth.float().norm().clamp_min(1e-20)
    ).item()


def kind_of(name: str) -> str:
    return re.sub(r"\.\d+", ".*", name)


class Ledger:
    def __init__(self) -> None:
        self.local = collections.defaultdict(list)
        self.accumulated = collections.defaultdict(list)
        self.distributions = collections.defaultdict(list)

    def add_local(self, kind: str, whole: float, fp32_card: float, rows: torch.Tensor):
        self.local[kind].append(
            {
                "whole": whole,
                "fp32_card": fp32_card,
                "row_mean": rows.mean().item(),
                "row_p99": rows.quantile(0.99).item() if rows.numel() > 1 else rows.max().item(),
                "row_max": rows.max().item(),
            }
        )

    def add_accumulated(self, key: str, whole: float, rows: torch.Tensor):
        self.accumulated[key].append(
            {
                "whole": whole,
                "row_mean": rows.mean().item(),
                "row_median": rows.median().item(),
                "row_p99": rows.quantile(0.99).item() if rows.numel() > 1 else rows.max().item(),
                "row_max": rows.max().item(),
            }
        )

    def add_distribution(self, key: str, values: dict):
        self.distributions[key].append(values)


def replay_local(root, recorder, ledger: Ledger, chunk_note: str) -> None:
    """Every recorded module, in BF16 and in FP32 on the card, on the FP32 inputs."""
    for name, calls in recorder.records.items():
        module = root.get_submodule(name)
        floor = bf16_copy(root, module, torch.bfloat16)
        card = bf16_copy(root, module, torch.float32)
        with torch.inference_mode():
            for args, kwargs, truth in calls:
                if truth is None or not truth.is_floating_point():
                    continue
                output = first_tensor(
                    floor(*cast(args, DEVICE, torch.bfloat16), **cast(kwargs, DEVICE, torch.bfloat16))
                )
                exact = first_tensor(
                    card(*cast(args, DEVICE, torch.float32), **cast(kwargs, DEVICE, torch.float32))
                )
                truth_card = truth.to(DEVICE).float()
                ledger.add_local(
                    f"{chunk_note}/{kind_of(name)}",
                    whole_error(output, truth_card),
                    whole_error(exact, truth_card),
                    row_errors(output, truth_card).cpu(),
                )
        del floor, card
        torch.cuda.empty_cache()


def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 1.0 - 0.5 * (p - q).abs().sum(-1)


def served_first_code_probs(logits, first_codes, suppressed, sampling) -> torch.Tensor:
    logits = logits.float().clone()
    penalty = sampling["repetition_penalty"]
    for frame in range(1, logits.shape[0]):
        seen = first_codes[:frame].unique()
        row = logits[frame, seen]
        logits[frame, seen] = torch.where(row > 0, row / penalty, row * penalty)
    logits[:, suppressed] = float("-inf")
    probs = torch.softmax(logits / sampling["temperature"], dim=-1)
    kept = probs.topk(sampling["top_k"], dim=-1)
    filtered = torch.zeros_like(probs).scatter_(-1, kept.indices, kept.values)
    return filtered / filtered.sum(-1, keepdim=True)


def residual_probs(logits: torch.Tensor) -> torch.Tensor:
    """HF's residual codebook draw: top-k 50, top-p 0.8, temperature 1."""
    probs = torch.softmax(logits.float(), dim=-1)
    kept = probs.topk(50, dim=-1)
    kept_probs = kept.values / kept.values.sum(-1, keepdim=True)
    cumulative = kept_probs.cumsum(-1)
    remove = cumulative > 0.8
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    kept_probs = kept_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter_(-1, kept.indices, kept_probs)
    return filtered / filtered.sum(-1, keepdim=True)


def run_decoder_chain(root, layers_name, hidden, kwargs_by_layer, truths, ledger, key, groups, on_layer=None):
    """The BF16 chain through a decoder stack, with FP32's position tables and masks, compared to FP32 per layer."""
    layers = root.get_submodule(layers_name)
    for index, layer in enumerate(layers):
        floor = bf16_copy(root, layer, torch.bfloat16)
        with torch.inference_mode():
            hidden = first_tensor(
                floor(hidden, **cast(kwargs_by_layer[index], DEVICE, torch.bfloat16))
            )
        truth = truths[index].to(DEVICE)
        rows = row_errors(hidden, truth)
        ledger.add_accumulated(f"{key}/layer_{index:02d}/all", whole_error(hidden, truth), rows.cpu())
        for group, mask in groups.items():
            if mask.any():
                ledger.add_accumulated(
                    f"{key}/layer_{index:02d}/{group}",
                    whole_error(hidden[0, mask], truth[0, mask]),
                    rows[mask.flatten()].cpu(),
                )
        if on_layer is not None:
            on_layer(index, hidden)
        del floor
        torch.cuda.empty_cache()
    return hidden


def main() -> None:
    out = Path(sys.argv[1])
    sample_ids = Path(sys.argv[2]).read_text().split()[: int(sys.argv[3])]
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.float32,
        device_map="cpu",
        enable_audio_output=True,
        experts_implementation="eager",
    ).eval()
    config = model.config
    thinker = model.thinker
    talker = model.talker
    samples = {
        sample.sample_id: sample
        for sample in load_seedtts_samples(SEEDTTS_META, None, split="en")
    }
    ledger = Ledger()
    talker_truth_card = bf16_copy(model, talker, torch.float32)
    suppressed = torch.tensor(
        [
            i
            for i in range(
                config.talker_config.text_config.vocab_size - 1024,
                config.talker_config.text_config.vocab_size,
            )
            if i != config.talker_config.codec_eos_token_id
        ],
        device=DEVICE,
    )
    talker_sampling = {"repetition_penalty": 1.05, "temperature": 0.9, "top_k": 50}
    speaker_id = config.talker_config.speaker_id["ethan"]
    audio_id = config.thinker_config.audio_token_id
    thinker_targets = (
        r"model\.layers\.\d+|model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn|mlp|mlp\.gate"
        r"|self_attn\.[qkvo]_proj)|model\.norm|lm_head"
    )
    audio_targets = (
        r"conv2d1|conv2d2|layers\.\d+|layers\.\d+\.(self_attn|self_attn_layer_norm|final_layer_norm|fc1|fc2)"
        r"|ln_post|proj1|proj2"
    )
    talker_targets = (
        r"model\.layers\.\d+|model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn|mlp|mlp\.gate)"
        r"|model\.norm|codec_head|text_projection|hidden_projection"
    )
    predictor_targets = (
        r"model\.layers\.\d+|model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|self_attn|mlp)"
        r"|model\.norm|lm_head\.\d+"
    )
    code2wav_targets = (
        r"pre_transformer|pre_transformer\.layers\.\d+|upsample\.\d+|upsample\.\d+\.\d+|decoder\.\d+|decoder\.\d+\.\d+"
        r"|quantizer|code_embedding|fc_post_a|pre_conv"
    )

    for count, sample_id in enumerate(sample_ids):
        sample = samples[sample_id]
        prompt = (
            f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
            f"Now please read the following text out loud in the same voice and style: "
            f"{sample.target_text}"
        )
        text = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": sample.ref_audio},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        waveform, _ = librosa.load(sample.ref_audio, sr=16000)
        inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)
        prompt_ids = inputs["input_ids"]
        reply_ids = processor.tokenizer(sample.target_text, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([prompt_ids, reply_ids], dim=1)
        prompt_len = prompt_ids.shape[1]
        audio_mask = full_ids[0] == audio_id
        reply_mask = torch.arange(full_ids.shape[1]) >= prompt_len
        groups = {
            "audio": audio_mask.to(DEVICE),
            "prompt_text": (~audio_mask & ~reply_mask).to(DEVICE),
            "reply": reply_mask.to(DEVICE),
        }

        # Audio tower.
        tower = thinker.audio_tower
        lengths = inputs["feature_attention_mask"].sum(dim=1)
        packed = inputs["input_features"].permute(0, 2, 1)[inputs["feature_attention_mask"].bool()].permute(1, 0)
        recorder = Recorder(tower, audio_targets)
        with torch.inference_mode():
            tower_truth = tower(packed.float(), feature_lens=lengths).last_hidden_state
        recorder.close()
        replay_local(tower, recorder, ledger, "audio_tower")
        tower_floor = bf16_copy(thinker, tower, torch.bfloat16)
        with torch.inference_mode():
            tower_bf16 = tower_floor(
                packed.to(DEVICE, torch.bfloat16), feature_lens=lengths.to(DEVICE)
            ).last_hidden_state
        ledger.add_accumulated(
            "audio_tower/output",
            whole_error(tower_bf16, tower_truth.to(DEVICE)),
            row_errors(tower_bf16, tower_truth.to(DEVICE)).cpu(),
        )
        del tower_floor

        # Thinker, the prompt and the target text teacher-forced.
        recorder = Recorder(thinker, thinker_targets)
        thinker.rope_deltas = None
        with torch.inference_mode():
            thinker_out = thinker(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                input_features=inputs["input_features"].float(),
                feature_attention_mask=inputs["feature_attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        recorder.close()
        truth_hidden_states = [state.detach().cpu() for state in thinker_out.hidden_states]
        truth_logits = thinker_out.logits[0].detach().float()
        layer_kwargs = []
        layer_truths = []
        for index in range(len(thinker.model.layers)):
            args, kwargs, output = recorder.records[f"model.layers.{index}"][0]
            layer_kwargs.append(kwargs)
            layer_truths.append(output)
        replay_local(thinker, recorder, ledger, "thinker")
        recorder.records.clear()

        embedding = bf16_copy(thinker, thinker.get_input_embeddings(), torch.bfloat16)
        with torch.inference_mode():
            hidden = embedding(full_ids.to(DEVICE))
            hidden[0, groups["audio"]] = tower_bf16.to(hidden)
        ledger.add_accumulated(
            "thinker/embeds",
            whole_error(hidden, truth_hidden_states[0].to(DEVICE)),
            row_errors(hidden, truth_hidden_states[0].to(DEVICE)).cpu(),
        )
        accept = config.talker_config.accept_hidden_layer
        kept = {}

        def keep(index, state, kept=kept, accept=accept):
            if index == accept - 1:
                kept["hidden"] = state.detach().clone()

        kept["embeds"] = hidden.detach().clone()
        hidden = run_decoder_chain(
            thinker, "model.layers", hidden, layer_kwargs, layer_truths, ledger, "thinker", groups, keep
        )
        norm = bf16_copy(thinker, thinker.model.norm, torch.bfloat16)
        head = bf16_copy(thinker, thinker.lm_head, torch.bfloat16)
        with torch.inference_mode():
            logits = head(norm(hidden))[0].float()
        steps = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
        truth_step_logits = truth_logits[steps].to(DEVICE)
        step_logits = logits[steps.to(DEVICE)]
        p_truth = torch.softmax(truth_step_logits / THINKER_TEMPERATURE, dim=-1)
        p_floor = torch.softmax(step_logits / THINKER_TEMPERATURE, dim=-1)
        ledger.add_distribution(
            "thinker/logits",
            {
                "overlap_mean": overlap(p_truth, p_floor).mean().item(),
                "overlap_min": overlap(p_truth, p_floor).min().item(),
                "top1_agreement": (truth_step_logits.argmax(-1) == step_logits.argmax(-1)).float().mean().item(),
                "steps": int(steps.numel()),
            },
        )
        del norm, head, embedding

        # The talker's prefill from each chain, with HF's own assembly.
        def talker_rows(owner, embed, hidden_rows, dtype):
            hidden_mm = hidden_rows
            thinker_embed = embed
            multimodal = (full_ids == audio_id).to(DEVICE)
            user_end = (
                torch.nonzero(
                    (prompt_ids[0, :-1] == config.im_start_token_id)
                    & (prompt_ids[0, 1:] == config.assistant_token_id)
                )
                .flatten()[-1]
                .item()
            )
            text_projection = owner.text_projection
            hidden_projection = owner.hidden_projection
            with torch.inference_mode():
                user_mask = multimodal[0, :user_end]
                user = torch.empty(user_end, thinker_embed.shape[-1] if False else owner.config.text_config.hidden_size, device=DEVICE, dtype=dtype)
                if user_mask.any():
                    user[user_mask] = hidden_projection(hidden_mm[0, :user_end][user_mask])
                user[~user_mask] = text_projection(thinker_embed[0, :user_end][~user_mask])
                specials = torch.tensor(
                    [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id], device=DEVICE
                )
                special_rows = text_projection(special_embedding(specials).to(DEVICE, dtype))
                tts_bos, tts_eos, tts_pad = special_rows[0], special_rows[1], special_rows[2]
                assistant = text_projection(thinker_embed[0, user_end:])
                codec_ids = torch.tensor(
                    [
                        config.talker_config.codec_nothink_id,
                        config.talker_config.codec_think_bos_id,
                        config.talker_config.codec_think_eos_id,
                        speaker_id,
                        config.talker_config.codec_pad_id,
                        config.talker_config.codec_bos_id,
                    ],
                    device=DEVICE,
                )
                codec = owner.get_input_embeddings()(codec_ids)
                tail_text = torch.cat(
                    [assistant[:3], tts_pad.expand(4, -1), tts_bos[None], assistant[3:4]], dim=0
                )
                tail_codec = torch.cat(
                    [torch.zeros(3, codec.shape[-1], device=DEVICE, dtype=dtype), codec], dim=0
                )
                prefill = torch.cat([user, tail_text + tail_codec], dim=0)
                trailing = torch.cat([assistant[4:], tts_eos[None]], dim=0)
            return prefill, trailing, tts_pad

        special_embedding = bf16_copy(thinker, thinker.get_input_embeddings(), torch.float32)
        talker_floor = bf16_copy(model, talker, torch.bfloat16)
        truth_embed_rows = truth_hidden_states[0].to(DEVICE)
        truth_hidden_rows = truth_hidden_states[accept].to(DEVICE)
        special_embedding_float = special_embedding
        prefill_truth, trailing_truth, pad_truth = talker_rows(
            talker_truth_card, truth_embed_rows, truth_hidden_rows, torch.float32
        )
        special_embedding = bf16_copy(thinker, thinker.get_input_embeddings(), torch.bfloat16)
        prefill_floor, trailing_floor, pad_floor = talker_rows(
            talker_floor, kept["embeds"], kept["hidden"], torch.bfloat16
        )
        special_embedding = special_embedding_float
        ledger.add_accumulated(
            "talker/prefill_rows",
            whole_error(prefill_floor, prefill_truth),
            row_errors(prefill_floor, prefill_truth).cpu(),
        )

        # A seeded FP32 draw of the codes on the card, then the talker over the whole sequence, FP32 and BF16.
        torch.manual_seed(SEED + count)
        with torch.inference_mode():
            talker_result = talker_truth_card.generate(
                inputs_embeds=prefill_truth[None],
                trailing_text_hidden=trailing_truth[None],
                tts_pad_embed=pad_truth[None, None],
                talker_input_ids=torch.full(
                    (1, prefill_truth.shape[0]), config.tts_pad_token_id, device=DEVICE
                ),
                attention_mask=torch.ones(1, prefill_truth.shape[0], device=DEVICE, dtype=torch.long),
                max_new_tokens=int(min(400, 60 + 12.5 * len(sample.target_text.split()))),
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                eos_token_id=config.talker_config.codec_eos_token_id,
                repetition_penalty=1.05,
                suppress_tokens=suppressed.tolist(),
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        codes = (
            torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
            .transpose(1, 2)[0]
            .T
        )
        spoken = codes[codes[:, 0] != config.talker_config.codec_eos_token_id]
        frames = codes.shape[0]

        def feedback(owner, frame_codes):
            rows = owner.model.get_input_embeddings()(frame_codes[:, 0])
            for group, table in enumerate(owner.code_predictor.get_input_embeddings()):
                rows = rows + table(frame_codes[:, group + 1])
            return rows

        def decode_rows(owner, trailing, pad):
            count_rows = frames - 1
            padding = pad.expand(max(count_rows - trailing.shape[0], 0), -1)
            text = torch.cat([trailing, padding])[:count_rows]
            return feedback(owner, codes[:count_rows]) + text

        with torch.inference_mode():
            decode_truth = decode_rows(talker_truth_card, trailing_truth, pad_truth)
            decode_floor = decode_rows(talker_floor, trailing_floor, pad_floor)
        ledger.add_accumulated(
            "talker/decode_rows",
            whole_error(decode_floor, decode_truth),
            row_errors(decode_floor, decode_truth).cpu(),
        )
        sequence_truth = torch.cat([prefill_truth, decode_truth])[None]
        sequence_floor = torch.cat([prefill_floor, decode_floor])[None]
        talker_cpu_sequence = sequence_truth.cpu()
        recorder = Recorder(talker, talker_targets)
        with torch.inference_mode():
            talker_out = talker.model(inputs_embeds=talker_cpu_sequence, use_cache=False)
            talker.codec_head(talker_out.last_hidden_state)
        recorder.close()
        talker_layer_kwargs = []
        talker_layer_truths = []
        for index in range(len(talker.model.layers)):
            args, kwargs, output = recorder.records[f"model.layers.{index}"][0]
            talker_layer_kwargs.append(kwargs)
            talker_layer_truths.append(output)
        replay_local(talker, recorder, ledger, "talker")
        recorder.records.clear()
        prefill_rows = prefill_truth.shape[0]
        talker_groups = {
            "prefill": (torch.arange(sequence_truth.shape[1]) < prefill_rows).to(DEVICE),
            "decode": (torch.arange(sequence_truth.shape[1]) >= prefill_rows).to(DEVICE),
        }
        hidden_floor = run_decoder_chain(
            talker,
            "model.layers",
            sequence_floor,
            talker_layer_kwargs,
            talker_layer_truths,
            ledger,
            "talker",
            talker_groups,
        )
        with torch.inference_mode():
            final_floor = talker_floor.model.norm(hidden_floor)[0]
            final_truth = talker_out.last_hidden_state[0].to(DEVICE)
            steps_talker = torch.arange(prefill_rows - 1, prefill_rows - 1 + frames, device=DEVICE)
            logits_truth = talker_truth_card.codec_head(final_truth[steps_talker])
            logits_floor = talker_floor.codec_head(final_floor[steps_talker]).float()
        p_truth = served_first_code_probs(logits_truth, codes[:, 0], suppressed, talker_sampling)
        p_floor = served_first_code_probs(logits_floor, codes[:, 0], suppressed, talker_sampling)
        ledger.add_distribution(
            "talker/first_code",
            {
                "overlap_mean": overlap(p_truth, p_floor).mean().item(),
                "overlap_min": overlap(p_truth, p_floor).min().item(),
                "top1_agreement": (logits_truth.argmax(-1) == logits_floor.argmax(-1)).float().mean().item(),
                "steps": int(frames),
            },
        )

        # The code predictor, teacher-forced frame by frame, on FP32 talker hiddens: local; on BF16's: accumulated.
        predictor = talker.code_predictor

        def predictor_inputs(owner, hiddens):
            tables = owner.code_predictor.get_input_embeddings()
            return torch.stack(
                [hiddens, owner.model.get_input_embeddings()(codes[:, 0])]
                + [tables[group](codes[:, group + 1]) for group in range(len(tables) - 1)],
                dim=1,
            )

        def predictor_logits(owner, inputs_rows):
            states = owner.code_predictor.model(inputs_embeds=inputs_rows).last_hidden_state
            return torch.stack(
                [head_module(states[:, group + 1]) for group, head_module in enumerate(owner.code_predictor.lm_head)],
                dim=1,
            ).float()

        with torch.inference_mode():
            pred_inputs_truth = predictor_inputs(talker_truth_card, final_truth[steps_talker])
            pred_inputs_floor = predictor_inputs(talker_floor, final_floor[steps_talker])
            pred_truth = predictor_logits(talker_truth_card, pred_inputs_truth)
            pred_floor = predictor_logits(talker_floor, pred_inputs_floor)
        recorder = Recorder(predictor, predictor_targets)
        with torch.inference_mode():
            states = predictor.model(inputs_embeds=pred_inputs_truth.cpu()).last_hidden_state
            for group, head_module in enumerate(predictor.lm_head):
                head_module(states[:, group + 1])
        recorder.close()
        replay_local(predictor, recorder, ledger, "code_predictor")
        recorder.records.clear()
        p_truth = residual_probs(pred_truth)
        p_floor = residual_probs(pred_floor)
        ledger.add_distribution(
            "code_predictor/residuals",
            {
                "overlap_mean": overlap(p_truth, p_floor).mean().item(),
                "overlap_min": overlap(p_truth, p_floor).min().item(),
                "top1_agreement": (pred_truth.argmax(-1) == pred_floor.argmax(-1)).float().mean().item(),
                "logits_match": 1.0 - whole_error(pred_floor, pred_truth),
                "steps": int(pred_truth.shape[0] * pred_truth.shape[1]),
            },
        )

        # Code2wav on the spoken codes.
        code2wav = model.code2wav
        wav_codes = spoken.T[None].cpu()
        recorder = Recorder(code2wav, code2wav_targets)
        with torch.inference_mode():
            wav_truth = code2wav(wav_codes)
        recorder.close()
        replay_local(code2wav, recorder, ledger, "code2wav")
        recorder.records.clear()
        wav_floor_module = bf16_copy(model, code2wav, torch.bfloat16)
        with torch.inference_mode():
            wav_floor = wav_floor_module(wav_codes.to(DEVICE))
        ledger.add_accumulated(
            "code2wav/output",
            whole_error(wav_floor, wav_truth.to(DEVICE)),
            torch.tensor([whole_error(wav_floor, wav_truth.to(DEVICE))]),
        )
        del talker_floor, wav_floor_module
        torch.cuda.empty_cache()
        print(f"done {sample_id}: frames {frames} rows {full_ids.shape[1]}", flush=True)
        write_report(out, ledger)


def summarize(entries: list[dict]) -> dict:
    keys = entries[0].keys()
    return {
        key: sum(entry[key] for entry in entries) / len(entries)
        for key in keys
        if isinstance(entries[0][key], (int, float))
    } | {"count": len(entries)}


def write_report(out: Path, ledger: Ledger) -> None:
    out.write_text(
        json.dumps(
            {
                "local": {key: summarize(entries) for key, entries in ledger.local.items()},
                "accumulated": {key: summarize(entries) for key, entries in ledger.accumulated.items()},
                "distributions": {key: summarize(entries) for key, entries in ledger.distributions.items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
