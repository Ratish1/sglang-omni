"""Does the MoE gate's BF16 rounding make the expert flips the numerics ledger found? HF's modules, three gate forms.

The ledger: every BF16 sub-operation of a thinker or talker layer matches FP32 at 99.5 % or better except the MoE block
(92.6 %, worst row 44 %), although its gate logits match at 99.8 %. HF's gate rounds router logits to BF16 before its
softmax and top-k, so experts within the rounding step of the k-th flip, and a flipped expert moves the row by its whole
share. Forms, each with BF16 experts: bf16 is HF's gate; gate_fp32 keeps the logits, softmax and top-k in FP32 from the
BF16 hidden state and BF16 weight (the kernel's FP32 accumulator kept instead of rounded); gate_fp32_input also starts
from the FP32 hidden state, what is left when the input's own rounding is gone (local only, not a servable form).

Local: FP32 inputs at each layer, flip rate against FP32's selection and the block's match. Accumulated: the BF16 chain
through the thinker, its talker rows, then the talker over FP32's codes teacher-forced, match per layer and row group,
flip rate, the output distributions.

usage: python3 moe_gate_precision.py OUT.json SAMPLE_IDS_FILE COUNT
"""

import collections
import json
import sys
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

sys.path.insert(0, str(Path(__file__).parent))
from hf_numerics_ledger import (  # noqa: E402
    DEVICE,
    SEEDTTS_META,
    Recorder,
    bf16_copy,
    cast,
    first_tensor,
    load_seedtts_samples,
    overlap,
    row_errors,
    whole_error,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
FORMS = ("bf16", "gate_fp32")
TEMPERATURE = 0.7


def install_fp32_gate(block: torch.nn.Module) -> None:
    gate = block.gate
    weight = gate.weight.detach().float()

    def fp32_gate(hidden_states):
        rows = hidden_states.reshape(-1, gate.hidden_dim)
        logits = F.linear(rows.float(), weight)
        probs = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, gate.top_k, dim=-1)
        if gate.norm_topk_prob:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        else:
            pass
        return logits, top_values.to(hidden_states.dtype), top_indices

    gate.forward = fp32_gate


def layer_copy(root, layer, form: str):
    floor = bf16_copy(root, layer, torch.bfloat16)
    if form == "gate_fp32" and hasattr(floor.mlp, "gate"):
        install_fp32_gate(floor.mlp)
    else:
        pass
    return floor


def truth_selection(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    return torch.topk(torch.softmax(logits.float(), dim=-1), top_k, dim=-1).indices.sort(-1).values


def flip_fraction(selected: torch.Tensor, truth: torch.Tensor) -> float:
    return (selected.sort(-1).values != truth).any(-1).float().mean().item()


def add(table, key, value):
    table[key].append(value)


def local_blocks(root, layer_count, records, table, name):
    for index in range(layer_count):
        block_name = f"model.layers.{index}.mlp"
        args, kwargs, truth = records[block_name][0]
        _, _, truth_logits = records[f"{block_name}.gate"][0]
        block = root.get_submodule(block_name)
        if not hasattr(block, "gate"):
            continue
        truth_top = truth_selection(truth_logits.to(DEVICE), block.gate.top_k)
        for form in (*FORMS, "gate_fp32_input"):
            floor = bf16_copy(root, block, torch.bfloat16)
            if form != "bf16":
                install_fp32_gate(floor)
            else:
                pass
            if form == "gate_fp32_input":
                exact_rows = args[0].reshape(-1, block.gate.hidden_dim).to(DEVICE).float()
                exact_gate = floor.gate.forward

                def from_exact(hidden_states, exact_rows=exact_rows, exact_gate=exact_gate):
                    logits, values, indices = exact_gate(exact_rows)
                    return logits, values.to(hidden_states.dtype), indices

                floor.gate.forward = from_exact
            else:
                pass
            seen = {}
            gate_forward = floor.gate.forward

            def keep(hidden_states, gate_forward=gate_forward, seen=seen):
                output = gate_forward(hidden_states)
                seen["indices"] = output[2]
                return output

            floor.gate.forward = keep
            with torch.inference_mode():
                output = first_tensor(floor(*cast(args, DEVICE, torch.bfloat16)))
            truth_card = truth.to(DEVICE).float()
            rows = row_errors(output, truth_card)
            add(table, f"{name}/{form}/flip_fraction", flip_fraction(seen["indices"], truth_top))
            add(table, f"{name}/{form}/block_match", 1.0 - whole_error(output, truth_card))
            add(table, f"{name}/{form}/block_row_mean", 1.0 - rows.mean().item())
            add(table, f"{name}/{form}/block_row_p99", 1.0 - rows.quantile(0.99).item())
            del floor
        torch.cuda.empty_cache()


def main() -> None:
    out = Path(sys.argv[1])
    sample_ids = Path(sys.argv[2]).read_text().split()[: int(sys.argv[3])]
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, device_map="cpu", enable_audio_output=True, experts_implementation="eager"
    ).eval()
    config = model.config
    thinker = model.thinker
    talker = model.talker
    samples = {s.sample_id: s for s in load_seedtts_samples(SEEDTTS_META, None, split="en")}
    audio_id = config.thinker_config.audio_token_id
    accept = config.talker_config.accept_hidden_layer
    table = collections.defaultdict(list)
    targets = r"model\.layers\.\d+|model\.layers\.\d+\.mlp|model\.layers\.\d+\.mlp\.gate"
    for sample_id in sample_ids:
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
        prompt_rows = prompt_ids.shape[1]
        audio_mask = (full_ids[0] == audio_id).to(DEVICE)
        recorder = Recorder(thinker, targets)
        thinker.rope_deltas = None
        with torch.inference_mode():
            output = thinker(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                input_features=inputs["input_features"].float(),
                feature_attention_mask=inputs["feature_attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        recorder.close()
        layer_count = len(thinker.model.layers)
        local_blocks(thinker, layer_count, recorder.records, table, "thinker_local")
        truth_logits = output.logits[0].detach().float()[prompt_rows - 1 : -1].to(DEVICE)
        tower = bf16_copy(thinker, thinker.audio_tower, torch.bfloat16)
        lengths = inputs["feature_attention_mask"].sum(dim=1)
        packed = inputs["input_features"].permute(0, 2, 1)[inputs["feature_attention_mask"].bool()].permute(1, 0)
        with torch.inference_mode():
            tower_rows = tower(packed.to(DEVICE, torch.bfloat16), feature_lens=lengths.to(DEVICE)).last_hidden_state
        del tower
        for form in FORMS:
            embedding = bf16_copy(thinker, thinker.get_input_embeddings(), torch.bfloat16)
            with torch.inference_mode():
                hidden = embedding(full_ids.to(DEVICE))
                hidden[0, audio_mask] = tower_rows.to(hidden)
            for index in range(layer_count):
                args, kwargs, truth = recorder.records[f"model.layers.{index}"][0]
                _, _, truth_gate = recorder.records[f"model.layers.{index}.mlp.gate"][0]
                floor = layer_copy(thinker, thinker.model.layers[index], form)
                seen = {}
                gate_forward = floor.mlp.gate.forward

                def keep(hidden_states, gate_forward=gate_forward, seen=seen):
                    result = gate_forward(hidden_states)
                    seen["indices"] = result[2]
                    return result

                floor.mlp.gate.forward = keep
                with torch.inference_mode():
                    hidden = first_tensor(floor(hidden, **cast(kwargs, DEVICE, torch.bfloat16)))
                truth_card = truth.to(DEVICE).float()
                rows = row_errors(hidden, truth_card)
                truth_top = truth_selection(truth_gate.to(DEVICE), floor.mlp.gate.top_k)
                add(table, f"thinker_chain/{form}/flip_fraction", flip_fraction(seen["indices"], truth_top))
                if index in (5, 11, accept - 1, 35, layer_count - 1):
                    add(table, f"thinker_chain/{form}/layer_{index:02d}/audio_row_mean", 1.0 - rows[audio_mask].mean().item())
                    add(table, f"thinker_chain/{form}/layer_{index:02d}/audio_row_median", 1.0 - rows[audio_mask].median().item())
                    add(table, f"thinker_chain/{form}/layer_{index:02d}/audio_whole", 1.0 - whole_error(hidden[0, audio_mask], truth_card[0, audio_mask]))
                    add(table, f"thinker_chain/{form}/layer_{index:02d}/text_row_mean", 1.0 - rows[~audio_mask].mean().item())
                else:
                    pass
                if index == accept - 1:
                    prefill_audio = hidden[0, audio_mask].detach().clone()
                else:
                    pass
                del floor
            torch.cuda.empty_cache()
            norm = bf16_copy(thinker, thinker.model.norm, torch.bfloat16)
            head = bf16_copy(thinker, thinker.lm_head, torch.bfloat16)
            with torch.inference_mode():
                logits = head(norm(hidden))[0].float()[prompt_rows - 1 : -1]
            p_truth = torch.softmax(truth_logits / TEMPERATURE, dim=-1)
            p_form = torch.softmax(logits / TEMPERATURE, dim=-1)
            add(table, f"thinker_chain/{form}/logits_overlap", overlap(p_truth, p_form).mean().item())
            add(table, f"thinker_chain/{form}/logits_top1", (truth_logits.argmax(-1) == logits.argmax(-1)).float().mean().item())
            projection = bf16_copy(talker, talker.hidden_projection, torch.bfloat16)
            truth_projection = bf16_copy(talker, talker.hidden_projection, torch.float32)
            with torch.inference_mode():
                rows_form = projection(prefill_audio)
                rows_truth = truth_projection(output.hidden_states[accept][0].to(DEVICE)[audio_mask])
            errors = row_errors(rows_form, rows_truth)
            add(table, f"talker_rows/{form}/audio_row_mean", 1.0 - errors.mean().item())
            add(table, f"talker_rows/{form}/audio_row_median", 1.0 - errors.median().item())
            add(table, f"talker_rows/{form}/audio_row_p99", 1.0 - errors.quantile(0.99).item())
            add(table, f"talker_rows/{form}/audio_whole", 1.0 - whole_error(rows_form, rows_truth))
            del norm, head, projection, truth_projection, embedding
        recorder.records.clear()
        print(f"done {sample_id}", flush=True)
        out.write_text(
            json.dumps({key: sum(values) / len(values) for key, values in sorted(table.items())}, indent=1)
        )


if __name__ == "__main__":
    main()
