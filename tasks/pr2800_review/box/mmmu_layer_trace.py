"""The MMMU talker stage's 20 requests traced layer by layer, and which component's rounding decides their answers.

Run on the box after the arms' trees hold the accuracy harness (mmmu_accuracy_flow.py copies it in):
    python3 /workspace/box/tasks/pr2800_review/box/mmmu_layer_trace.py run --arms upy,vpg --out /data/<run>
    python3 mmmu_layer_trace.py read --arms upy,vpg --out /data/<run>
Every request goes in as HF's processor outputs, which equal omni's served inputs bit for bit (harness
test_image_inputs_match_processor). Phases, each a subprocess on its own card:
  features   per arm tree: HF's vision encoder in FP32 (truth) and BF16, and the arm's served encoder, one request at a
             time; the relative L2 to FP32 after each of the 27 blocks, the merger and the three deepstack mergers;
             the encoder outputs saved per request.
  answers    the HF thinker in FP32 and in BF16, each fed the vision features of every source (fp32, hf_bf16, each arm)
             through HF's own get_image_features (patched to return them), greedy as the stage: answer letter,
             correctness and the logits of the choice letters where the answer letter is emitted.
  internals  the FP32 and BF16 HF thinkers teacher-forced on the FP32 thinker's own tokens with FP32 features: per layer
             the hidden-state error and the share of positions whose top-8 MoE experts differ, over the answer and at
             the first step where the BF16 argmax leaves the FP32 tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CARD_NODES = {0: 0, 1: 0, 2: 1, 3: 1}
ANSWER_TREE = "vpg"


def checkpoint_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL)


def full_precision() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def requests():
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    from tests.test_model.qwen3_omni_component_accuracy.mmmu_requests import (
        load_image_questions,
        reference_inputs,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    questions = load_image_questions()
    return processor, questions, [reference_inputs(processor, sample) for sample in questions]


def hf_visual(checkpoint: str, dtype):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

    from sglang_omni.models.qwen3_omni.components.common import load_thinker_config

    # note (ratish): constructed in its dtype as from_pretrained does, so the rotary table stays float32 in BF16 too.
    visual = hf_modeling.Qwen3OmniMoeVisionEncoder._from_config(  # noqa: leading-underscore  # Transformers construction API
        load_thinker_config(checkpoint).vision_config, dtype=dtype
    )
    weight_map = json.loads(Path(checkpoint, "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({file for name, file in weight_map.items() if name.startswith("thinker.visual.")}):
        with safe_open(f"{checkpoint}/{shard}", framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("thinker.visual."):
                    state[name.removeprefix("thinker.visual.")] = handle.get_tensor(name)
                else:
                    pass
    visual.load_state_dict(state, strict=True)
    print(f"hf visual {dtype}: rotary table {visual.rotary_pos_emb.inv_freq.dtype}", flush=True)
    return visual.to("cuda:0").eval()


def patch_vision(thinker, stored: list, grid) -> None:
    """Make the thinker's own get_image_features return stored encoder outputs, in the thinker's dtype."""
    import torch
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

    merged, *deepstack = (tensor.to(thinker.device, thinker.dtype) for tensor in stored)
    split_sizes = (grid.prod(-1) // thinker.visual.spatial_merge_size**2).tolist()
    output = hf_modeling.BaseModelOutputWithDeepstackFeatures(
        pooler_output=torch.split(merged, split_sizes),
        deepstack_features=[torch.split(feature, split_sizes) for feature in deepstack],
    )

    # note (ratish): generate pre-encodes images through this method, choosing its arguments by parameter name.
    def get_image_features(pixel_values, image_grid_thw=None, **kwargs):
        return output

    thinker.get_image_features = get_image_features


def stored_features(out: Path, source: str) -> dict:
    import torch

    return torch.load(sorted(out.glob(f"features_{source}_from_*.pt"))[0])


def block_trace(visual, run) -> tuple[list, list]:
    """Every block's output, then the encoder's merged and deepstack outputs, as float32 on the host."""
    import torch

    states = []
    handles = [
        block.register_forward_hook(
            lambda module, args, output: states.append(
                (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
            )
        )
        for block in visual.blocks
    ]
    with torch.inference_mode():
        outputs = run()
    for handle in handles:
        handle.remove()
    return states, [output.detach().float().cpu() for output in outputs]


def relative_l2(candidate, truth) -> float:
    return ((candidate - truth).norm() / truth.norm()).item()


def features(arm: str, out: Path) -> None:
    import torch

    from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder

    full_precision()
    checkpoint = checkpoint_dir()
    _, questions, request_inputs = requests()
    truth = hf_visual(checkpoint, torch.float32)
    floor = hf_visual(checkpoint, torch.bfloat16)
    served = Qwen3OmniImageEncoder(checkpoint, device="cuda:0")

    def hf_run(visual, pixels, grid):
        def run():
            output = visual(pixels.to(visual.dtype), grid_thw=grid)
            return [output.pooler_output, *output.deepstack_features]

        return run

    def served_run(pixels, grid):
        def run():
            output = served(pixel_values=pixels, image_grid_thw=grid)
            return [output["image_embeds"], *output["deepstack_visual_embeds_image"]]

        return run

    saved = {"fp32": {}, "hf_bf16": {}, arm: {}}
    layers = {}
    for sample, inputs in zip(questions, request_inputs):
        pixels = inputs["pixel_values"].to("cuda:0")
        grid = inputs["image_grid_thw"].to("cuda:0")
        truth_blocks, truth_outputs = block_trace(truth, hf_run(truth, pixels, grid))
        traces = {
            "hf_bf16": block_trace(floor, hf_run(floor, pixels, grid)),
            arm: block_trace(served.visual, served_run(pixels, grid)),
        }
        saved["fp32"][sample.sample_id] = truth_outputs
        layers[sample.sample_id] = {}
        for name, (blocks, outputs) in traces.items():
            saved[name][sample.sample_id] = outputs
            layers[sample.sample_id][name] = {
                "blocks": [relative_l2(block, reference) for block, reference in zip(blocks, truth_blocks)],
                "outputs": [relative_l2(output, reference) for output, reference in zip(outputs, truth_outputs)],
            }
    for name, by_sample in saved.items():
        torch.save(by_sample, out / f"features_{name}_from_{arm}.pt")
    (out / f"vision_layers_{arm}.json").write_text(json.dumps(layers))


def vision_ablation(out: Path) -> None:
    """The served encoder with each omni op put back to HF's, one at a time and all together, against FP32."""
    import torch
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

    from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder

    full_precision()
    checkpoint = checkpoint_dir()
    _, questions, request_inputs = requests()
    truth = hf_visual(checkpoint, torch.float32)
    floor = hf_visual(checkpoint, torch.bfloat16)

    def hf_attention(visual) -> None:
        visual.packed_attention = None
        for block in visual.blocks:
            block.attn.packed_attention = None

    def hf_rotary(visual) -> None:
        visual.joint_rope_kernel = None
        for block in visual.blocks:
            block.attn.joint_rope_kernel = None

    def hf_interpolation(visual) -> None:
        def interpolate(grid_thw):
            indices, weights = hf_modeling.get_vision_interpolation_indices_and_weights(
                grid_thw.to(visual.pos_embed.weight.device),
                num_grid_per_side=visual.num_grid_per_side,
                mode=visual.interpolation_mode,
                align_corners=visual.interpolation_align_corners,
                spatial_merge_size=visual.config.spatial_merge_size,
                kwargs={},
            )
            return (visual.pos_embed(indices) * weights[:, :, None]).sum(1)

        visual.legacy_pos_embed_interpolate = interpolate

    def hf_patch_embed(visual) -> None:
        visual.patch_embed = floor.patch_embed

    swaps = {
        "attention": [hf_attention],
        "rotary": [hf_rotary],
        "interpolation": [hf_interpolation],
        "patch embed": [hf_patch_embed],
        "all four": [hf_attention, hf_rotary, hf_interpolation, hf_patch_embed],
    }
    variants = {"served": [], **{f"hf {name}": functions for name, functions in swaps.items()}}
    encoders = {}
    for name, functions in variants.items():
        encoder = Qwen3OmniImageEncoder(checkpoint, device="cuda:0")
        for function in functions:
            function(encoder.visual)
        encoders[name] = encoder

    def outputs_of(visual, pixels, grid):
        with torch.inference_mode():
            output = visual(pixels.to(visual.dtype), grid_thw=grid)
        return [output.pooler_output.float(), *(feature.float() for feature in output.deepstack_features)]

    errors = {name: [] for name in ["hf_bf16", *variants]}
    closure = []
    for sample, inputs in zip(questions, request_inputs):
        pixels = inputs["pixel_values"].to("cuda:0")
        grid = inputs["image_grid_thw"].to("cuda:0")
        reference = outputs_of(truth, pixels, grid)
        floor_outputs = outputs_of(floor, pixels, grid)
        errors["hf_bf16"].append([relative_l2(o, r) for o, r in zip(floor_outputs, reference)])
        for name, encoder in encoders.items():
            with torch.inference_mode():
                encoded = encoder(pixel_values=pixels, image_grid_thw=grid)
            outputs = [encoded["image_embeds"].float(), *(f.float() for f in encoded["deepstack_visual_embeds_image"])]
            errors[name].append([relative_l2(o, r) for o, r in zip(outputs, reference)])
            if name == "hf all four":
                closure.append(max((o - f).abs().max().item() for o, f in zip(outputs, floor_outputs)))
            else:
                pass
    (out / "vision_ablation.json").write_text(json.dumps({"errors": errors, "closure_max_abs": closure}))
    for name, rows in errors.items():
        means = [sum(row[index] for row in rows) / len(rows) for index in range(4)]
        print(f"{name:20s} merged {means[0]:.3e} deepstack " + " ".join(f"{value:.3e}" for value in means[1:]), flush=True)
    print(f"all four put back against HF BF16, largest abs difference per request: {closure}", flush=True)


def choice_logits(tokenizer, tokens: list[int], step_logits, choices: list[str]) -> dict[str, object]:
    """The choice letters' logits at the step that emits the final answer letter, and their top-two margin."""
    for step in range(len(tokens) - 1, -1, -1):
        piece = tokenizer.decode(tokens[step : step + 1]).strip()
        prefix = tokenizer.decode(tokens[:step]).rstrip().rstrip("*").rstrip()
        if piece in choices and prefix.lower().endswith("answer:"):
            letter_logits = {}
            for choice in choices:
                ids = {
                    tokenizer.convert_tokens_to_ids(variant)
                    for variant in (choice, f"Ġ{choice}")
                    if tokenizer.convert_tokens_to_ids(variant) is not None
                }
                letter_logits[choice] = max(step_logits[step][index].item() for index in ids)
            ranked = sorted(letter_logits.values(), reverse=True)
            return {"step": step, "letter_logits": letter_logits, "margin": ranked[0] - ranked[1]}
        else:
            pass
    return {"step": None, "letter_logits": {}, "margin": None}


def answers(thinker_dtype: str, sources: list[str], out: Path) -> None:
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.tasks.visual_understand import build_mmmu_result_records
    from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MAX_TOKENS

    full_precision()
    processor, questions, request_inputs = requests()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[thinker_dtype]
    thinker = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=dtype, device_map={"": "cuda:0"}, enable_audio_output=False, experts_implementation="eager"
    ).thinker.eval()
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for source in sources:
        stored = stored_features(out, source)
        records = []
        texts = []
        for sample, inputs in zip(questions, request_inputs):
            patch_vision(thinker, stored[sample.sample_id], inputs["image_grid_thw"])
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = thinker.generate(
                    input_ids=inputs["input_ids"].to("cuda:0"),
                    attention_mask=inputs["attention_mask"].to("cuda:0"),
                    pixel_values=inputs["pixel_values"].to("cuda:0", dtype),
                    image_grid_thw=inputs["image_grid_thw"].to("cuda:0"),
                    max_new_tokens=MAX_TOKENS,
                    eos_token_id=eos_token_id,
                    do_sample=False,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
            tokens = generated.sequences[0, prompt_length:].tolist()
            text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "tokens": tokens,
                    **choice_logits(
                        processor.tokenizer,
                        tokens,
                        [step[0].float().cpu() for step in generated.logits],
                        sample.all_choices,
                    ),
                }
            )
        scored = build_mmmu_result_records(
            questions,
            [RequestResult(request_id=sample.sample_id, text=text, is_success=True) for sample, text in zip(questions, texts)],
        )
        for record, result, text in zip(records, scored, texts):
            record.update(
                predicted=result["predicted"], is_correct=result["is_correct"], gold=result["expected"], text=text
            )
        (out / f"answers_{thinker_dtype}_{source}.json").write_text(json.dumps(records))
        print(f"{thinker_dtype} thinker, {source} features: {sum(r['is_correct'] for r in records)} correct", flush=True)


def internals(out: Path) -> None:
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    full_precision()
    _, questions, request_inputs = requests()
    reference = {record["sample_id"]: record for record in json.loads((out / "answers_float32_fp32.json").read_text())}
    thinkers = {
        name: Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL, dtype=dtype, device_map={"": device}, enable_audio_output=False, experts_implementation="eager"
        ).thinker.eval()
        for name, dtype, device in (("fp32", torch.float32, "cuda:0"), ("bf16", torch.bfloat16, "cuda:1"))
    }
    truth_features = stored_features(out, "fp32")
    report = {}
    for sample, inputs in zip(questions, request_inputs):
        for thinker in thinkers.values():
            patch_vision(thinker, truth_features[sample.sample_id], inputs["image_grid_thw"])
        tokens = reference[sample.sample_id]["tokens"]
        full_ids = torch.cat([inputs["input_ids"][0], torch.tensor(tokens, dtype=torch.long)])[None]
        start = inputs["input_ids"].shape[1] - 1
        traced = {}
        for name, thinker in thinkers.items():
            device = thinker.device
            with torch.inference_mode():
                output = thinker(
                    input_ids=full_ids.to(device),
                    attention_mask=torch.ones_like(full_ids).to(device),
                    pixel_values=inputs["pixel_values"].to(device, thinker.dtype),
                    image_grid_thw=inputs["image_grid_thw"].to(device),
                    use_cache=False,
                    output_hidden_states=True,
                    output_router_logits=True,
                )
            traced[name] = {
                "logits": output.logits[0, start : start + len(tokens)].float().cpu(),
                "hidden": [state[0].float().cpu() for state in output.hidden_states],
                "experts": [
                    logits.float().topk(thinker.config.text_config.num_experts_per_tok, dim=-1).indices.sort(dim=-1).values.cpu()
                    for logits in output.router_logits
                ],
            }
        fp32, bf16 = traced["fp32"], traced["bf16"]
        answer = slice(start, start + len(tokens))
        departures = (bf16["logits"].argmax(dim=-1) != torch.tensor(tokens)).nonzero().flatten().tolist()
        first = departures[0] if departures else None
        top2 = fp32["logits"].topk(2, dim=-1).values
        report[sample.sample_id] = {
            "departures": departures,
            "first_departure": first,
            "fp32_margin_at_first": (top2[first, 0] - top2[first, 1]).item() if first is not None else None,
            "hidden_error_answer": [relative_l2(b[answer], f[answer]) for b, f in zip(bf16["hidden"], fp32["hidden"])],
            "expert_flip_share_answer": [
                (b[answer] != f[answer]).any(dim=-1).float().mean().item() for b, f in zip(bf16["experts"], fp32["experts"])
            ],
            "expert_flip_share_prompt": [
                (b[: start + 1] != f[: start + 1]).any(dim=-1).float().mean().item()
                for b, f in zip(bf16["experts"], fp32["experts"])
            ],
            "at_first": (
                {
                    "hidden_error": [
                        relative_l2(b[start + first], f[start + first]) for b, f in zip(bf16["hidden"], fp32["hidden"])
                    ],
                    "expert_flips": [
                        int((b[start + first] != f[start + first]).any().item())
                        for b, f in zip(bf16["experts"], fp32["experts"])
                    ],
                }
                if first is not None
                else None
            ),
        }
        print(f"{sample.sample_id}: departures {len(departures)}, first {first}", flush=True)
    (out / "internals.json").write_text(json.dumps(report))


def subprocess_on(cards: tuple[int, ...], tree: Path, args: list[str], log_path: Path) -> int:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(card) for card in cards),
        "PYTHONPATH": str(tree),
    }
    with open(log_path, "w") as log:
        return subprocess.run(
            [
                "numactl",
                "--physcpubind=" + ",".join(pair_check.CARD_CPUS[card] for card in cards),
                f"--membind={CARD_NODES[cards[0]]}",
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *args,
            ],
            cwd=tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def in_parallel(jobs: list[tuple[tuple[int, ...], Path, list[str], Path]], out: Path) -> None:
    codes = {}

    def job(index, cards, tree, args, log_path):
        codes[index] = subprocess_on(cards, tree, args, log_path)

    threads = [threading.Thread(target=job, args=(index, *spec)) for index, spec in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with open(out / "run.log", "a") as log:
        for index, (cards, _, args, log_path) in enumerate(jobs):
            log.write(f"{' '.join(args[:3])} cards {cards} rc {codes[index]} ({log_path.name})\n")


def run(arms: list[str], out: Path, with_features: bool) -> None:
    trees = {arm: pair_check.tree(pair_check.ARMS[arm][0]) for arm in arms}
    if with_features:
        in_parallel(
            [((card,), trees[arm], ["features", "--arm", arm, "--out", str(out)], out / f"features_{arm}.log") for card, arm in enumerate(arms)],
            out,
        )
    else:
        pass
    sources = ["fp32", "hf_bf16", *arms]
    answer_tree = trees[ANSWER_TREE]
    in_parallel(
        [
            ((card,), answer_tree, ["answers", "--thinker", "float32", "--sources", source, "--out", str(out)], out / f"answers_float32_{source}.log")
            for card, source in enumerate(sources)
        ],
        out,
    )
    in_parallel(
        [
            ((card,), answer_tree, ["answers", "--thinker", "bfloat16", "--sources", source, "--out", str(out)], out / f"answers_bfloat16_{source}.log")
            for card, source in enumerate(sources)
        ],
        out,
    )
    in_parallel([((0, 1), answer_tree, ["internals", "--out", str(out)], out / "internals.log")], out)
    with open(out / "run.log", "a") as log:
        log.write("done\n")


def read(arms: list[str], out: Path) -> None:
    import statistics

    import torch

    for source in ("fp32", "hf_bf16"):
        copies = [torch.load(path) for path in sorted(out.glob(f"features_{source}_from_*.pt"))]
        if len(copies) == 2:
            differing = [
                sample
                for sample in copies[0]
                if any(not torch.equal(a, b) for a, b in zip(copies[0][sample], copies[1][sample]))
            ]
            print(f"{source} features from the two processes: {len(differing)} of {len(copies[0])} requests differ")
        else:
            pass
    for arm in arms:
        layers = json.loads((out / f"vision_layers_{arm}.json").read_text())
        for name in ("hf_bf16", arm):
            blocks = [statistics.fmean(entry[name]["blocks"][index] for entry in layers.values()) for index in range(27)]
            outputs = [statistics.fmean(entry[name]["outputs"][index] for entry in layers.values()) for index in range(4)]
            print(f"vision {name:8s} (in {arm} tree) blocks " + " ".join(f"{value:.1e}" for value in blocks))
            print(f"vision {name:8s} (in {arm} tree) merged, deepstack " + " ".join(f"{value:.2e}" for value in outputs))
    sources = ["fp32", "hf_bf16", *arms]
    table = {}
    for dtype in ("float32", "bfloat16"):
        for source in sources:
            path = out / f"answers_{dtype}_{source}.json"
            if path.exists():
                table[(dtype, source)] = {record["sample_id"]: record for record in json.loads(path.read_text())}
            else:
                pass
    truth = table.get(("float32", "fp32"), {})
    for key, records in table.items():
        correct = sum(record["is_correct"] for record in records.values())
        agree = sum(records[sample]["predicted"] == truth[sample]["predicted"] for sample in records if sample in truth)
        print(f"thinker {key[0]:8s} features {key[1]:8s}: correct {correct}/20, answers equal to fp32/fp32 {agree}/20")
    if truth:
        print("sample  gold | " + " | ".join(f"{d[:4]}/{s}" for d, s in table))
        for sample in truth:
            print(
                f"{sample.replace('validation_Accounting_', 'A'):6s} {truth[sample]['gold']}  | "
                + " | ".join(
                    f"{table[key][sample]['predicted']}{'*' if table[key][sample]['is_correct'] else ' '} "
                    f"m{table[key][sample]['margin'] if table[key][sample]['margin'] is None else round(table[key][sample]['margin'], 2)}"
                    for key in table
                )
            )
    else:
        pass
    internals_path = out / "internals.json"
    if internals_path.exists():
        report = json.loads(internals_path.read_text())
        layers_count = len(next(iter(report.values()))["expert_flip_share_answer"])
        mean = lambda key: [statistics.fmean(entry[key][index] for entry in report.values()) for index in range(layers_count)]
        print("bf16 vs fp32 thinker, teacher-forced on fp32's tokens:")
        print("  hidden error over the answer per layer " + " ".join(f"{value:.1e}" for value in mean("hidden_error_answer")))
        print("  top-8 expert set differs, answer positions " + " ".join(f"{value:.2f}" for value in mean("expert_flip_share_answer")))
        print("  top-8 expert set differs, prompt positions " + " ".join(f"{value:.2f}" for value in mean("expert_flip_share_prompt")))
        for sample, entry in report.items():
            at_first = entry["at_first"]
            print(
                f"  {sample.replace('validation_Accounting_', 'A')}: departures {len(entry['departures'])}, first {entry['first_departure']}, "
                f"fp32 margin {entry['fp32_margin_at_first']}"
                + (
                    f", layers with expert flips at it {[index for index, flip in enumerate(at_first['expert_flips']) if flip]}"
                    if at_first is not None
                    else ""
                )
            )
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "read", "features", "answers", "internals", "vision_ablation"))
    parser.add_argument("--arms")
    parser.add_argument("--arm")
    parser.add_argument("--thinker", choices=("float32", "bfloat16"))
    parser.add_argument("--sources")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-features", action="store_true", help="reuse the features already under --out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        run(args.arms.split(","), args.out, not args.skip_features)
    elif args.mode == "read":
        read(args.arms.split(","), args.out)
    elif args.mode == "features":
        features(args.arm, args.out)
    elif args.mode == "answers":
        answers(args.thinker, args.sources.split(","), args.out)
    elif args.mode == "vision_ablation":
        vision_ablation(args.out)
    else:
        internals(args.out)


if __name__ == "__main__":
    main()
