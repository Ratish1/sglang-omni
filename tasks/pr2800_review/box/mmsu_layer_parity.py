"""CI stage 5 (MMSU understanding) through the served thinker, boundary by boundary against HF's thinker in fp32.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/mmsu_layer_parity.py --arm fle --out /data/pr2800/<run>
1. Card 0 serves the arm on the CI's thinker-only config with layer_dump/sitecustomize.py, which records per request
   the processor's output, the audio embeddings the thinker receives, the composed prefill input embeddings, the M-RoPE
   positions and delta, and the next-token logits; every mmsu-ci-2000 question goes one at a time with the CI's request
   (mmsu_hf_pairing.omni_client).
2. HF's thinker (the image's transformers) runs each request's own prompt ids and features in fp32 on card 0 (the
   truth) and in bf16 on card 1 (the floor), hooking the audio tower's output, the text model's inputs_embeds and
   position_ids, the rope delta and the last position's logits.
3. Per request: positions and delta equal to HF's; for the audio embeddings, the input embeddings and the logits, the
   relative L2 to fp32 of the served value and of HF bf16's, and their ratio; the next token against fp32's, and the
   fp32 margin of fp32's top token over the served one where they differ. Writes <out>/parity.jsonl and
   <out>/summary.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import mmsu_hf_pairing  # noqa: E402
import pair_check  # noqa: E402

CONFIG = "examples/configs/qwen3_omni_mmmu_h100.yaml"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT = 35000
HF_RUNS = (("fp32", 0), ("bf16", 1))


def serve_and_record(out: Path, arm: str) -> int:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    server_env = {
        **pair_check.card_env(0, arm_tree),
        "PYTHONPATH": f"{BOX / 'layer_dump'}:{arm_tree}",
        "LAYER_DUMP_DIR": str(arm_out / "layers"),
        "PROCESSOR_DUMP_DIR": str(arm_out / "dumps"),
    }
    pair_check.log(out, f"{arm} card 0 serving")
    server = pair_check.serve(0, arm_tree, MODEL, PORT, ["--config", CONFIG], arm_out / "server.log", env=server_env)
    try:
        pair_check.log(out, f"{arm} card 0 healthy, client")
        rc = subprocess.run(
            [sys.executable, str(BOX / "mmsu_hf_pairing.py"), "--omni-client", "--port", str(PORT), "--out",
             str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(0, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(0, server)
    pair_check.log(out, f"{arm} card 0 client rc {rc}")
    return rc


def requests(arm_out: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for inputs in sorted((arm_out / "layers").glob("*_inputs.pt")):
        logits = inputs.with_name(inputs.name.replace("_inputs.pt", "_logits.pt"))
        if logits.exists():
            pairs.append((inputs, logits))
        else:
            pass
    return pairs


def hf_client(arm_out: Path, dtype_name: str, out: Path, reverse: bool) -> None:
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
    features = {}
    for path in (arm_out / "dumps").glob("*.pt"):
        dump = torch.load(path, weights_only=False)
        features[tuple(dump["input_ids"][0].tolist())] = (dump["input_features"], dump["feature_attention_mask"])
    # note (ratish): the truth runs eager experts in fp32; the floor is HF's default bf16 path.
    precision_kwargs = {"experts_implementation": "eager"} if dtype_name == "fp32" else {}
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=dtype, device_map="cuda:0", enable_audio_output=False, **precision_kwargs
    ).eval()
    thinker = model.thinker
    captured = {}
    thinker.audio_tower.register_forward_hook(
        lambda module, args, output: captured.__setitem__("audio", output.last_hidden_state.detach())
    )
    thinker.model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(
            inputs_embeds=kwargs["inputs_embeds"].detach(), position_ids=kwargs["position_ids"].detach()
        ),
        with_kwargs=True,
    )
    results_dir = out / f"hf_{dtype_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    audio_token_id = model.config.thinker_config.audio_token_id
    vocab_size = model.config.thinker_config.text_config.vocab_size
    # note (ratish): two processes of one dtype split the requests by walking from opposite ends.
    pairs = requests(arm_out)[::-1] if reverse else requests(arm_out)
    for index, (inputs_path, _) in enumerate(pairs):
        if (results_dir / inputs_path.name.replace("_inputs.pt", ".pt")).exists():
            continue
        else:
            pass
        served = torch.load(inputs_path, weights_only=False)
        # note (ratish): the thinker's prompt carries the audio's cache pad value at its audio rows.
        prompt_ids = [
            audio_token_id if token_id >= vocab_size else token_id for token_id in served["origin_input_ids"]
        ]
        if tuple(prompt_ids) not in features:
            print(f"no processor call for {inputs_path.name} (the server's warmup), skipped", flush=True)
            continue
        else:
            pass
        input_ids = torch.tensor([prompt_ids])
        input_features, feature_mask = features[tuple(prompt_ids)]
        captured.clear()
        thinker.rope_deltas = None
        with torch.no_grad():
            output = thinker(
                input_ids=input_ids.to("cuda:0"),
                attention_mask=torch.ones_like(input_ids).to("cuda:0"),
                input_features=input_features.to("cuda:0", dtype),
                feature_attention_mask=feature_mask.to("cuda:0"),
                use_cache=False,
            )
        torch.save(
            dict(
                audio=captured["audio"].float().cpu(),
                inputs_embeds=captured["inputs_embeds"][0].float().cpu(),
                position_ids=captured["position_ids"].cpu(),
                rope_deltas=thinker.rope_deltas.cpu(),
                logits=output.logits[0, -1].float().cpu(),
            ),
            results_dir / inputs_path.name.replace("_inputs.pt", ".pt"),
        )
        if index % 200 == 0:
            print(f"{time.strftime('%H:%M:%S')} hf {dtype_name} {index}/{len(pairs)}", flush=True)
        else:
            pass


def run_hf(out: Path, arm: str, dtype_name: str, card: int, tree_path: Path) -> int:
    pair_check.log(out, f"hf {dtype_name} card {card} start")
    rc = subprocess.run(
        pair_check.pinned(card, [sys.executable, str(Path(__file__).resolve()), "--hf-client", "--dtype", dtype_name,
                                 "--arm", arm, "--out", str(out)]),
        cwd=tree_path, env=pair_check.card_env(card, tree_path), stdout=open(out / f"hf_{dtype_name}.log", "w"),
        stderr=subprocess.STDOUT).returncode
    pair_check.log(out, f"hf {dtype_name} card {card} rc {rc}")
    return rc


def relative_l2(value, truth) -> float:
    return float((value.float() - truth.float()).norm() / truth.float().norm().clamp_min(1e-30))


def analyze(out: Path, arm: str) -> None:
    import torch

    rows = []
    for inputs_path, logits_path in requests(out / arm):
        served = torch.load(inputs_path, weights_only=False)
        served_logits = torch.load(logits_path, weights_only=False)
        name = inputs_path.name.replace("_inputs.pt", ".pt")
        if not (out / "hf_fp32" / name).exists():
            continue
        else:
            pass
        truth = torch.load(out / "hf_fp32" / name, weights_only=False)
        floor = torch.load(out / "hf_bf16" / name, weights_only=False)
        served_positions = served["mrope_positions"]
        hf_positions = truth["position_ids"][:, 0, :]
        served_top = int(served_logits["logits"].argmax())
        truth_top = int(truth["logits"].argmax())
        floor_top = int(floor["logits"].argmax())
        row = dict(
            rid=served["rid"],
            prompt_tokens=len(served["origin_input_ids"]),
            single_chunk=served["prefix"] == 0 and served["input_embeds"].shape[0] == len(served["origin_input_ids"]),
            positions_equal=bool(
                served_positions is not None
                and served_positions.shape == hf_positions.shape
                and torch.equal(served_positions.long(), hf_positions.long())
            ),
            delta_equal=bool(
                served["mrope_position_delta"] is not None
                and int(served["mrope_position_delta"].flatten()[0]) == int(truth["rope_deltas"].flatten()[0])
            ),
            audio_rows=int(truth["audio"].shape[0]),
            audio_shape_equal=served["audio_embeds"] is not None
            and tuple(served["audio_embeds"].shape) == tuple(truth["audio"].shape),
            audio_served=relative_l2(served["audio_embeds"], truth["audio"]),
            audio_floor=relative_l2(floor["audio"], truth["audio"]),
            embeds_served=relative_l2(served["input_embeds"], truth["inputs_embeds"]),
            embeds_floor=relative_l2(floor["inputs_embeds"], truth["inputs_embeds"]),
            logits_served=relative_l2(served_logits["logits"], truth["logits"]),
            logits_floor=relative_l2(floor["logits"], truth["logits"]),
            served_top=served_top,
            sampled=served_logits["next_token_id"],
            truth_top=truth_top,
            floor_top=floor_top,
            served_margin_in_truth=float(truth["logits"][truth_top] - truth["logits"][served_top]),
            floor_margin_in_truth=float(truth["logits"][truth_top] - truth["logits"][floor_top]),
        )
        rows.append(row)
    (out / "parity.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def quantiles(values: list[float]) -> str:
        ordered = sorted(values)
        return " / ".join(f"{ordered[int(q * (len(ordered) - 1))]:.4g}" for q in (0.5, 0.9, 0.99, 1.0))

    lines = [
        f"{len(rows)} requests, single prefill chunk {sum(row['single_chunk'] for row in rows)}; positions equal to "
        f"HF's {sum(row['positions_equal'] for row in rows)}, delta equal {sum(row['delta_equal'] for row in rows)}, "
        f"audio rows equal in shape {sum(row['audio_shape_equal'] for row in rows)}; sampled token is the served "
        f"argmax {sum(row['sampled'] == row['served_top'] for row in rows)}",
        "relative L2 to HF fp32, median / p90 / p99 / max (served, then HF bf16):",
    ]
    for boundary in ("audio", "embeds", "logits"):
        served = [row[f"{boundary}_served"] for row in rows]
        floor = [row[f"{boundary}_floor"] for row in rows]
        ratios = [s / f for s, f in zip(served, floor) if f > 0]
        lines.append(
            f"  {boundary}: served {quantiles(served)}; HF bf16 {quantiles(floor)}; served over bf16 "
            f"{quantiles(ratios) if ratios else 'n/a (bf16 equals fp32)'}"
        )
    served_miss = [row for row in rows if row["served_top"] != row["truth_top"]]
    floor_miss = [row for row in rows if row["floor_top"] != row["truth_top"]]
    lines.append(
        f"next token equal to fp32's: served {len(rows) - len(served_miss)}, HF bf16 {len(rows) - len(floor_miss)}; "
        f"fp32 margin over the served token where they differ, median / max "
        f"{quantiles([row['served_margin_in_truth'] for row in served_miss]) if served_miss else 'none'}; over HF "
        f"bf16's {quantiles([row['floor_margin_in_truth'] for row in floor_miss]) if floor_miss else 'none'}"
    )
    worst = sorted(rows, key=lambda row: row["logits_served"] / max(row["logits_floor"], 1e-30), reverse=True)[:8]
    lines.append("largest served-over-bf16 logits ratios:")
    lines += [
        f"  {row['rid']}: logits {row['logits_served']:.4g} / {row['logits_floor']:.4g}, embeds "
        f"{row['embeds_served']:.4g} / {row['embeds_floor']:.4g}, audio {row['audio_served']:.4g} / "
        f"{row['audio_floor']:.4g}, tokens served {row['served_top']} truth {row['truth_top']}"
        for row in worst
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        pair_check.log(out, line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="fle")
    parser.add_argument("--hf-client", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--dtype", choices=("fp32", "bf16"))
    parser.add_argument("--reuse-served", action="store_true", help="skip serving; read <out>/<arm>'s dumps")
    parser.add_argument("--reverse", action="store_true", help="internal: walk the requests from the last")
    args = parser.parse_args()
    if args.hf_client:
        hf_client(args.out / args.arm, args.dtype, args.out, args.reverse)
        return
    elif args.analyze:
        analyze(args.out, args.arm)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[args.arm][0])
    if not args.reuse_served and serve_and_record(args.out, args.arm) != 0:
        pair_check.log(args.out, "stopped: the served pass failed")
        return
    else:
        pass
    with ThreadPoolExecutor(max_workers=len(HF_RUNS)) as pool:
        codes = list(pool.map(lambda run: run_hf(args.out, args.arm, *run, arm_tree), HF_RUNS))
    if any(codes):
        pair_check.log(args.out, f"stopped: hf rcs {codes}")
        return
    else:
        pass
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--analyze", "--arm", args.arm, "--out", str(args.out)],
        cwd=arm_tree, env={**pair_check.card_env(0, arm_tree), "CUDA_VISIBLE_DEVICES": ""},
        stdout=open(args.out / "analyze.log", "w"), stderr=subprocess.STDOUT).returncode
    pair_check.log(args.out, f"analyze rc {rc}")
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
