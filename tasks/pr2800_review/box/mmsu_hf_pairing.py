"""MMSU understanding (CI stage 5) answered by omni and by HF's own thinker on identical inputs, item by item.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/mmsu_hf_pairing.py --main upw --branch fle
    --out /data/pr2800/<run>
1. Omni: each arm serves the CI's thinker-only config (qwen3_omni_mmmu_h100.yaml) on its card, main on card 0 and the
   branch on card 1, with processor_dump/sitecustomize.py saving what every processor call hands the thinker, and gets
   test_qwen3_omni_mmsu_ci.py's requests (every mmsu-ci-2000 question, text out, greedy, 32 tokens) one at a time.
2. HF: Qwen3-Omni's thinker in bf16 on the image's transformers, on cards 0 and 1 once the servers stop, prompted as
   omni prompts (the audio before the question, the chat template), greedy, 32 tokens, stopping at <|im_end|> as omni
   does. Every clip with the reference audio (librosa.load at 16 kHz, qwen_omni_utils' load of a local path), and every
   clip not stored at 16 kHz again with main's linear array (main's load_audio_path).
3. Scored by the benchmark's own parser. Per arm, correct of 2000, and for each pair fed the same audio (the branch and
   HF on the reference audio, main and HF on main's array): input_ids and features equal, answers equal, the items only
   one side gets right and the exact McNemar p, split into clips stored at 16 kHz and resampled clips.
Writes <out>/<arm>/mmsu.json, <out>/hf_<shard>.json and <out>/summary.txt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REPO_ID = "zhaochenyang20/mmsu-ci-2000"
CONFIG = "examples/configs/qwen3_omni_mmmu_h100.yaml"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_TOKENS = 32
CARDS = (0, 1)
PORT = 34000


def omni_client(port: int, out: Path) -> None:
    from benchmarks.eval.benchmark_omni_mmsu import run

    args = argparse.Namespace(
        base_url=None, host="127.0.0.1", port=port, model="qwen3-omni", modalities="text", output_dir=str(out),
        max_samples=None, task_names=None, categories=None, prompt=None, max_tokens=MAX_TOKENS, temperature=0.0,
        warmup=0, max_concurrency=1, request_rate=float("inf"), timeout_s=300, save_audio=False, disable_tqdm=True,
        seed=None, fingerprint=False, repo_id=REPO_ID, lang="en", asr_device="cuda:0")
    results = asyncio.run(run(args))
    (out / "mmsu.json").write_text(json.dumps(results, indent=1))


def run_omni(out: Path, arm: str, card: int) -> int:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    server_env = {
        **pair_check.card_env(card, arm_tree),
        "PYTHONPATH": f"{BOX / 'processor_dump'}:{arm_tree}",
        "PROCESSOR_DUMP_DIR": str(arm_out / "dumps"),
    }
    pair_check.log(out, f"{arm} card {card} serving")
    server = pair_check.serve(card, arm_tree, MODEL, port, ["--config", CONFIG], arm_out / "server.log", env=server_env)
    try:
        pair_check.log(out, f"{arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--omni-client", "--port", str(port), "--out",
             str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    pair_check.log(out, f"{arm} card {card} client rc {rc}")
    return rc


def hf_items() -> list[tuple[object, str, int]]:
    import soundfile
    from benchmarks.dataset.mmsu import load_mmsu_samples

    samples = load_mmsu_samples(repo_id=REPO_ID)
    items = []
    for sample in samples:
        rate = soundfile.info(sample.audio_path).samplerate
        items.append((sample, "reference", rate))
        if rate != 16000:
            items.append((sample, "main_linear", rate))
        else:
            pass
    return items


def hf_client(shard: int, shards: int, out: Path) -> None:
    import librosa
    import torch
    from benchmarks.dataset.mmsu import normalize_text
    from benchmarks.metrics.accuracy import INDEX_TO_LETTER
    from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT, _build_question_text, _extract_prediction
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    from sglang_omni.preprocessing.audio import load_audio_path

    items = hf_items()[shard::shards]
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    stop_ids = [processor.tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")]
    records = []
    for index, (sample, source, rate) in enumerate(items):
        question = _build_question_text(sample, DEFAULT_PROMPT)
        messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": question}]}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audio = (
            librosa.load(sample.audio_path, sr=16000)[0]
            if source == "reference"
            else load_audio_path(sample.audio_path, target_sr=16000)
        )
        inputs = processor(text=prompt_text, audio=[audio], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            generated = model.thinker.generate(
                **inputs.to("cuda:0"), max_new_tokens=MAX_TOKENS, do_sample=False, repetition_penalty=1.0,
                eos_token_id=stop_ids)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted_index, predicted_answer = _extract_prediction(response, sample.choices)
        records.append(dict(
            sample_id=sample.sample_id, source=source, stored_rate=rate, response=response,
            predicted_choice=INDEX_TO_LETTER.get(predicted_index, ""),
            is_correct=predicted_index == sample.answer_index or (
                bool(predicted_answer) and normalize_text(predicted_answer) == normalize_text(sample.answer_text)),
            input_ids=inputs["input_ids"][0].tolist(),
            features_sha256=hashlib.sha256(inputs["input_features"].cpu().numpy().tobytes()).hexdigest()))
        if index % 100 == 0:
            print(f"{time.strftime('%H:%M:%S')} hf shard {shard} {index}/{len(items)}", flush=True)
        else:
            pass
    (out / f"hf_{shard}.json").write_text(json.dumps(records, indent=1))


def run_hf(out: Path, shard: int, card: int, tree_path: Path) -> int:
    pair_check.log(out, f"hf shard {shard} card {card} start")
    rc = subprocess.run(
        pair_check.pinned(card, [sys.executable, str(Path(__file__).resolve()), "--hf-client", "--shard", str(shard),
                                 "--shards", str(len(CARDS)), "--out", str(out)]),
        cwd=tree_path, env=pair_check.card_env(card, tree_path), stdout=open(out / f"hf_{shard}.log", "w"),
        stderr=subprocess.STDOUT).returncode
    pair_check.log(out, f"hf shard {shard} card {card} rc {rc}")
    return rc


def mcnemar_p(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    else:
        pass
    tail = sum(math.comb(discordant, k) for k in range(min(only_a, only_b) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def dumped_inputs(arm_out: Path) -> dict[tuple[int, ...], set[str]]:
    import torch

    by_prompt: dict[tuple[int, ...], set[str]] = {}
    for path in sorted((arm_out / "dumps").glob("*.pt")):
        dump = torch.load(path, weights_only=False)
        by_prompt.setdefault(tuple(dump["input_ids"][0].tolist()), set()).add(
            hashlib.sha256(dump["input_features"].numpy().tobytes()).hexdigest()
        )
    return by_prompt


def analyze(out: Path, main_arm: str, branch_arm: str) -> None:
    hf_records = [record for shard in range(len(CARDS)) for record in json.loads((out / f"hf_{shard}.json").read_text())]
    hf = {(record["source"], record["sample_id"]): record for record in hf_records}
    omni = {
        arm: {row["sample_id"]: row for row in json.loads((out / arm / "mmsu.json").read_text())["per_sample"]}
        for arm in (main_arm, branch_arm)
    }
    rates = {record["sample_id"]: record["stored_rate"] for record in hf_records}
    ids = sorted(rates)
    lines = []
    for label, correct in (
        (f"omni {main_arm}", {item: omni[main_arm][item]["is_correct"] for item in ids}),
        (f"omni {branch_arm}", {item: omni[branch_arm][item]["is_correct"] for item in ids}),
        ("hf reference audio", {item: hf[("reference", item)]["is_correct"] for item in ids}),
        ("hf main's audio", {
            item: hf[("main_linear" if rates[item] != 16000 else "reference", item)]["is_correct"] for item in ids}),
    ):
        resampled = [item for item in ids if rates[item] != 16000]
        lines.append(
            f"{label}: {sum(correct.values())} / {len(ids)} correct ({100 * sum(correct.values()) / len(ids):.2f} %), "
            f"resampled clips {sum(correct[item] for item in resampled)} / {len(resampled)}, 16 kHz clips "
            f"{sum(correct[item] for item in ids if rates[item] == 16000)} / {len(ids) - len(resampled)}"
        )
    for arm, source in ((branch_arm, "reference"), (main_arm, "main_linear")):
        dumps = dumped_inputs(out / arm)
        for group, members in (
            ("16 kHz", [item for item in ids if rates[item] == 16000]),
            ("resampled", [item for item in ids if rates[item] != 16000]),
        ):
            pairs = [(omni[arm][item], hf[(source if rates[item] != 16000 else "reference", item)]) for item in members]
            inputs_equal = sum(
                hf_row["features_sha256"] in dumps.get(tuple(hf_row["input_ids"]), set()) for _, hf_row in pairs
            )
            only_omni = sum(o["is_correct"] and not h["is_correct"] for o, h in pairs)
            only_hf = sum(h["is_correct"] and not o["is_correct"] for o, h in pairs)
            lines.append(
                f"omni {arm} against hf on the same audio, {group} ({len(pairs)}): input_ids and features equal "
                f"{inputs_equal}, answers equal {sum(o['predicted_choice'] == h['predicted_choice'] for o, h in pairs)}, "
                f"right only on omni {only_omni}, only on hf {only_hf}, McNemar p {mcnemar_p(only_omni, only_hf):.3f}"
            )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        pair_check.log(out, line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main", default="upw", help="main's arm, served on card 0")
    parser.add_argument("--branch", default="fle", help="the branch's arm, served on card 1")
    parser.add_argument("--omni-client", action="store_true")
    parser.add_argument("--hf-client", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    args = parser.parse_args()
    if args.omni_client:
        omni_client(args.port, args.out)
        return
    elif args.hf_client:
        hf_client(args.shard, args.shards, args.out)
        return
    elif args.analyze:
        analyze(args.out, args.main, args.branch)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    main_tree = pair_check.tree(pair_check.ARMS[args.main][0])
    with ThreadPoolExecutor(max_workers=len(CARDS)) as pool:
        codes = list(pool.map(lambda placement: run_omni(args.out, *placement), zip((args.main, args.branch), CARDS)))
    if any(codes):
        pair_check.log(args.out, f"stopped: omni rcs {codes}")
        return
    else:
        pass
    with ThreadPoolExecutor(max_workers=len(CARDS)) as pool:
        codes = list(pool.map(lambda shard: run_hf(args.out, shard, CARDS[shard], main_tree), range(len(CARDS))))
    if any(codes):
        pair_check.log(args.out, f"stopped: hf rcs {codes}")
        return
    else:
        pass
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--analyze", "--out", str(args.out), "--main", args.main,
         "--branch", args.branch],
        cwd=main_tree, env={**pair_check.card_env(0, main_tree), "CUDA_VISIBLE_DEVICES": ""},
        stdout=open(args.out / "analyze.log", "w"), stderr=subprocess.STDOUT).returncode
    pair_check.log(args.out, f"analyze rc {rc}")
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
