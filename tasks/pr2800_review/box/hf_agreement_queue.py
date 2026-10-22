"""Whether omni's linear or band-limited audio loading gives HF's answers, checked from the audio up: MMSU in the CI's
understanding shape, one request at a time.

Run on the box: python3 /workspace/box4/tasks/pr2800_review/box/hf_agreement_queue.py --out /data/pr2800/<run>
At the same time:
1. ups (main, linear resampling) on card 0 and phx (#2861, band-limited) on card 1 serve the thinker-only CI config
   (qwen3_omni_mmmu_h100.yaml) with processor_dump/sitecustomize.py, so every processor call saves the audio array,
   input_ids and input_features the thinker gets; the full mmsu-ci-2000 at c1, 32 tokens, greedy, the default
   prompt.
2. HF's Qwen3-Omni thinker (bf16, eager) on cards 2 and 3, half the clips each: every clip not stored at 16 kHz and
   the first 200 at 16 kHz (the control, where both omni arms load the same samples), the prompt built as omni builds
   it (audio placeholder before the question, HF's chat template), the audio loaded by librosa.load at 16 kHz as
   HF's reference does, greedy, 32 tokens, repetition penalty 1.0.
Then --analyze compares, per clip: each arm's audio array and features against HF's, the input_ids, and each arm's
answer against HF's. Writes <out>/analysis.json and <out>/analysis.txt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
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
MAX_TOKENS = 32
CONTROL_COUNT = 200
OMNI_ARMS = (("ups", 0), ("phx", 1))
HF_CARDS = (2, 3)
PORT = 33000
EXPECTED_TREES = {"ups": "4743b286", "phx": "6ffbdb09"}


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


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
    log(out, f"{arm} card {card} serving")
    server = pair_check.serve(
        card, arm_tree, pair_check.MODEL, port, ["--config", CONFIG], arm_out / "server.log", env=server_env)
    try:
        log(out, f"{arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--omni-client", "--port", str(port), "--out",
             str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    log(out, f"{arm} card {card} client rc {rc}")
    return rc


def selected_samples() -> tuple[list[object], dict[str, int]]:
    import soundfile
    from benchmarks.dataset.mmsu import load_mmsu_samples

    samples = load_mmsu_samples(repo_id=REPO_ID)
    rates = {
        sample.sample_id: soundfile.info(io.BytesIO(Path(sample.audio_path).read_bytes())).samplerate
        for sample in samples
    }
    resampled = [sample for sample in samples if rates[sample.sample_id] != 16000]
    control = [sample for sample in samples if rates[sample.sample_id] == 16000][:CONTROL_COUNT]
    return [*resampled, *control], rates


def hf_client(shard: int, shards: int, out: Path) -> None:
    import librosa
    import torch
    from benchmarks.dataset.mmsu import normalize_text
    from benchmarks.metrics.accuracy import INDEX_TO_LETTER
    from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT, _build_question_text, _extract_prediction
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    samples, rates = selected_samples()
    samples = samples[shard::shards]
    processor = Qwen3OmniMoeProcessor.from_pretrained(pair_check.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        pair_check.MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    records = []
    for index, sample in enumerate(samples):
        question = _build_question_text(sample, DEFAULT_PROMPT)
        messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": question}]}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audio = librosa.load(sample.audio_path, sr=16000)[0]
        inputs = processor(text=prompt_text, audio=[audio], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            generated = model.thinker.generate(
                **inputs.to("cuda:0"), max_new_tokens=MAX_TOKENS, do_sample=False, repetition_penalty=1.0)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted_index, predicted_answer = _extract_prediction(response, sample.choices)
        records.append(dict(
            sample_id=sample.sample_id, native_rate=rates[sample.sample_id], prompt_text=prompt_text,
            audio_samples=int(audio.shape[0]), response=response,
            predicted_choice=INDEX_TO_LETTER.get(predicted_index, ""),
            is_correct=predicted_index == sample.answer_index or (
                bool(predicted_answer) and normalize_text(predicted_answer) == normalize_text(sample.answer_text)),
            input_ids=inputs["input_ids"][0].tolist(),
            features_sha256=hashlib.sha256(inputs["input_features"].cpu().numpy().tobytes()).hexdigest()))
        if index % 100 == 0:
            print(f"{time.strftime('%H:%M:%S')} hf shard {shard} {index}/{len(samples)}", flush=True)
        else:
            pass
    (out / f"hf_{shard}.json").write_text(json.dumps(records, indent=1))


def run_hf(out: Path, shard: int, card: int, tree_path: Path) -> int:
    log(out, f"hf shard {shard} card {card} start")
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--hf-client", "--shard", str(shard), "--shards",
         str(len(HF_CARDS)), "--out", str(out)],
        cwd=tree_path, env=pair_check.card_env(card, tree_path), stdout=open(out / f"hf_{shard}.log", "w"),
        stderr=subprocess.STDOUT).returncode
    log(out, f"hf shard {shard} card {card} rc {rc}")
    return rc


def analyze(out: Path) -> None:
    import librosa
    import numpy as np
    import torch

    hf = {record["sample_id"]: record for shard in range(len(HF_CARDS))
          for record in json.loads((out / f"hf_{shard}.json").read_text())}
    samples, _ = selected_samples()
    by_id = {sample.sample_id: sample for sample in samples}
    lines = []
    summary: dict[str, object] = {}
    for arm, _ in OMNI_ARMS:
        answers = {item["sample_id"]: item for item in
                   json.loads((out / arm / "mmsu.json").read_text())["per_sample"]}
        dumps: dict[str, list[dict[str, object]]] = {}
        for path in sorted((out / arm / "dumps").glob("*.pt")):
            dump = torch.load(path, weights_only=False)
            dumps.setdefault(dump["text"], []).append(dump)
        groups = {"resampled": [], "16 kHz": []}
        for sample_id, reference in hf.items():
            group = "16 kHz" if reference["native_rate"] == 16000 else "resampled"
            candidates = dumps.get(reference["prompt_text"], [])
            dump = min(candidates, key=lambda item: abs(len(item["audio"][0]) - reference["audio_samples"]),
                       default=None)
            hf_audio = librosa.load(by_id[sample_id].audio_path, sr=16000)[0]
            if dump is None:
                groups[group].append(dict(sample_id=sample_id, has_dump=False))
                continue
            else:
                pass
            audio = dump["audio"][0]
            length = min(len(audio), len(hf_audio))
            features_sha = hashlib.sha256(dump["input_features"].numpy().tobytes()).hexdigest()
            groups[group].append(dict(
                sample_id=sample_id, has_dump=True,
                audio_equal=len(audio) == len(hf_audio) and bool(np.array_equal(audio, hf_audio)),
                audio_length_delta=len(audio) - len(hf_audio),
                audio_max_abs_diff=float(np.max(np.abs(audio[:length] - hf_audio[:length]))) if length else 0.0,
                input_ids_equal=dump["input_ids"][0].tolist() == reference["input_ids"],
                features_equal=features_sha == reference["features_sha256"],
                answer_equal=answers[sample_id]["predicted_choice"] == reference["predicted_choice"],
                omni_correct=bool(answers[sample_id]["is_correct"]), hf_correct=bool(reference["is_correct"])))
        for group, rows in groups.items():
            dumped = [row for row in rows if row["has_dump"]]
            entry = dict(
                clips=len(rows), dumped=len(dumped),
                audio_equal=sum(row["audio_equal"] for row in dumped),
                audio_max_abs_diff_median=float(np.median([row["audio_max_abs_diff"] for row in dumped])) if dumped else None,
                length_deltas=sorted({row["audio_length_delta"] for row in dumped}),
                input_ids_equal=sum(row["input_ids_equal"] for row in dumped),
                features_equal=sum(row["features_equal"] for row in dumped),
                answers_equal_hf=sum(row["answer_equal"] for row in dumped),
                omni_correct=sum(row["omni_correct"] for row in dumped),
                hf_correct=sum(row["hf_correct"] for row in dumped))
            summary[f"{arm} {group}"] = entry
            lines.append(f"{arm} {group}: {entry}")
    (out / "analysis.json").write_text(json.dumps(summary, indent=1))
    (out / "analysis.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        log(out, line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
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
        analyze(args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED_TREES.items():
        path = pair_check.tree(name)
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"], cwd=path, capture_output=True, text=True).stdout.strip()
        log(args.out, f"{name} sglang_omni {tree_hash} expected {expected}")
        if not tree_hash.startswith(expected):
            log(args.out, "stopped: the tree is not the Mac's")
            return
        else:
            pass
    hf_tree = pair_check.tree("ups")
    with ThreadPoolExecutor(max_workers=len(OMNI_ARMS) + len(HF_CARDS)) as pool:
        omni_jobs = [pool.submit(run_omni, args.out, arm, card) for arm, card in OMNI_ARMS]
        hf_jobs = [pool.submit(run_hf, args.out, shard, card, hf_tree) for shard, card in enumerate(HF_CARDS)]
        codes = [job.result() for job in [*omni_jobs, *hf_jobs]]
    if any(codes):
        log(args.out, f"stopped: rcs {codes}")
        return
    else:
        pass
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--analyze", "--out", str(args.out)],
        cwd=hf_tree, env={**pair_check.card_env(0, hf_tree)}, stdout=open(args.out / "analyze.log", "w"),
        stderr=subprocess.STDOUT).returncode
    log(args.out, f"analyze rc {rc}")
    log(args.out, "done")


if __name__ == "__main__":
    main()
