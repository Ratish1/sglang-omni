"""Whether two trees' thinkers compute the same text for the same input, with no batching: MMSU at c1, text out.

Run on the box: python3 /workspace/box2/tasks/pr2800_review/box/thinker_text_c1_probe.py --arms ups@0,phx@1
    --out /data/pr2800/<run>
Each arm serves the colocated bf16 speech config on its card (pair_check.serve) and gets, one at a time, the MMSU
talker prompt on the first 200 mmsu-ci-2000 clips stored at 16 kHz (so both arms load the same samples), text out,
greedy, 256 tokens. Equal text on every prompt means the thinker's computation is the same in both trees; a prompt
whose text differs is a computation difference, not batching. Writes <out>/<arm>/mmsu.json and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
from mmsu_talker_pair import MAX_TOKENS, MMSU_TTS_PROMPT, REPO_ID  # noqa: E402

SAMPLE_COUNT = 200
PORT = 31000


def client(port: int, out: Path) -> None:
    import soundfile
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.eval.benchmark_omni_mmsu import run

    samples = [
        sample for sample in load_mmsu_samples(repo_id=REPO_ID)
        if soundfile.info(io.BytesIO(Path(sample.audio_path).read_bytes())).samplerate == 16000
    ][:SAMPLE_COUNT]
    args = argparse.Namespace(
        base_url=None, host="127.0.0.1", port=port, model="qwen3-omni", modalities="text", output_dir=str(out),
        max_samples=None, task_names=None, categories=None, prompt=MMSU_TTS_PROMPT, max_tokens=MAX_TOKENS,
        temperature=0.0, warmup=0, max_concurrency=1, request_rate=float("inf"), save_audio=False,
        disable_tqdm=True, seed=None, fingerprint=False, lang="en", asr_device="cuda:0", timeout_s=500)
    results = asyncio.run(run(args, samples=samples, compute_wer=False))
    (out / "mmsu.json").write_text(json.dumps(results, indent=1))


def run_arm(out: Path, arm: str, card: int) -> int:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    pair_check.log(out, f"{arm} card {card} serving")
    server = pair_check.serve(card, arm_tree, pair_check.MODEL, port, ["--config", pair_check.CONFIG, "--colocate"],
                              arm_out / "server.log")
    try:
        pair_check.log(out, f"{arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--client", "--port", str(port), "--out", str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    pair_check.log(out, f"{arm} card {card} client rc {rc}")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arm@card,arm@card")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    with ThreadPoolExecutor(max_workers=len(placements)) as pool:
        codes = list(pool.map(lambda placement: run_arm(args.out, *placement), placements))
    if any(codes):
        pair_check.log(args.out, f"client rcs {codes}")
        return
    else:
        pass
    (first, _), (second, _) = placements
    texts = [
        {sample["sample_id"]: sample["raw_response"] for sample in
         json.loads((args.out / arm / "mmsu.json").read_text())["per_sample"]}
        for arm in (first, second)
    ]
    differing = sorted(sample_id for sample_id in texts[0] if texts[0][sample_id] != texts[1].get(sample_id))
    summary = dict(prompts=len(texts[0]), differing=len(differing), differing_ids=differing)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    pair_check.log(args.out, f"{len(differing)} of {len(texts[0])} texts differ")
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
