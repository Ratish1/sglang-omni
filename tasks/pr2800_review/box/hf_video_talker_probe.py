"""Whether HF's own Qwen3-Omni runs on as omni's talker does on CI stage 10's video prompts.

Run on the box: python3 /workspace/box7/tasks/pr2800_review/box/hf_video_talker_probe.py --out /data/pr2800/<run>
1. Main's server (ups, the colocated bf16 config with CI's 32768-token preprocessing and thinker) on card 0 with
   processor_dump/sitecustomize.py gets stage 10's 10 Video-AMME requests (video at 2 fps, at most 128 frames of at
   most 401408 pixels, the spoken question), so every tensor the processor hands the thinker is saved.
2. HF's Qwen3OmniMoeForConditionalGeneration (bf16) on cards 0 to 3, the samples split across them, generates speech
   from exactly those tensors DRAWS times a sample, each draw under its own torch seed: the thinker greedy at 256
   tokens, the talker with the checkpoint's sampling (0.9, top_k 50, repetition 1.05), speaker Ethan.
3. One Qwen3-ASR server on card 0 transcribes every HF output and scores it against HF's text as CI does.
Writes <out>/hf/generated.json, <out>/hf/audio/*.wav, <out>/hf/wer.json and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REPO_ID = "zhaochenyang20/Video_AMME_ci"
SAMPLES = 10
DRAWS = 6
HF_TALKER_MAX_FRAMES = 1024
HF_CARDS = (0, 1, 2, 3)
PORT = 35000
SERVE_ARGS = [
    "--config", pair_check.CONFIG, "--colocate",
    "--preprocessing.factory.max_seq_len", "32768", "--thinker.factory.max_seq_len", "32768",
]


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def dump_client(port: int, out: Path) -> None:
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

    config = VideoEvalConfig(
        model="qwen3-omni", host="127.0.0.1", port=port, max_samples=SAMPLES, max_tokens=256, max_concurrency=1,
        output_dir=str(out), repo_id=REPO_ID, video_fps=2, video_max_frames=128, video_max_pixels=401408,
        disable_tqdm=True, timeout_s=500, warmup=0)
    results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
    (out / "videoamme.json").write_text(json.dumps(results, indent=1, default=str))


def hf_client(shard: int, shards: int, out: Path) -> None:
    import soundfile
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    omni = json.loads((out / "omni" / "videoamme.json").read_text())["per_sample"]
    order = [sample["sample_id"] for sample in omni]
    dumps = sorted((out / "omni" / "dumps").glob("*.pt"), key=lambda path: path.stat().st_mtime)
    dumps = [torch.load(path, weights_only=False) for path in dumps]
    dumps = [dump for dump in dumps if "pixel_values_videos" in dump["outputs"]]
    assert len(dumps) == len(order), f"{len(dumps)} video captures for {len(order)} requests"
    processor = Qwen3OmniMoeProcessor.from_pretrained(pair_check.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        pair_check.MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    # note (ratish): transformers 5.19's talker reuses the thinker's rope index, whose audio branch reads
    # config.audio_config.n_window, which the talker's config lacks; the thinker's audio config supplies it.
    model.talker.config.audio_config = model.config.thinker_config.audio_config
    (out / "hf" / "audio").mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id, dump in list(zip(order, dumps))[shard::shards]:
        inputs = {name: value.to("cuda:0") for name, value in dump["outputs"].items()}
        for draw in range(DRAWS):
            torch.manual_seed(draw)
            start = time.perf_counter()
            with torch.inference_mode():
                text_ids, wav = model.generate(
                    **inputs, speaker="Ethan", thinker_do_sample=False, thinker_max_new_tokens=256,
                    talker_max_new_tokens=HF_TALKER_MAX_FRAMES)
            seconds = time.perf_counter() - start
            sequence = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
            text = processor.batch_decode(
                sequence[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            wav = wav.reshape(-1).float().cpu().numpy()
            path = out / "hf" / "audio" / f"{sample_id}__{draw}.wav"
            soundfile.write(path, wav, 24000)
            records.append(dict(sample_id=f"{sample_id}__{draw}", raw_response=text, wav_path=str(path),
                                is_success=True, latency_s=round(seconds, 2),
                                audio_duration_s=round(len(wav) / 24000, 2)))
            print(f"{time.strftime('%H:%M:%S')} {sample_id} draw {draw}: {len(wav) / 24000:.1f} s audio, "
                  f"{seconds:.0f} s", flush=True)
    (out / "hf" / f"generated_{shard}.json").write_text(json.dumps(records, indent=1))


def score(port: int, out: Path) -> None:
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    records = [record for shard in range(len(HF_CARDS))
               for record in json.loads((out / "hf" / f"generated_{shard}.json").read_text())]
    (out / "hf" / "generated.json").write_text(json.dumps(records, indent=1))
    wer = compute_text_audio_consistency_from_records(
        records, "en", "cuda:0", audio_dir=str(out / "hf" / "audio"), asr_router_port=port, asr_concurrency=4)
    (out / "hf" / "wer.json").write_text(json.dumps(wer, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-client", action="store_true")
    parser.add_argument("--hf-client", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    args = parser.parse_args()
    if args.dump_client:
        dump_client(args.port, args.out / "omni")
        return
    elif args.hf_client:
        hf_client(args.shard, args.shards, args.out)
        return
    elif args.score:
        score(args.port, args.out)
        return
    else:
        pass
    (args.out / "omni").mkdir(parents=True, exist_ok=True)
    tree_path = pair_check.tree("ups")
    if (args.out / "omni" / "videoamme.json").exists():
        log(args.out, "omni captures present, reused")
    else:
        rc = capture(args.out, tree_path)
        if rc != 0:
            return
        else:
            pass

    def run_hf(shard: int, card: int) -> int:
        code = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--hf-client", "--shard", str(shard), "--shards",
             str(len(HF_CARDS)), "--out", str(args.out)],
            cwd=tree_path, env=pair_check.card_env(card, tree_path),
            stdout=open(args.out / f"hf_{shard}.log", "w"), stderr=subprocess.STDOUT).returncode
        log(args.out, f"hf shard {shard} card {card} rc {code}")
        return code

    with ThreadPoolExecutor(max_workers=len(HF_CARDS)) as pool:
        codes = list(pool.map(lambda placement: run_hf(*placement), enumerate(HF_CARDS)))
    if any(codes):
        return
    else:
        pass
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = PORT + 1000
    asr = pair_check.serve(0, asr_tree, pair_check.ASR_MODEL, asr_port, [], args.out / "asr_server.log")
    try:
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--score", "--port", str(asr_port), "--out",
             str(args.out)],
            cwd=asr_tree, env=pair_check.card_env(0, asr_tree), stdout=open(args.out / "score.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(0, asr)
    wer = json.loads((args.out / "hf" / "wer.json").read_text())["summary"] if rc == 0 else None
    (args.out / "summary.json").write_text(json.dumps(dict(score_rc=rc, wer=wer), indent=1))
    log(args.out, f"score rc {rc} wer {wer}")
    log(args.out, "done")


def capture(out: Path, tree_path: Path) -> int:
    server_env = {
        **pair_check.card_env(0, tree_path),
        "PYTHONPATH": f"{BOX / 'processor_dump'}:{tree_path}",
        "PROCESSOR_DUMP_DIR": str(out / "omni" / "dumps"),
    }
    log(out, "ups card 0 serving")
    server = pair_check.serve(0, tree_path, pair_check.MODEL, PORT, SERVE_ARGS, out / "omni" / "server.log",
                              env=server_env)
    try:
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dump-client", "--port", str(PORT), "--out",
             str(out)],
            cwd=tree_path, env=pair_check.card_env(0, tree_path), stdout=open(out / "omni" / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(0, server)
    log(out, f"omni capture rc {rc}, captures {len(list((out / 'omni' / 'dumps').glob('*.pt')))}")
    return rc


if __name__ == "__main__":
    main()
