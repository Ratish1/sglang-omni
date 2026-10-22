"""Whether a stage 10 talker failure follows its request's sampling seed: each failed request replayed alone.

Run on the box: python3 /workspace/boxN/tasks/pr2800_review/box/videoamme_replay.py --arm ups --cards 0+1
    --replay "002-1=f0a1e487-...,003-2=5913a3e0-..." --repeats 3 --out /data/pr2800/<run>
An unseeded request's talker draws with seed xxh64(request id) & 0x7FFFFFFF. One server of the arm in stage 10's shape
(videoamme_talker_tp2.serve) gets each listed sample alone at c1, --repeats times, with the seed the failed request
drew. The thinker is greedy, so a replay whose text matches the failed request's text gives the talker the same input
and the same draws at batch size 1. One Qwen3-ASR server on the first card scores every replay as CI does. Writes
<out>/<sample>__<repeat>/videoamme.json and wer.json, and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xxhash

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
import videoamme_talker_tp2 as stage10  # noqa: E402

SEED_MASK = 0x7FFFFFFF


def run_client(arm_tree: Path, port: int, out: Path, sample_id: str, seed: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(BOX / "videoamme_talker_tp2.py"), "--client", "--port", str(port), "--out", str(out),
         "--concurrency", "1", "--seed", str(seed), "--sample-ids", sample_id],
        cwd=arm_tree, env={**os.environ, "PYTHONPATH": str(arm_tree)},
        stdout=open(out / "client.log", "w"), stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cards", required=True, help="a+b: the pair the server runs on")
    parser.add_argument("--replay", required=True, help="comma-separated sample=request id of each failed request")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=stage10.MODEL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cards = tuple(int(card) for card in args.cards.split("+"))
    replays = [entry.split("=", 1) for entry in args.replay.split(",")]
    arm_tree = pair_check.tree(pair_check.ARMS[args.arm][0])
    port = stage10.PORT + 200 * cards[0]
    stage10.log(args.out, f"{args.arm} cards {cards} serving")
    server = stage10.serve(arm_tree, args.out / "server.log", cards, port, None, args.model)
    runs = []
    try:
        for sample_id, request_id in replays:
            seed = xxhash.xxh64_intdigest(request_id.encode("utf-8")) & SEED_MASK
            for repeat in range(args.repeats):
                run_dir = args.out / f"{sample_id}__{repeat}"
                rc = run_client(arm_tree, port, run_dir, sample_id, seed)
                stage10.log(args.out, f"{sample_id} seed {seed} repeat {repeat} client rc {rc}")
                runs.append((run_dir, rc))
    finally:
        stage10.stop(server, cards)
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = pair_check.ASR_PORT + 1000 + 200 * cards[0]
    asr = pair_check.serve(cards[0], asr_tree, pair_check.ASR_MODEL, asr_port, [], args.out / "asr_server.log")
    summary = {}
    try:
        for run_dir, rc in runs:
            if rc != 0:
                continue
            else:
                pass
            score_rc = subprocess.run(
                [sys.executable, str(BOX / "videoamme_talker_tp2.py"), "--score", "--port", str(asr_port), "--out",
                 str(run_dir)],
                cwd=asr_tree, env=pair_check.card_env(cards[0], asr_tree), stdout=open(run_dir / "score.log", "w"),
                stderr=subprocess.STDOUT).returncode
            if score_rc != 0:
                stage10.log(args.out, f"{run_dir.name} score rc {score_rc}")
                continue
            else:
                pass
            record = json.loads((run_dir / "wer.json").read_text())["per_sample"][0]
            text = json.loads((run_dir / "videoamme.json").read_text())["per_sample"][0]["raw_response"]
            summary[run_dir.name] = dict(wer=record["wer"], audio_s=record["audio_duration_s"], text=text,
                                         hyp=record["hyp_text"])
            stage10.log(args.out, f"{run_dir.name} wer {record['wer']:.2f} audio {record['audio_duration_s']:.1f} s "
                                  f"hyp {record['hyp_text'][:120]!r}")
    finally:
        pair_check.stop(cards[0], asr)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    stage10.log(args.out, "done")


if __name__ == "__main__":
    main()
