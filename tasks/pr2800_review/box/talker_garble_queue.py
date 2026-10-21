"""run210: talker garbles and run-ons, HF's generate against omni on the same voice-clone prompts, all in the container.

Run in the container: python3 /workspace/box/tasks/pr2800_review/box/talker_garble_queue.py
1. talker_garble_probe.py --arm hf on card 0 beside --arm hf_zero_rows on card 1, 16 outputs a prompt.
2. main's server (upr, the served colocated config) on card 0: the probe's server arm at c1, then at c16.
3. Qwen3-ASR on card 0 transcribes every arm through benchmark_omni_seedtts --transcribe-only.
4. Per arm and prompt group (the 45 recorded prompts, the 40 controls): outputs, garbles (WER above 50 %),
   catastrophic outputs (above 90 %), run-ons (over 20 s of audio), mean WER; then the prompts that garble most.
Writes /data/pr2800/run210/<arm>/ and summary.json; logs to /data/pr2800/run210/queue.log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402

CPUS, NODE = card_pins()
os.environ["PR2800_CARD_CPUS"], os.environ["PR2800_NODE"] = CPUS, NODE
from pair_check import ASR_MODEL, ASR_PORT, CONFIG, MODEL, benchmark, serve, stop, tree  # noqa: E402

OUT = Path("/data/pr2800/run210")
SAMPLES = 16
SERVER_PORT = 24000
PROBE = BOX / "talker_garble_probe.py"
RECORDED = set(json.loads((BOX / "talker_garble_prompts.json").read_text())["ids"])


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(OUT / "queue.log", "a") as handle:
        handle.write(line + "\n")


def probe(arm: str, card: int, main_tree: Path, extra: list[str]) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONPATH": str(main_tree)}
    return subprocess.Popen(
        [sys.executable, str(PROBE), "--arm", arm, "--samples", str(SAMPLES), "--out", str(OUT / arm.split("@")[0]),
         *extra],
        cwd=main_tree, env=env, stdout=open(OUT / f"{arm}.log", "w"), stderr=subprocess.STDOUT,
    )


def read(arm_dir: Path) -> dict[str, object]:
    samples = json.loads((arm_dir / "wer_results.json").read_text())["per_sample"]
    groups: dict[str, dict[str, float]] = {}
    garbles: Counter = Counter()
    for group in ("recorded", "controls"):
        rows = [s for s in samples if (s["id"].rsplit("__", 1)[0] in RECORDED) == (group == "recorded")]
        wers = [s["wer"] for s in rows if s.get("wer") is not None]
        groups[group] = dict(
            outputs=len(rows),
            garbles=sum(w > 0.5 for w in wers),
            catastrophic=sum(w > 0.9 for w in wers),
            runons=sum((s.get("audio_duration_s") or 0) > 20 for s in rows),
            mean_wer=round(100 * sum(wers) / max(len(wers), 1), 3),
        )
    for s in samples:
        if (s.get("wer") or 0) > 0.5:
            garbles[s["id"].rsplit("__", 1)[0]] += 1
        else:
            pass
    return dict(groups=groups, most=garbles.most_common(12))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_tree = tree("upr")
    log(f"pins {CPUS} node {NODE}; main tree {main_tree.name}")
    hf = [probe("hf", 0, main_tree, ["--device", "cuda:0"]), probe("hf_zero_rows", 1, main_tree, ["--device", "cuda:0"])]
    for arm, process in zip(("hf", "hf_zero_rows"), hf):
        log(f"{arm} rc {process.wait()}")
    server = serve(0, main_tree, MODEL, SERVER_PORT, ["--config", CONFIG, "--colocate"], OUT / "server.log")
    try:
        for concurrency in (1, 16):
            arm = f"server_c{concurrency}"
            env = {**os.environ, "PYTHONPATH": str(main_tree)}
            rc = subprocess.run(
                [sys.executable, str(PROBE), "--arm", "server", "--port", str(SERVER_PORT), "--concurrency",
                 str(concurrency), "--samples", str(SAMPLES), "--out", str(OUT / arm)],
                cwd=main_tree, env=env, stdout=open(OUT / f"{arm}.log", "w"), stderr=subprocess.STDOUT,
            ).returncode
            log(f"{arm} rc {rc}")
    finally:
        stop(0, server)
    arms = ["hf", "hf_zero_rows", "server_c1", "server_c16"]
    asr = serve(0, main_tree, ASR_MODEL, ASR_PORT, [], OUT / "asr_server.log")
    try:
        for arm in arms:
            rc = benchmark(0, ["--transcribe-only", "--port", str(ASR_PORT), "--host", "127.0.0.1", "--lang", "en",
                               "--output-dir", str(OUT / arm)], OUT / f"{arm}_wer.log")
            log(f"wer {arm} rc {rc}")
    finally:
        stop(0, asr)
    summary = {arm: read(OUT / arm) for arm in arms if (OUT / arm / "wer_results.json").exists()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for arm, result in summary.items():
        for group, row in result["groups"].items():
            log(f"{arm:13s} {group:9s} outputs {row['outputs']:5d} garbles {row['garbles']:4d} catastrophic "
                f"{row['catastrophic']:4d} run-ons {row['runons']:3d} mean WER {row['mean_wer']:.3f} %")
        log(f"{arm:13s} most garbled prompts {result['most']}")
    log("done")


if __name__ == "__main__":
    main()
