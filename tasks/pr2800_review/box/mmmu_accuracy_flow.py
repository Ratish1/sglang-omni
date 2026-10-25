"""The component accuracy harness over the MMMU talker stage's data flow, per arm, against the HF reference in FP32.

Run on the box (all four cards):
    python3 /workspace/box/tasks/pr2800_review/box/mmmu_accuracy_flow.py --arms upy,vpg --out /data/<run>
    python3 mmmu_accuracy_flow.py --read /data/<run> --arms upy,vpg
The harness tests (pair_check tree "acc", fork branch test/qwen3-omni-component-accuracy) are copied into each arm's
tree, so they run against the arm's own sglang_omni. Phase 1 runs the component suite of every arm at once, one card
each: image inputs against HF's processor, the vision encoder on synthetic grids and on the stage's images alone and
packed, MRoPE positions, the audio encoder and preprocessing, the talker prefill and code2wav. Phase 2 serves each arm
in the CI's bf16 disaggregated shape on cards 0+1 and runs the image thinker check with the FP32 reference on card 2
and its BF16 floor on card 3; the HF recordings are computed for the first arm and reused for the rest through
<out>/thinker_reference.json. Reports land in <out>/<arm>/{suite,thinker}/ with the pytest logs beside them; --read
prints each arm's verdicts and numbers next to each other.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402
import videoamme_talker_tp2  # noqa: E402
import videomme_disagg  # noqa: E402

HARNESS = "acc"
HARNESS_DIR = "tests/test_model/qwen3_omni_component_accuracy"
SUITE = (
    "test_vision_preprocessing.py",
    "test_vision_encoder.py",
    "test_mrope_positions.py",
    "test_audio_encoder.py",
    "test_preprocessing.py",
    "test_talker_prefill.py",
    "test_code2wav.py",
)
THINKER = "test_thinker_images.py"
SERVER_CARDS = (0, 1)
REFERENCE_CARDS = (2, 3)
CARD_NODES = {0: 0, 1: 0, 2: 1, 3: 1}


def overlay(arm_tree: Path) -> None:
    shutil.copytree(
        pair_check.tree(HARNESS) / HARNESS_DIR, arm_tree / HARNESS_DIR, dirs_exist_ok=True
    )


def run_pytest(
    arm_tree: Path, files: list[str], cards: tuple[int, ...], report_dir: Path, extra_env: dict[str, str]
) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **pair_check.card_env(cards[0], arm_tree),
        "CUDA_VISIBLE_DEVICES": ",".join(str(card) for card in cards),
        "QWEN3_OMNI_ACCURACY_MODEL": videomme_disagg.MODEL,
        "QWEN3_OMNI_ACCURACY_REPORT_DIR": str(report_dir),
        **extra_env,
    }
    with open(report_dir / "pytest.log", "w") as log:
        return subprocess.run(
            [
                "numactl",
                "--physcpubind=" + ",".join(pair_check.CARD_CPUS[card] for card in cards),
                f"--membind={CARD_NODES[cards[0]]}",
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-s",
                "-p",
                "no:cacheprovider",
                *(f"{HARNESS_DIR}/{name}" for name in files),
            ],
            cwd=arm_tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def suite(out: Path, arm: str, card: int, files: list[str]) -> None:
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    rc = run_pytest(arm_tree, files, (card,), out / arm / "suite", {"QWEN3_OMNI_ACCURACY_DEVICE": "cuda:0"})
    videoamme_talker_tp2.log(out, f"{arm} suite card {card} rc {rc}")


def thinker(out: Path, arm: str) -> None:
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = videomme_disagg.PORT + 300
    videoamme_talker_tp2.log(out, f"{arm} serving on cards {SERVER_CARDS}")
    server = videoamme_talker_tp2.serve(
        arm_tree,
        out / arm / "server.log",
        SERVER_CARDS,
        port,
        None,
        videomme_disagg.MODEL,
        videomme_disagg.SERVE_ARGS,
    )
    try:
        rc = run_pytest(
            arm_tree,
            [THINKER],
            REFERENCE_CARDS,
            out / arm / "thinker",
            {
                "QWEN3_OMNI_ACCURACY_SERVER": f"http://127.0.0.1:{port}",
                "QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE": "cuda:0",
                "QWEN3_OMNI_ACCURACY_FLOOR_DEVICE": "cuda:1",
                "QWEN3_OMNI_ACCURACY_REFERENCE_CACHE": str(out / "thinker_reference.json"),
            },
        )
    finally:
        videoamme_talker_tp2.stop(server, SERVER_CARDS)
    videoamme_talker_tp2.log(out, f"{arm} thinker rc {rc}")


def run(out: Path, arms: list[str], files: list[str], with_thinker: bool) -> None:
    for arm in arms:
        overlay(pair_check.tree(pair_check.ARMS[arm][0]))
    threads = [threading.Thread(target=suite, args=(out, arm, card, files)) for card, arm in enumerate(arms)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if with_thinker:
        for arm in arms:
            thinker(out, arm)
    else:
        pass
    videoamme_talker_tp2.log(out, "done")


def outcome_lines(log_path: Path) -> list[str]:
    return [
        line.strip()
        for line in log_path.read_text(errors="replace").splitlines()
        if " PASSED" in line or " FAILED" in line or " SKIPPED" in line or " ERROR" in line
    ]


def read(root: Path, arms: list[str]) -> None:
    for arm in arms:
        print(f"== {arm}")
        for part in ("suite", "thinker"):
            log_path = root / arm / part / "pytest.log"
            if log_path.exists():
                print("\n".join(outcome_lines(log_path)))
            else:
                print(f"{part}: no log")
        encoder = root / arm / "suite" / "vision_encoder.json"
        if encoder.exists():
            records = json.loads(encoder.read_text())["records"]
            for record in records:
                if record["case"].startswith("mmmu"):
                    print(
                        f"  {record['case']}: omni {record['omni']['relative_l2']:.3e}, "
                        f"floor {record['floor']['relative_l2']:.3e}, ratio {record['floor_ratio']:.2f}"
                    )
                else:
                    pass
        else:
            pass
        report_path = root / arm / "thinker" / "thinker_images.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            print(f"  tau {report['tau']:.3f}")
            for name in ("omni", "floor", "truth"):
                print(f"  {name}: {json.dumps(report[name]['summary'])}")
            truth = {entry["sample_id"]: entry["predicted"] for entry in report["truth"]["per_request"]}
            for entry in report["omni"]["per_request"]:
                print(
                    f"  {entry['sample_id']}: omni {entry['predicted']} correct {entry['is_correct']}, "
                    f"truth {truth[entry['sample_id']]}, departures {len(entry['departure_margins'])} "
                    f"max margin {max(entry['departure_margins'], default=0.0):.3f}"
                )
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--read", type=Path)
    parser.add_argument("--suite", default=",".join(SUITE), help="the suite's test files to run")
    parser.add_argument("--no-thinker", action="store_true", help="run the suite only")
    args = parser.parse_args()
    if args.read is not None:
        read(args.read, args.arms.split(","))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        run(args.out, args.arms.split(","), args.suite.split(","), not args.no_thinker)


if __name__ == "__main__":
    main()
