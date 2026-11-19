"""The CI's Video-MME talker test (tests/test_model/test_qwen3_omni_videomme_talker_ci.py) against a running server.

The test's own samples, prompt suffix, VideoEvalConfig and floors, run --passes times; each pass writes its results to
<out>/pass<i>/ and one line to <out>/summary.jsonl: accuracy against the calibrated floor, and the speed metrics the
test gates at c16 against their thresholds. The test's WER step (an ASR server after the omni server stops) is not run.

usage: PYTHONPATH=<tree> python3 videomme_talker_ci.py --port 28400 --out /data/run/talker --passes 3
"""

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from tests.test_model import test_qwen3_omni_videomme_talker_ci as talker_test
from tests.test_model.omni_ci_config import (
    QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY,
    QWEN3_OMNI_VIDEOMME_TALKER_P95,
)


def run_pass(port: int, output_dir: Path) -> dict[str, object]:
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=port,
        host="127.0.0.1",
        max_samples=talker_test.MAX_SAMPLES,
        max_tokens=talker_test.MAX_TOKENS,
        max_concurrency=talker_test.CONCURRENCY,
        output_dir=str(output_dir),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device=talker_test.ASR_DEVICE,
        asr_concurrency=talker_test.QWEN3_ASR_WER_CONCURRENCY,
        disable_tqdm=True,
        timeout_s=500,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            samples=talker_test.load_short_answer_samples(),
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default="results/videomme_audio",
            compute_wer=False,
        )
    )
    summary = results["summary"]
    speed = results["speed"]
    gated = QWEN3_OMNI_VIDEOMME_TALKER_P95[talker_test.CONCURRENCY]
    return {
        "accuracy": summary.get("accuracy"),
        "correct": summary.get("correct"),
        "total": summary.get("total_samples"),
        "failed": summary.get("failed", 0),
        "accuracy_floor": QWEN3_OMNI_VIDEOMME_TALKER_MIN_ACCURACY,
        "speed": {key: speed.get(key) for key in gated},
        "speed_p95_reference": gated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index in range(args.passes):
        row = {"pass": index, **run_pass(args.port, out / f"pass{index}")}
        with open(out / "summary.jsonl", "a", encoding="utf-8") as sink:
            sink.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
