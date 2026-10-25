"""Compare the CI Talker prompt on the available single-H200 FP8 profile."""

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MMMU_TTS_PROMPT


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["base", "head"], required=True)
    arguments = parser.parse_args()
    directory = Path("/data/review3175oct25") / f"{arguments.arm}_talker"
    directory.mkdir(exist_ok=True)
    for repetition in range(2):
        output_directory = directory / f"round{repetition}"
        output_directory.mkdir(exist_ok=True)
        configuration = MMMUEvalConfig(
            model="qwen3-omni",
            port=13775,
            max_samples=20,
            max_tokens=256,
            max_concurrency=16,
            warmup=16,
            output_dir=str(output_directory),
            enable_audio=True,
            repo_id="zhaochenyang20/mmmu-ci-50",
            prompt_override=MMMU_TTS_PROMPT,
            timeout_s=500,
        )
        results = await run_mmmu_eval(configuration, compute_wer=False)
        (output_directory / "results.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    "arm": arguments.arm,
                    "round": repetition,
                    "summary": results["summary"],
                    "limit_reaching": sum(
                        record["completion_tokens"] >= 256
                        for record in results["per_sample"]
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
