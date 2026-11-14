"""Greedy Video-MME replies of chosen questions sent alone and sent together with other videos, on one server.

usage: python3 cobatch_probe.py --port P --out probe.json [--targets 005-3,017-1] [--partners 001-1,008-2,014-3]

Requests are the CI benchmark's own (make_video_send_fn at 2 fps, 128 frames, 401,408 pixels, 256 tokens, temperature 0).
Each target runs alone three times in a row, then three times at once with one partner, then twice with three partners.
Within one server, a reply that differs only when other requests share the batch is moved by batch composition.
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from benchmarks.dataset.videomme import load_videomme_samples
from benchmarks.tasks.video_understanding import make_video_send_fn


async def run(args: argparse.Namespace) -> list[dict]:
    samples = {
        sample.question_id: sample
        for sample in load_videomme_samples(repo_id="zhaochenyang20/Video_MME_ci")
    }
    send = make_video_send_fn(
        "qwen3-omni",
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        max_tokens=256,
        temperature=0.0,
        video_fps=2.0,
        video_max_frames=128,
        video_min_pixels=None,
        video_max_pixels=401408,
        video_total_pixels=None,
    )
    partners = args.partners.split(",")
    records = []
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for target in args.targets.split(","):
            plan = [("alone", [])] * 3 + [("with1", partners[:1])] * 3
            plan += [("with3", partners[:3])] * 2
            for mode, others in plan:
                results = await asyncio.gather(
                    send(session, samples[target]),
                    *(send(session, samples[other]) for other in others),
                )
                head = results[0]
                records.append(
                    {
                        "target": target,
                        "mode": mode,
                        "partners": others,
                        "ok": head.is_success,
                        "completion_tokens": head.completion_tokens,
                        "prompt_tokens": head.prompt_tokens,
                        "text": head.text if head.is_success else head.error,
                    }
                )
                print(
                    f"{target} {mode:5s} tok {head.completion_tokens:3d} "
                    f"{' '.join((head.text or head.error or '').split())[:90]}",
                    flush=True,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--targets", default="005-3,017-1")
    parser.add_argument("--partners", default="001-1,008-2,014-3")
    args = parser.parse_args()
    records = asyncio.run(run(args))
    Path(args.out).write_text(json.dumps(records, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
