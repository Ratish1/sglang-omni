"""The CI talker test's own request for one question, alone and sent together with other videos, on a server running
thinker_numerics_probe; then, per run, where its logits first differ from the first alone run and where its token
does.

Each run records its time window; the probe's lines for the target's prompt length are grouped into runs by time.
Per run: the reply, the first generated step whose logits fingerprint differs from alone run 0 (with that step's
forward mode, rows and batch), and the first step whose top token differs, with both runs' top-2 logits there.

usage: PYTHONPATH=<tree> python3 thinker_cobatch_numerics.py --port P --probe-prefix /data/run/num --out result.json \
           [--target 005-3] [--partners 001-1,008-2,014-3,020-1,010-2,017-1,019-2]
"""

import argparse
import asyncio
import glob
import json
import time

import aiohttp

from benchmarks.tasks.video_understanding import make_video_send_fn
from tests.test_model import test_qwen3_omni_videomme_talker_ci as talker_test


async def run(args: argparse.Namespace) -> list[dict]:
    samples = {sample.question_id: sample for sample in talker_test.load_short_answer_samples()}
    from benchmarks.dataset.videomme import load_videomme_samples

    for sample in load_videomme_samples(repo_id="zhaochenyang20/Video_MME_ci"):
        samples.setdefault(sample.question_id, sample)
    send = make_video_send_fn(
        "qwen3-omni",
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        max_tokens=talker_test.MAX_TOKENS,
        temperature=0.0,
        video_fps=2.0,
        video_max_frames=128,
        video_min_pixels=None,
        video_max_pixels=401408,
        video_total_pixels=None,
    )
    partners = args.partners.split(",")
    scenarios = [("alone", [])] * 3
    for count in (1, 3, 7):
        scenarios += [(f"with {count}", partners[:count])] * 2
    runs = []
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for label, others in scenarios:
            started = time.time()
            results = await asyncio.gather(
                send(session, samples[args.target]), *(send(session, samples[name]) for name in others)
            )
            target = results[0]
            runs.append(
                {
                    "label": label,
                    "partners": others,
                    "start": started,
                    "end": time.time(),
                    "reply": target.text if target.is_success else target.error,
                    "prompt_tokens": target.prompt_tokens,
                }
            )
            print(f"{label}: {runs[-1]['reply'][-80:]!r}", flush=True)
            await asyncio.sleep(1.0)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--probe-prefix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="005-3")
    parser.add_argument("--partners", default="001-1,008-2,014-3,020-1,010-2,017-1,019-2")
    parser.add_argument("--target-len", type=int, default=0, help="the target's prompt token count (default: its reply's)")
    args = parser.parse_args()
    runs = asyncio.run(run(args))
    time.sleep(2.0)
    lines = []
    for path in glob.glob(f"{args.probe_prefix}.*.jsonl"):
        with open(path) as source:
            lines += [json.loads(line) for line in source if line.strip()]
    target_len = runs[0]["prompt_tokens"] if args.target_len == 0 else args.target_len
    lines = [line for line in lines if line["prompt_len"] == target_len]
    import torch

    prefix_dir = args.probe_prefix
    for entry in runs:
        steps = {}
        for line in sorted((line for line in lines if entry["start"] <= line["time"] <= entry["end"] + 1.0 and line.get("sampled")), key=lambda line: line["time"]):
            steps.setdefault(line["position"], line)
        entry["steps"] = [steps[key] for key in sorted(steps)]

    def logits_of(line):
        paths = glob.glob(f"{prefix_dir}.*.logits.{line['rid']}.{line['position']}.pt")
        return torch.load(paths[0]) if paths else None

    reference = runs[0]["steps"]
    for entry in runs:
        pairs = list(zip(reference, entry["steps"]))
        first_logits = next((index for index, (a, b) in enumerate(pairs) if a["fingerprint"] != b["fingerprint"]), None)
        first_token = next((index for index, (a, b) in enumerate(pairs) if a["top_ids"][0] != b["top_ids"][0]), None)
        entry["first_logits_index"] = first_logits
        entry["first_token_index"] = first_token
        detail = ""
        if first_logits is not None:
            ref, mine = pairs[first_logits]
            a, b = logits_of(ref), logits_of(mine)
            gap = float((a - b).abs().max()) if a is not None and b is not None else None
            detail += (
                f" logits first differ at position {mine['position']} (output token {first_logits}): this run mode {mine['mode']}, rows {mine['forward_rows']}, reqs {mine['batch_reqs']};"
                f" alone mode {ref['mode']}, rows {ref['forward_rows']}; max abs logit difference {gap}"
            )
            entry["first_logits_max_abs"] = gap
        else:
            pass
        if first_token is not None:
            ref, mine = pairs[first_token]
            a, b = logits_of(ref), logits_of(mine)
            gap = float((a - b).abs().max()) if a is not None and b is not None else None
            detail += (
                f"; token splits at position {mine['position']} (output token {first_token}): here {mine['top_ids']} {[round(v, 4) for v in mine['top_logits']]},"
                f" alone {ref['top_ids']} {[round(v, 4) for v in ref['top_logits']]} (alone margin {round(ref['top_logits'][0] - ref['top_logits'][1], 4)}), max abs logit difference there {gap}"
            )
        else:
            pass
        print(f"{entry['label']} {entry['partners']}: {len(entry['steps'])} sampled positions;{detail}", flush=True)
    with open(args.out, "w") as sink:
        json.dump(runs, sink, indent=1)


if __name__ == "__main__":
    main()
