# SPDX-License-Identifier: Apache-2.0
"""Answer-token logprobs for MMSU questions through the omni /generate endpoint.

Usage (from the omni checkout root, PYTHONPATH=$PWD):
    python logprob_probe.py --port P --out probe.json [--sample-ids ID,ID,...] [--concurrency 16]

Sends every selected mmsu-ci-2000 question (text only, temperature 0, the
benchmark's prompt text) to /generate with return_logprob, which returns
[logprob, token_id] per generated token (sglang_omni/model_runner/base.py:888,
serve/openai_api.py:1178). For each question the script records the answer
letter, its token logprob p and the margin lower bound log(p / (1 - p)) over
any runner-up token, since /generate reports the sampled token only.

Without --sample-ids all 2000 questions are probed, which gives the margin
distribution of a whole stack. The chat endpoint the benchmarks use has no
logprob field, and /generate carries no video or audio inputs, so this probe
covers stage 5 only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys

import aiohttp


async def _one(session, url, sample, question_text, sem):
    payload = {
        "messages": [{"role": "user", "content": question_text}],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 32},
        "return_logprob": True,
    }
    async with sem:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
    text = body.get("text", "") or ""
    lps = (body.get("meta_info") or {}).get("output_token_logprobs") or []
    first = lps[0] if lps else None
    p = math.exp(first[0]) if first else None
    margin = math.log(p / (1.0 - p)) if p is not None and 0.0 < p < 1.0 else None
    return {
        "sample_id": sample.sample_id,
        "answer_index": sample.answer_index,
        "text": text,
        "completion_tokens": len(lps),
        "first_token_logprob": first[0] if first else None,
        "first_token_id": first[1] if first else None,
        "first_token_p": p,
        "margin_lower_bound": margin,
        "output_token_logprobs": lps,
    }


async def probe(args) -> list[dict]:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.tasks.audio_understanding import (
        DEFAULT_PROMPT,
        _build_question_text,
    )

    samples = load_mmsu_samples(repo_id=DATASETS["mmsu-ci-2000"])
    if args.sample_ids:
        wanted = set(args.sample_ids.split(","))
        samples = [s for s in samples if s.sample_id in wanted]
        missing = wanted - {s.sample_id for s in samples}
        if missing:
            print(f"unknown sample ids: {sorted(missing)}", file=sys.stderr)
    url = f"http://{args.host}:{args.port}/generate"
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *[
                _one(session, url, s, _build_question_text(s, DEFAULT_PROMPT), sem)
                for s in samples
            ]
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-ids", default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout-s", type=int, default=300)
    args = p.parse_args(argv)
    rows = asyncio.run(probe(args))
    json.dump(rows, open(args.out, "w"), indent=1)
    margins = [
        r["margin_lower_bound"] for r in rows if r["margin_lower_bound"] is not None
    ]
    near = sum(1 for m in margins if m < 0.1)
    print(
        f"{len(rows)} questions, {len(margins)} with a first-token probability, {near} with margin bound < 0.1 nat"
    )
    for r in rows[: args.concurrency]:
        print(
            f"{r['sample_id']} text={r['text']!r} p={r['first_token_p']} margin>={r['margin_lower_bound']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
