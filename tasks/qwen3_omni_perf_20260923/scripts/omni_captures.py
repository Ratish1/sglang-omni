"""Degenerate prefill and decode torch captures of a running Qwen3-Omni server.

One stage process is counted per capture (--stage thinker or talker_ar); code2wav has no
forward counter and is captured by the live window instead.

prefill: every request is distinct (no radix or encoder cache hit) and stops after one
token of the counted stage, so every counted forward is an extend. Text prompts are sized
with the checkpoint tokenizer to --prompt-tokens; --media-list gives one media file per
request (image, audio or video). --warmup requests at the same batch run first, then the
window is armed and --steps forwards are counted.
  thinker:   modalities text, max_tokens 1
  talker_ar: modalities text+audio, thinker max_tokens --max-tokens, talker_max_new_tokens 1
decode: a short prompt that answers long; the window is armed once every request has
streamed --warmup text chunks (thinker) or --warmup audio chunks (talker_ar), so the
counted forwards are decode at the workload's batch size. The batch of every counted
forward is in the trace span names; the ledger reads it from there.

Every run records per request: prompt and completion tokens, first text, first audio,
last chunk, audio chunks. For thinker decode, the client cadence (ms per text chunk while
every request streams) is the unprofiled step time when --no-capture is set.
Output: OUT/<label>/<stage>_<kind>_<modality>/b<batch>/ with the trace and workload.json.

usage: python omni_captures.py --url http://127.0.0.1:8000 --model MODEL --stage thinker
         --kind prefill --modality text --batch 1 --steps 5 --label formal --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import aiohttp

TRACE_WAIT_S = 900.0
DECODE_PROMPT = (
    "Write a very long, detailed story about a lighthouse keeper and the ships "
    "that pass by over one year. Do not stop early."
)
FILLER = (
    "The committee reviewed the quarterly logistics report, compared shipping "
    "volumes across every regional warehouse, and noted which routes were delayed. "
)


def sized_text(tokenizer, target: int) -> str:
    ids = tokenizer(FILLER * (target // 20 + 8), add_special_tokens=False).input_ids
    return tokenizer.decode(ids[:target])


def prefill_prompts(args: argparse.Namespace, count: int) -> list[dict]:
    """One distinct request body per prefill request."""
    if args.modality == "text":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        body = sized_text(tokenizer, args.prompt_tokens)
        return [
            {"messages": [{"role": "user", "content": f"{uuid.uuid4().hex}. {body}"}]}
            for _ in range(count)
        ]
    media = [
        line.strip()
        for line in Path(args.media_list).read_text().splitlines()
        if line.strip()
    ]
    if len(media) < count:
        raise ValueError(f"{len(media)} media files, {count} distinct ones needed")
    key = {"image": "images", "audio": "audios", "video": "videos"}[args.modality]
    bodies = []
    for path in media[:count]:
        body = {
            "messages": [{"role": "user", "content": "Describe this in one word."}],
            key: [path],
        }
        if args.modality == "video":
            body.update(
                video_fps=2, video_max_frames=128, video_max_pixels=401408
            )
        bodies.append(body)
    return bodies


def request_body(args: argparse.Namespace, base: dict) -> dict:
    body = {"model": args.model, "stream": True, "temperature": 0.0, **base}
    speech = args.stage == "talker_ar" or args.speech
    body["modalities"] = ["text", "audio"] if speech else ["text"]
    if speech:
        body["audio"] = {"format": "wav"}
    if args.kind == "prefill":
        body["max_tokens"] = args.max_tokens if speech else 1
        if speech:
            body["talker_max_new_tokens"] = 1
    else:
        body["max_tokens"] = args.max_tokens
    return body


async def stream_one(
    session: aiohttp.ClientSession,
    url: str,
    body: dict,
    warmup_chunks: int,
    armed_ready: asyncio.Event | None,
    count_audio: bool,
) -> dict:
    """Stream one chat request to the end; returns its timeline."""
    record: dict = {"text_times": [], "audio_times": [], "usage": None}
    began = time.perf_counter()
    async with session.post(f"{url}/v1/chat/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[len("data: ") :])
            now = time.perf_counter()
            if data.get("usage"):
                record["usage"] = data["usage"]
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    record["text_times"].append(now)
                if delta.get("audio"):
                    record["audio_times"].append(now)
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
            seen = record["audio_times"] if count_audio else record["text_times"]
            if (
                armed_ready is not None
                and not armed_ready.is_set()
                and len(seen) >= warmup_chunks
            ):
                armed_ready.set()
    if armed_ready is not None and not armed_ready.is_set():
        armed_ready.set()
        record["ended_before_warmup"] = True
    record["began"] = began
    record["ended"] = time.perf_counter()
    return record


def text_cadence_ms(records: list[dict]) -> float | None:
    """Median ms per text chunk across requests while every request is streaming."""
    begin = max(r["text_times"][0] for r in records if r["text_times"])
    end = min(r["text_times"][-1] for r in records if r["text_times"])
    per_request = []
    for r in records:
        inside = [t for t in r["text_times"] if begin <= t <= end]
        if len(inside) >= 2:
            per_request.append(1e3 * (inside[-1] - inside[0]) / (len(inside) - 1))
    return statistics.median(per_request) if per_request else None


async def post(
    session: aiohttp.ClientSession, url: str, route: str, body: dict
) -> dict:
    async with session.post(f"{url}{route}", json=body) as response:
        if response.status != 200:
            raise RuntimeError(
                f"{route} HTTP {response.status}: {await response.text()}"
            )
        return await response.json()


async def run(args: argparse.Namespace) -> None:
    count_audio = args.stage == "talker_ar"
    output = "speech" if args.stage == "talker_ar" or args.speech else "text"
    name = f"{args.stage}_{args.kind}_{args.modality}_{output}"
    trace_dir = Path(args.out).resolve() / args.label / name / f"b{args.batch}"
    trace_dir.mkdir(parents=True, exist_ok=False)
    arm = {
        "run_id": f"{args.label}-{name}-b{args.batch}-{int(time.time())}",
        "trace_path_template": str(trace_dir / "trace"),
        "num_steps": args.steps,
        "step_stage": args.stage,
        "with_stack": args.with_stack,
        "record_shapes": args.record_shapes,
    }
    workload: dict = {
        "stage": args.stage,
        "kind": args.kind,
        "modality": args.modality,
        "output": output,
        "batch": args.batch,
        "steps": args.steps,
        "warmup": args.warmup,
        "prompt_tokens_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "capture": not args.no_capture,
        "with_stack": args.with_stack,
    }
    records: list[dict] = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        if args.kind == "prefill":
            rounds = args.steps if args.batch == 1 else 1
            bodies = prefill_prompts(args, args.warmup + args.batch * rounds)
            warmup, captured = bodies[: args.warmup], bodies[args.warmup :]
            for start in range(0, len(warmup), args.batch):
                await asyncio.gather(
                    *(
                        stream_one(
                            session, args.url, request_body(args, b), 0, None, count_audio
                        )
                        for b in warmup[start : start + args.batch]
                    )
                )
            if not args.no_capture:
                workload["start_profile"] = await post(
                    session, args.url, "/start_profile", arm
                )
            for start in range(0, len(captured), args.batch):
                records.extend(
                    await asyncio.gather(
                        *(
                            stream_one(
                                session, args.url, request_body(args, b), 0, None, count_audio
                            )
                            for b in captured[start : start + args.batch]
                        )
                    )
                )
        else:
            base = {"messages": [{"role": "user", "content": DECODE_PROMPT}]}
            ready = [asyncio.Event() for _ in range(args.batch)]
            tasks = [
                asyncio.create_task(
                    stream_one(
                        session,
                        args.url,
                        request_body(args, base),
                        args.warmup,
                        event,
                        count_audio,
                    )
                )
                for event in ready
            ]
            await asyncio.gather(*(event.wait() for event in ready))
            workload["armed_at"] = time.perf_counter()
            if not args.no_capture:
                workload["start_profile"] = await post(
                    session, args.url, "/start_profile", arm
                )
            records = list(await asyncio.gather(*tasks))
            if not count_audio:
                workload["client_ms_per_text_chunk"] = text_cadence_ms(records)
        if not args.no_capture:
            # a window with no forward after its last one never closes itself;
            # the stop exports it, and is a no-op when it already closed
            await post(session, args.url, "/stop_profile", {"run_id": arm["run_id"]})
    workload["requests"] = [
        {
            "usage": r["usage"],
            "finish_reason": r.get("finish_reason"),
            "ended_before_warmup": r.get("ended_before_warmup", False),
            "text_chunks": len(r["text_times"]),
            "audio_chunks": len(r["audio_times"]),
            "first_text_s": r["text_times"][0] - r["began"] if r["text_times"] else None,
            "first_audio_s": (
                r["audio_times"][0] - r["began"] if r["audio_times"] else None
            ),
            "total_s": r["ended"] - r["began"],
            "last_chunk_after_arm_s": (
                max(r["text_times"] + r["audio_times"]) - workload["armed_at"]
                if "armed_at" in workload and (r["text_times"] or r["audio_times"])
                else None
            ),
        }
        for r in records
    ]
    if not args.no_capture:
        deadline = time.monotonic() + TRACE_WAIT_S
        while True:
            traces = sorted(trace_dir.glob("trace*.trace.json.gz"))
            if traces and not list(trace_dir.glob("trace*.trace.json")):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"no finished trace under {trace_dir}")
            await asyncio.sleep(1.0)
        workload["traces"] = [str(path) for path in traces]
    (trace_dir / "workload.json").write_text(json.dumps(workload, indent=1))
    print(
        json.dumps(
            {
                "dir": str(trace_dir),
                "traces": workload.get("traces"),
                "client_ms_per_text_chunk": workload.get("client_ms_per_text_chunk"),
                "prompt_tokens": [
                    (r["usage"] or {}).get("prompt_tokens")
                    for r in workload["requests"]
                ],
                "completion_tokens": [
                    (r["usage"] or {}).get("completion_tokens")
                    for r in workload["requests"]
                ],
                "ended_before_warmup": sum(
                    r["ended_before_warmup"] for r in workload["requests"]
                ),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", choices=("thinker", "talker_ar"), required=True)
    parser.add_argument("--kind", choices=("prefill", "decode"), required=True)
    parser.add_argument(
        "--modality", choices=("text", "image", "audio", "video"), default="text"
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5, help="forwards to capture")
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="prefill: warmup requests; decode: chunks every request streams before arming",
    )
    parser.add_argument("--label", required=True, help="formal, mapping, uncaptured")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8000)
    parser.add_argument("--media-list", help="one media path per line, prefill only")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="decode: thinker max_tokens; talker prefill: thinker max_tokens",
    )
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument(
        "--speech", action="store_true", help="thinker requests also ask for audio"
    )
    args = parser.parse_args()
    if args.kind == "prefill" and args.modality != "text" and not args.media_list:
        parser.error("--media-list is required for media prefill")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
