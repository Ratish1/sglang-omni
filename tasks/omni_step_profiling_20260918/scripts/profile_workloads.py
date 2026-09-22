"""Degenerate prefill and decode torch captures of a running Qwen3-TTS server.

prefill: requests capped at one codec frame, so every counted forward is one extend.
The window is armed before the requests go out; batch 1 sends --steps requests one
after another, a larger batch sends one burst.
decode: long texts; the window is armed after every request has streamed its first
chunk, so the counted forwards are decode at the workload's batch size.

Every run records chunk arrival times. For decode, the client cadence (ms per codec
frame while all requests stream) is the unprofiled step time when --no-capture is set.
Output: OUT/<label>/<prefill|decode>/b<batch>/ with the trace and workload.json; the
prefill/decode directory is the stage label the llm-torch-profiler-analysis triage reads.

usage: python profile_workloads.py --url http://127.0.0.1:8000 --model MODEL
         --kind decode --batch 16 --steps 40 --label formal --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
import soundfile

from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.tasks.tts import _validate_raw_pcm_response_headers

TRACE_WAIT_S = 600.0


def pick_samples(
    samples: list[SampleInput], kind: str, count: int
) -> list[SampleInput]:
    """Longest reference clips for prefill, longest texts for decode, one per reference."""
    if kind == "prefill":
        ranked = sorted(
            samples,
            key=lambda s: (soundfile.info(s.ref_audio).duration, len(s.target_text)),
            reverse=True,
        )
    else:
        ranked = sorted(samples, key=lambda s: len(s.target_text), reverse=True)
    picked: list[SampleInput] = []
    seen_refs: set[str] = set()
    for sample in ranked:
        if sample.ref_audio in seen_refs:
            continue
        seen_refs.add(sample.ref_audio)
        picked.append(sample)
        if len(picked) == count:
            return picked
    raise ValueError(f"only {len(picked)} distinct references, {count} needed")


def payload(model: str, sample: SampleInput, max_new_tokens: int) -> dict:
    return {
        "model": model,
        "input": sample.target_text,
        "ref_audio": sample.ref_audio,
        "ref_text": sample.ref_text,
        "response_format": "pcm",
        "stream": True,
        "max_new_tokens": max_new_tokens,
    }


async def speak(
    session: aiohttp.ClientSession,
    url: str,
    body: dict,
    frame_rate: float,
    first_chunk: asyncio.Event | None = None,
) -> list[tuple[float, float]]:
    """Stream one request to the end; returns (arrival time, codec frames so far) per chunk."""
    arrivals: list[tuple[float, float]] = []
    received = 0
    async with session.post(f"{url}/v1/audio/speech", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        sample_rate, channels, width = _validate_raw_pcm_response_headers(
            response.headers
        )
        bytes_per_frame = sample_rate / frame_rate * channels * width
        async for chunk in response.content.iter_any():
            received += len(chunk)
            arrivals.append((time.perf_counter(), received / bytes_per_frame))
            if first_chunk is not None and not first_chunk.is_set():
                first_chunk.set()
    return arrivals


def decode_cadence_ms(streams: list[list[tuple[float, float]]]) -> float | None:
    """Median ms per codec frame across requests while every request is streaming."""
    begin = max(stream[0][0] for stream in streams)
    end = min(stream[-1][0] for stream in streams)
    per_request = []
    for stream in streams:
        inside = [point for point in stream if begin <= point[0] <= end]
        if len(inside) >= 2 and inside[-1][1] > inside[0][1]:
            per_request.append(
                1e3 * (inside[-1][0] - inside[0][0]) / (inside[-1][1] - inside[0][1])
            )
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
    samples = load_seedtts_samples(args.meta, split=args.lang)
    rounds = args.steps if args.kind == "prefill" and args.batch == 1 else 1
    captured_count = args.batch * rounds
    picked = pick_samples(samples, args.kind, args.batch * args.warmup + captured_count)
    warmup, captured = (
        picked[: args.batch * args.warmup],
        picked[args.batch * args.warmup :],
    )
    max_new_tokens = 1 if args.kind == "prefill" else args.decode_max_new_tokens
    trace_dir = Path(args.out).resolve() / args.label / args.kind / f"b{args.batch}"
    trace_dir.mkdir(parents=True, exist_ok=False)
    arm = {
        "run_id": f"{args.label}-{args.kind}-b{args.batch}-{int(time.time())}",
        "trace_path_template": str(trace_dir / "trace"),
        "num_steps": args.steps,
        "step_stage": args.step_stage,
        "with_stack": args.with_stack,
        "record_shapes": args.record_shapes,
    }
    record: dict = {
        "kind": args.kind,
        "batch": args.batch,
        "steps": args.steps,
        "warmup_rounds": args.warmup,
        "max_new_tokens": max_new_tokens,
        "capture": not args.no_capture,
        "with_stack": args.with_stack,
        "captured": [],
    }
    streams: list[list[tuple[float, float]]] = []

    def speak_one(sample: SampleInput, event: asyncio.Event | None = None):
        return speak(
            session,
            args.url,
            payload(args.model, sample, max_new_tokens),
            args.frame_rate,
            event,
        )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        for start in range(0, len(warmup), args.batch):
            await asyncio.gather(
                *(speak_one(s) for s in warmup[start : start + args.batch])
            )
        if args.kind == "prefill":
            if not args.no_capture:
                record["start_profile"] = await post(
                    session, args.url, "/start_profile", arm
                )
            for start in range(0, len(captured), args.batch):
                group = captured[start : start + args.batch]
                streams.extend(await asyncio.gather(*(speak_one(s) for s in group)))
        else:
            firsts = [asyncio.Event() for _ in captured]
            tasks = [
                asyncio.create_task(speak_one(s, event))
                for s, event in zip(captured, firsts)
            ]
            await asyncio.gather(*(event.wait() for event in firsts))
            if not args.no_capture:
                record["start_profile"] = await post(
                    session, args.url, "/start_profile", arm
                )
            streams = list(await asyncio.gather(*tasks))
            record["client_ms_per_frame"] = decode_cadence_ms(streams)
        if not args.no_capture:
            # a window with no forward after its last one never closes
            # itself; the stop exports it, and is a no-op when it already closed.
            await post(session, args.url, "/stop_profile", {"run_id": arm["run_id"]})
    for sample, stream in zip(captured, streams):
        record["captured"].append(
            {
                "sample_id": sample.sample_id,
                "text_chars": len(sample.target_text),
                "ref_seconds": soundfile.info(sample.ref_audio).duration,
                "frames": stream[-1][1],
                "first_chunk_s": stream[0][0],
                "last_chunk_s": stream[-1][0],
            }
        )
    if not args.no_capture:
        deadline = time.monotonic() + TRACE_WAIT_S
        while True:
            traces = sorted(trace_dir.glob("trace*.trace.json.gz"))
            if traces and not list(trace_dir.glob("trace*.trace.json")):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"no finished trace under {trace_dir}")
            await asyncio.sleep(1.0)
        record["traces"] = [str(path) for path in traces]
    (trace_dir / "workload.json").write_text(json.dumps(record, indent=1))
    print(
        json.dumps(
            {
                "dir": str(trace_dir),
                "traces": record.get("traces"),
                "client_ms_per_frame": record.get("client_ms_per_frame"),
                "min_frames": min(r["frames"] for r in record["captured"]),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True, help="forwards to capture")
    parser.add_argument(
        "--label", required=True, help="formal, mapping, uncaptured, ..."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--warmup", type=int, default=2, help="warmup rounds of --batch"
    )
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--step-stage", default="tts_engine")
    parser.add_argument("--decode-max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--frame-rate", type=float, default=12.5, help="codec frames per second"
    )
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
