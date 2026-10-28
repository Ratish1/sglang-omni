"""#2533's speaker_embedding on a running Qwen3-TTS Base server built from probe/pr2533-speaker-dump.

usage: python3 speaker_embedding_e2e.py exact --base-url URL --model PATH --dump-dir DIR --out OUT.jsonl
       python3 speaker_embedding_e2e.py contract --base-url URL --model PATH --embedding EMB.pt --reference WAV
           [--custom-voice-url URL --custom-voice-model PATH]

exact: every SeedTTS EN sample, one request at a time, seed 1234, non-streaming WAV: A clones its reference in
x-vector mode (the server encodes it and the probe dumps the embedding it computed), B sends that embedding as
speaker_embedding, C repeats A (the reference now cached); records whether B's bytes equal A's and each latency.
contract: each acceptance and rejection rule of the field, the non-finite values with a health request after each,
the batch and WebSocket routes, an uploaded voice with an embedding, and a CustomVoice server if given.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import httpx
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples

TIMEOUT_S = 300.0
SEED = 1234


def speech(client: httpx.Client, base_url: str, body: dict[str, object]) -> tuple[int, bytes, float]:
    start = time.perf_counter()
    response = client.post(
        f"{base_url}/v1/audio/speech", content=json.dumps(body), headers={"content-type": "application/json"}
    )
    return response.status_code, response.content, 1000 * (time.perf_counter() - start)


def run_exact(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir)
    embeddings: dict[str, list[float]] = {}
    if args.from_dumps:
        for path in sorted(dump_dir.glob("*.pt")):
            dump = torch.load(path)
            if dump["ref_audio"].startswith("/"):
                embeddings[Path(dump["ref_audio"]).name] = dump["embedding"].tolist()
            else:
                pass
        print(f"{len(embeddings)} embeddings from earlier dumps", flush=True)
    else:
        pass
    seen_dumps = len(list(dump_dir.glob("*.pt")))
    records = []
    with httpx.Client(timeout=TIMEOUT_S, trust_env=False) as client:
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en"):
            clip_name = Path(sample.ref_audio).name
            base = {"model": args.model, "input": sample.target_text, "response_format": "wav", "seed": SEED}
            reference = {**base, "ref_audio": sample.ref_audio, "x_vector_only_mode": True}
            status_a, audio_a, latency_a = speech(client, args.base_url, reference)
            if not args.from_dumps:
                dumps = sorted(dump_dir.glob("*.pt"))
                for path in dumps[seen_dumps:]:
                    embeddings[clip_name] = torch.load(path)["embedding"].tolist()
                seen_dumps = len(dumps)
            else:
                pass
            record = {"id": sample.sample_id, "status_a": status_a, "latency_a_ms": latency_a}
            if clip_name in embeddings:
                embedded = {**base, "references": [{"speaker_embedding": embeddings[clip_name]}]}
                status_b, audio_b, latency_b = speech(client, args.base_url, embedded)
                status_c, audio_c, latency_c = speech(client, args.base_url, reference)
                record.update(
                    status_b=status_b, latency_b_ms=latency_b, status_c=status_c, latency_c_ms=latency_c,
                    identical_b=audio_a == audio_b, identical_c=audio_a == audio_c, bytes_a=len(audio_a),
                )
            else:
                record["error"] = "no embedding dumped"
            records.append(record)
            if len(records) % 100 == 0:
                print(f"{len(records)} done", flush=True)
            else:
                pass
    Path(args.out).write_text("".join(json.dumps(record) + "\n" for record in records))
    compared = [record for record in records if "identical_b" in record]
    print(
        f"samples {len(records)}, compared {len(compared)}, B identical to A {sum(r['identical_b'] for r in compared)}, "
        f"C identical to A {sum(r['identical_c'] for r in compared)}, statuses "
        f"{sorted({(r['status_a'], r.get('status_b'), r.get('status_c')) for r in records})}",
        flush=True,
    )
    for key in ("latency_a_ms", "latency_b_ms", "latency_c_ms"):
        values = sorted(r[key] for r in compared)
        print(f"{key} p50 {values[len(values) // 2]:.1f} p90 {values[int(0.9 * len(values))]:.1f}", flush=True)


async def websocket_case(base_url: str, session: dict[str, object]) -> str:
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(base_url.replace("http", "ws") + "/v1/audio/speech/stream") as socket:
            await socket.send_json({"type": "session.config", "session": session})
            configured = await socket.receive_json(timeout=60)
            if configured.get("type") != "session.configured":
                return f"config answered {configured}"
            else:
                pass
            await socket.send_json({"type": "input.text", "text": "Good morning, how can I help you today?"})
            await socket.send_json({"type": "input.done"})
            audio_bytes = 0
            events = []
            while True:
                message = await socket.receive(timeout=120)
                if message.type == aiohttp.WSMsgType.BINARY:
                    audio_bytes += len(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    events.append(event.get("type"))
                    if event.get("type") in ("session.done", "error"):
                        break
                    else:
                        pass
                else:
                    events.append(str(message.type))
                    break
            return f"events {events}, audio bytes {audio_bytes}"


def run_contract(args: argparse.Namespace) -> None:
    embedding = torch.load(args.embedding)["embedding"].tolist()
    base = {"model": args.model, "input": "Good morning, how can I help you today?", "seed": SEED}
    def given(values: object, **reference: object) -> list[dict[str, object]]:
        return [{"speaker_embedding": values, **reference}]

    cases = [
        ("valid, wav", {"references": given(embedding), "response_format": "wav"}),
        ("valid, streaming pcm", {"references": given(embedding), "response_format": "pcm", "stream": True}),
        ("valid, x_vector_only_mode true", {"references": given(embedding), "x_vector_only_mode": True}),
        ("valid, task_type Base", {"references": given(embedding), "task_type": "Base"}),
        ("valid, with the reference's text", {"references": given(embedding, text="unused")}),
        ("valid, with ref_text", {"references": given(embedding), "ref_text": "unused"}),
        ("one value short", {"references": given(embedding[:-1])}),
        ("empty", {"references": given([])}),
        ("a string item", {"references": given(["x"] + embedding[1:])}),
        ("a boolean item", {"references": given([True] + embedding[1:])}),
        ("nested", {"references": given([embedding])}),
        ("an object", {"references": given({"values": embedding})}),
        ("with audio in the same reference", {"references": given(embedding, audio_path=args.reference)}),
        ("with a second reference", {"references": given(embedding) + [{"audio_path": args.reference}]}),
        ("with ref_audio", {"references": given(embedding), "ref_audio": args.reference}),
        ("with x_vector_only_mode false", {"references": given(embedding), "x_vector_only_mode": False}),
        ("with task_type CustomVoice", {"references": given(embedding), "task_type": "CustomVoice"}),
        ("with task_type VoiceDesign", {"references": given(embedding), "task_type": "VoiceDesign", "instructions": "warm"}),
        ("NaN first", {"references": given([float("nan")] + embedding[1:])}),
        ("Infinity first", {"references": given([float("inf")] + embedding[1:])}),
        ("1e39 first (past float32)", {"references": given([1e39] + embedding[1:])}),
        ("3.4e38 first (past bfloat16)", {"references": given([3.4e38] + embedding[1:])}),
        ("1e6 everywhere", {"references": given([1e6] * len(embedding))}),
        ("zeros", {"references": given([0.0] * len(embedding))}),
        ("the old top-level field", {"speaker_embedding": embedding}),
    ]
    with httpx.Client(timeout=TIMEOUT_S, trust_env=False) as client:
        for name, extra in cases:
            status, content, latency = speech(client, args.base_url, {**base, **extra})
            detail = content[:160].decode("utf-8", "replace") if status != 200 else f"{len(content)} bytes"
            health_status, health_content, _ = speech(
                client, args.base_url, {**base, "references": given(embedding), "response_format": "wav"}
            )
            print(
                f"{name}: HTTP {status} ({latency:.0f} ms) {detail} | then a valid request: HTTP {health_status} "
                f"{len(health_content)} bytes",
                flush=True,
            )
        batch = {
            "model": args.model, "references": given(embedding), "seed": SEED, "response_format": "wav",
            "items": [{"input": "First line."}, {"input": "Second line."}, {"input": "Third.", "ref_audio": args.reference}],
        }
        response = client.post(f"{args.base_url}/v1/audio/speech/batch", json=batch)
        results = response.json().get("results", []) if response.status_code == 200 else response.text[:200]
        print(
            f"batch: HTTP {response.status_code} "
            f"{[(r.get('index'), r.get('status'), (r.get('error') or {}).get('message', '')[:80]) for r in results] if isinstance(results, list) else results}",
            flush=True,
        )
        with open(args.reference, "rb") as handle:
            upload = client.post(
                f"{args.base_url}/v1/audio/voices",
                files={"audio_sample": ("probe.wav", handle, "audio/wav")},
                data={"name": "probe_voice", "consent": "recorded with consent for testing"},
            )
        print(f"upload voice: HTTP {upload.status_code} {upload.text[:120]}", flush=True)
        status, content, _ = speech(
            client, args.base_url, {**base, "voice": "probe_voice", "references": given(embedding), "response_format": "wav"}
        )
        reference_only = speech(client, args.base_url, {**base, "references": given(embedding), "response_format": "wav"})
        print(
            f"uploaded voice name with an embedding reference: HTTP {status} {len(content)} bytes, same bytes as the "
            f"reference alone {content == reference_only[1]}",
            flush=True,
        )
        if args.custom_voice_url:
            status, content, _ = speech(
                client, args.custom_voice_url,
                {"model": args.custom_voice_model, "input": base["input"], "references": given(embedding)},
            )
            print(f"CustomVoice server with embedding: HTTP {status} {content[:160].decode('utf-8', 'replace')}", flush=True)
        else:
            pass
    for name, session in (
        ("websocket, embedding", {"model": args.model, "references": given(embedding), "response_format": "pcm", "stream_audio": True}),
        ("websocket, embedding with ref_audio", {"model": args.model, "references": given(embedding), "ref_audio": args.reference}),
        ("websocket, NaN", {"model": args.model, "references": given([float("nan")] + embedding[1:])}),
        ("websocket, 1e39", {"model": args.model, "references": given([1e39] + embedding[1:])}),
        ("websocket, 3.4e38", {"model": args.model, "references": given([3.4e38] + embedding[1:])}),
    ):
        try:
            outcome = asyncio.run(websocket_case(args.base_url, session))
        except Exception as error:
            outcome = f"{type(error).__name__}: {error}"
        print(f"{name}: {outcome}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    exact = commands.add_parser("exact")
    exact.add_argument("--base-url", required=True)
    exact.add_argument("--model", required=True)
    exact.add_argument("--dump-dir", required=True)
    exact.add_argument("--out", required=True)
    exact.add_argument("--from-dumps", action="store_true")
    contract = commands.add_parser("contract")
    contract.add_argument("--base-url", required=True)
    contract.add_argument("--model", required=True)
    contract.add_argument("--embedding", required=True)
    contract.add_argument("--reference", required=True)
    contract.add_argument("--custom-voice-url", default="")
    contract.add_argument("--custom-voice-model", default="")
    args = parser.parse_args()
    if args.command == "exact":
        run_exact(args)
    else:
        run_contract(args)


if __name__ == "__main__":
    sys.exit(main())
