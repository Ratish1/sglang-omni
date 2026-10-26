"""Runaways and natural lengths of Qwen3-TTS Base voice clones, our server against official qwen-tts, on the requests
the streaming cascade sends.

usage: python3 parity.py cases OUT_DIR                     (from an omni tree, so benchmarks/ imports)
       python3 parity.py serve CASES.jsonl --base-url URL --model PATH --label L --out OUT.jsonl [--max-new-tokens 300]
       python3 parity.py compare REFERENCE.json SERVED.jsonl [--max-new-tokens 300]

cases: every SeedTTS EN sample (1,088) as its sentence, its last word and its last three words, each cloned with its own
reference in ICL mode (the transcript sent) and x-vector-only mode (no transcript), the clips copied beside the cases.
serve: each case as the cascade's synthesizer posts it (streaming PCM, voice default, language English, references
with the clip's server-side path and its text when ICL, no seed), with max_new_tokens set so both arms share one cap;
the finish reason and frame count come from the stream's outcome. compare: per set, runaways (no codec EOS within the
cap) in each arm with Fisher's exact test, and the natural stops' frame counts with a two-sample KS test.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
from scipy.stats import fisher_exact, ks_2samp

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples

REQUEST_TIMEOUT_S = 900.0
PROGRESS_EVERY = 500


def build_cases(out_dir: Path) -> None:
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    cases = []
    for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en"):
        # note (ratish): the loader's clips live in a temporary directory removed when this process exits.
        wav_path = out_dir / "wavs" / f"{sample.sample_id}.wav"
        wav_path.write_bytes(Path(sample.ref_audio).read_bytes())
        words = sample.target_text.split()
        texts = {"sentence": sample.target_text}
        for count in (1, 3):
            if len(words) > count:
                texts[f"last {count}"] = " ".join(words[-count:])
            else:
                pass
        for mode in ("icl", "xvec"):
            for kind, text in texts.items():
                cases.append({
                    "id": f"{sample.sample_id}_{mode}_{kind.replace(' ', '')}", "set": f"{mode} {kind}",
                    "mode": mode, "input": text, "ref_audio": str(wav_path),
                    **({"ref_text": sample.ref_text} if mode == "icl" else {}),
                })
    (out_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
    )
    print(f"{len(cases)} cases")


async def serve(cases_path: Path, base_url: str, model: str, label: str, out_path: Path, max_new_tokens: int,
                concurrency: int) -> None:
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    limit = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "runaways": 0, "errors": 0}

    def report(record: dict[str, object]) -> None:
        progress["done"] += 1
        progress["runaways"] += int(record.get("frames", 0) >= max_new_tokens)
        progress["errors"] += int("error" in record)
        if progress["done"] % PROGRESS_EVERY == 0 or progress["done"] == len(cases):
            print(
                f"{label}: {progress['done']} of {len(cases)} done, {progress['runaways']} runaways, "
                f"{progress['errors']} errors",
                flush=True,
            )
        else:
            pass

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        record = await request(client, case)
        report(record)
        return record

    async def request(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        reference: dict[str, str] = {"audio_path": case["ref_audio"]}
        if "ref_text" in case:
            reference["text"] = case["ref_text"]
        else:
            pass
        body = {
            "model": model, "voice": "default", "input": case["input"], "response_format": "pcm", "stream": True,
            "language": "English", "references": [reference], "max_new_tokens": max_new_tokens,
        }
        record: dict[str, object] = {"id": case["id"], "set": case["set"], "label": label}
        async with limit:
            try:
                async with client.stream("POST", f"{base_url}/v1/audio/speech", json=body) as response:
                    if response.status_code != 200:
                        await response.aread()
                        record["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
                        return record
                    else:
                        pass
                    speech_id = response.headers["X-SGLang-Omni-Speech-Id"]
                    worker = response.headers.get("X-SGLang-Omni-Worker")
                    record["pcm_bytes"] = sum([len(chunk) async for chunk in response.aiter_bytes()])
                outcome = await client.get(
                    f"{base_url}/v1/audio/speech/{quote(speech_id, safe='')}",
                    headers={"x-sglang-omni-route-worker": worker} if worker else {},
                )
                outcome.raise_for_status()
                data = outcome.json()
                record["finish_reason"] = data.get("finish_reason")
                record["frames"] = int((data.get("usage") or {}).get("completion_tokens"))
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in cases))
    out_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    errors = [record for record in records if "error" in record]
    print(f"{label}: {len(records)} requests, {len(errors)} errors{': ' + errors[0]['error'] if errors else ''}")


def compare(reference_path: Path, served_paths: list[Path], max_new_tokens: int) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    arms: dict[str, dict[str, list[int]]] = {"official": {}}
    for record in reference["records"]:
        arms["official"].setdefault(record["set"], []).append(record["frames"])
    for path in served_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "frames" in record:
                arms.setdefault(str(record["label"]), {}).setdefault(record["set"], []).append(record["frames"])
            else:
                pass
    print(f"{reference['model']}, cap {max_new_tokens} frames; runaway = no codec EOS within the cap")
    print("set | arm | requests | runaways (rate) | Fisher p against official | natural frames median / p99 / max "
          "| KS p against official")
    for set_name in sorted(arms["official"]):
        official = arms["official"][set_name]
        official_runaways = sum(frames >= max_new_tokens for frames in official)
        official_natural = sorted(frames for frames in official if frames < max_new_tokens)
        for arm, sets in arms.items():
            frames = sets.get(set_name, [])
            if not frames:
                continue
            else:
                pass
            runaways = sum(count >= max_new_tokens for count in frames)
            natural = sorted(count for count in frames if count < max_new_tokens)
            if arm == "official":
                fisher_text = "-"
                ks_text = "-"
            else:
                fisher_text = (
                    f"{fisher_exact([[runaways, len(frames) - runaways], [official_runaways, len(official) - official_runaways]])[1]:.2f}"
                )
                ks_text = f"{ks_2samp(natural, official_natural).pvalue:.2f}"
            print(
                f"{set_name} | {arm} | {len(frames)} | {runaways} ({100 * runaways / len(frames):.2f} %) | {fisher_text} | "
                f"{natural[len(natural) // 2]} / {natural[int(len(natural) * 0.99)]} / {natural[-1]} | {ks_text}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("cases", type=Path)
    serve_parser.add_argument("--base-url", required=True)
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--label", required=True)
    serve_parser.add_argument("--out", type=Path, required=True)
    serve_parser.add_argument("--max-new-tokens", type=int, default=300)
    serve_parser.add_argument("--concurrency", type=int, default=16)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("served", type=Path, nargs="+")
    compare_parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()
    if args.command == "cases":
        build_cases(args.out_dir)
    elif args.command == "serve":
        asyncio.run(serve(args.cases, args.base_url, args.model, args.label, args.out, args.max_new_tokens,
                          args.concurrency))
    else:
        compare(args.reference, args.served, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())
