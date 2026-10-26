"""Natural Qwen3-TTS utterance lengths per task in English, against #3157's default budget and vLLM-Omni's per-task
runaway ceilings.

usage: python3 budget_scan.py cases OUT_DIR                     (from an omni tree, so benchmarks/ imports)
       python3 budget_scan.py scan CASES.jsonl --base-url URL --model PATH --label L --out RESULTS.jsonl
       python3 budget_scan.py summarize RESULTS.jsonl [RESULTS.jsonl ...]

cases writes one JSONL per task from SeedTTS EN (1,088 samples): Base with each sample's own reference; CustomVoice
with a native English speaker (ryan) and a non-native one (vivian), each plain and with a slow-speech instruction;
VoiceDesign with a plain and a slow description; and short spans (each target's last word and last three words, as
the cascade sends a short sentence) for Base, and with the slow instruction or description for the other two tasks.
scan sends every case once with max_new_tokens 2048 (main's flat default before #3157), seeded, non-streaming, and
records the finish reason and codec frames from the response headers. Text tokens are the official processor's
count of the target, as both budgets count them. A natural stop (finish "stop") past a budget is speech that budget
cuts; a stop at 2048 is a runaway.
"""

import argparse
import asyncio
import base64
import json
import math
import sys
from pathlib import Path

import httpx
from transformers import AutoConfig, AutoProcessor

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.request_builders import text_length_budget
from sglang_omni.utils.checkpoint import resolve_checkpoint

SCAN_MAX_NEW_TOKENS = 2048
SCAN_SEED = 1234
REQUEST_TIMEOUT_S = 900.0
SLOW_INSTRUCTION = "Speak very slowly and calmly, with long pauses between phrases."
VOICE_DESIGN_PLAIN = "A young woman with a clear, warm voice speaking at a natural pace."
VOICE_DESIGN_SLOW = (
    "A very old man with a deep, tired voice who speaks extremely slowly, pausing for a long time between words."
)
# vLLM-Omni's Qwen3-TTS runaway ceilings (tts_adapters/qwen3_tts.py DEFAULT_CODEC_BUDGETS): min frames, frames per token.
VLLM_OMNI_CEILINGS = {"base": (192, 12.0), "customvoice": (400, 40.0), "voicedesign": (400, 40.0)}


def write_cases(path: Path, cases: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"{path.name}: {len(cases)} cases")


def tails(text: str) -> dict[int, str]:
    words = text.split()
    return {count: " ".join(words[-count:]) for count in (1, 3) if len(words) > count}


def build_cases(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    base = []
    custom = []
    design = []
    for sample in samples:
        reference = {"ref_audio": sample.ref_audio, "ref_text": sample.ref_text}
        base.append({"id": sample.sample_id, "set": "base", "input": sample.target_text, **reference})
        for count, tail in tails(sample.target_text).items():
            base.append({"id": f"{sample.sample_id}_tail{count}", "set": f"base span {count}", "input": tail, **reference})
            custom.append({
                "id": f"{sample.sample_id}_vivian_slow_tail{count}", "set": f"customvoice span {count} vivian slow",
                "input": tail, "voice": "vivian", "instructions": SLOW_INSTRUCTION,
            })
            design.append({
                "id": f"{sample.sample_id}_slow_tail{count}", "set": f"voicedesign span {count} slow",
                "input": tail, "instructions": VOICE_DESIGN_SLOW,
            })
        for speaker in ("ryan", "vivian"):
            for style, instructions in (("plain", None), ("slow", SLOW_INSTRUCTION)):
                custom.append({
                    "id": f"{sample.sample_id}_{speaker}_{style}", "set": f"customvoice {speaker} {style}",
                    "input": sample.target_text, "voice": speaker,
                    **({"instructions": instructions} if instructions else {}),
                })
        for style, description in (("plain", VOICE_DESIGN_PLAIN), ("slow", VOICE_DESIGN_SLOW)):
            design.append({
                "id": f"{sample.sample_id}_{style}", "set": f"voicedesign {style}",
                "input": sample.target_text, "instructions": description,
            })
    write_cases(out_dir / "base.jsonl", base)
    write_cases(out_dir / "customvoice.jsonl", custom)
    write_cases(out_dir / "voicedesign.jsonl", design)


async def scan(cases_path: Path, base_url: str, model: str, label: str, out_path: Path, concurrency: int) -> None:
    # note (ratish): qwen-tts imports only after the transformers shim the server applies first.
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.inference import qwen3_tts_model as official

    AutoConfig.register("qwen3_tts", official.Qwen3TTSConfig)
    AutoProcessor.register(official.Qwen3TTSConfig, official.Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(resolve_checkpoint(model), fix_mistral_regex=True)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    limit = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        text_tokens = int(processor(text=case["input"], return_tensors="pt", padding=True)["input_ids"].shape[-1])
        body: dict[str, object] = {
            "model": model, "input": case["input"], "language": "English", "response_format": "wav",
            "stream": False, "seed": SCAN_SEED, "max_new_tokens": SCAN_MAX_NEW_TOKENS,
        }
        for field in ("voice", "instructions", "ref_text"):
            if field in case:
                body[field] = case[field]
            else:
                pass
        if "ref_audio" in case:
            # note (ratish): the API takes reference audio as a data URI, as the SeedTTS client sends it.
            encoded = base64.b64encode(Path(case["ref_audio"]).read_bytes()).decode("ascii")
            body["ref_audio"] = f"data:audio/wav;base64,{encoded}"
        else:
            pass
        record: dict[str, object] = {
            "id": case["id"], "set": f"{label} {case['set']}", "task": label.rstrip("0123456789"),
            "text_tokens": text_tokens,
        }
        async with limit:
            try:
                response = await client.post(f"{base_url}/v1/audio/speech", json=body)
                if response.status_code != 200:
                    record["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
                    return record
                else:
                    pass
                record["finish_reason"] = response.headers.get("X-Finish-Reason")
                record["frames"] = int(response.headers["X-Completion-Tokens"])
            except (httpx.HTTPError, KeyError, ValueError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in cases))
    out_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summarize([out_path])


def vllm_omni_ceiling(task: str, text_tokens: int) -> int:
    min_frames, frames_per_token = VLLM_OMNI_CEILINGS[task]
    return max(min_frames, math.ceil(frames_per_token * text_tokens))


def summarize(paths: list[Path]) -> None:
    by_set: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            by_set.setdefault(str(record["set"]), []).append(record)
    print(
        "set | requests | errors | runaways at 2048 | natural stops | frames per token p50 / p99 / max | "
        "longest natural stop, frames (tokens) | cut by #3157 | cut by vLLM-Omni's ceiling"
    )
    for name, records in by_set.items():
        natural = [record for record in records if record.get("finish_reason") == "stop"]
        capped = [record for record in records if record.get("finish_reason") == "length"]
        errors = [record for record in records if "error" in record]
        rates = sorted(record["frames"] / max(record["text_tokens"], 1) for record in natural)
        longest = max(natural, key=lambda record: record["frames"], default=None)
        cut_3157 = [record for record in natural if record["frames"] > text_length_budget(record["text_tokens"])]
        cut_vllm = [
            record for record in natural
            if record["frames"] > vllm_omni_ceiling(str(record["task"]), record["text_tokens"])
        ]
        rate_text = (
            f"{rates[len(rates) // 2]:.1f} / {rates[min(len(rates) - 1, int(len(rates) * 0.99))]:.1f} / {rates[-1]:.1f}"
            if rates
            else "-"
        )
        longest_text = f"{longest['frames']} ({longest['text_tokens']})" if longest is not None else "-"
        print(
            f"{name} | {len(records)} | {len(errors)} | {len(capped)} | {len(natural)} | {rate_text} | "
            f"{longest_text} | {len(cut_3157)} | {len(cut_vllm)}"
        )
        for message in sorted({str(record["error"]) for record in errors})[:2]:
            print(f"  error: {message}")
        for record in sorted(cut_3157, key=lambda record: -record["frames"] / text_length_budget(record["text_tokens"]))[:5]:
            print(
                f"  cut by #3157: {record['id']} {record['text_tokens']} tokens, {record['frames']} frames, "
                f"budget {text_length_budget(record['text_tokens'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("cases", type=Path)
    scan_parser.add_argument("--base-url", required=True)
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument("--label", required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    scan_parser.add_argument("--concurrency", type=int, default=32)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "cases":
        build_cases(args.out_dir)
    elif args.command == "scan":
        asyncio.run(scan(args.cases, args.base_url, args.model, args.label, args.out, args.concurrency))
    else:
        summarize(args.results)


if __name__ == "__main__":
    sys.exit(main())
