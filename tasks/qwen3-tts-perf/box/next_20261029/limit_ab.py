"""The CustomVoice and VoiceDesign limit branch against main, where it acts (STATE 10-29 item 5): requests that send no
max_new_tokens. Main gives them #3157's text budget, the branch the checkpoint's generation_config limit.

usage: python3 limit_ab.py cases OUT_DIR --custom-voice-model PATH        (budget_scan's wide cases, Chinese only)
       python3 limit_ab.py scan CASES.jsonl --base-url URL --model PATH --label L --out RESULTS.jsonl [--concurrency 32]
       python3 limit_ab.py score RESULTS.jsonl --asr-port PORT [--concurrency 64]
       python3 limit_ab.py compare MAIN.scored.jsonl BRANCH.scored.jsonl

scan sends each case once, seeded, non-streaming, without max_new_tokens, and keeps the audio of every length stop and
every natural stop past #3157's budget. score transcribes each kept output with the Qwen3-ASR server (Chinese per
character) and records its error rate and the target characters it says. compare reads, per set: length stops and
their frames on each side, the branch's natural stops past the budget, and on the cases main cut, the target
characters each side says.
"""

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from jiwer import process_words
from transformers import AutoConfig, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review_20261026"))

from budget_scan import SCAN_SEED, REQUEST_TIMEOUT_S, build_wide_cases, language_of, write_cases  # noqa: E402

from benchmarks.tasks.asr import load_router_asr, normalize_text, transcribe  # noqa: E402
from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches  # noqa: E402
from sglang_omni.models.qwen3_tts.request_builders import text_length_budget  # noqa: E402
from sglang_omni.utils.checkpoint import resolve_checkpoint  # noqa: E402


def cases(out_dir: Path, custom_voice_model: str) -> None:
    build_wide_cases(out_dir, custom_voice_model)
    for name in ("customvoice_wide", "voicedesign_wide"):
        path = out_dir / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        chinese = [row for row in rows if " zh " in f" {row['set']} "]
        write_cases(out_dir / f"{name.split('_')[0]}_zh.jsonl", chinese)


async def scan(cases_path: Path, base_url: str, model: str, label: str, out_path: Path, concurrency: int) -> None:
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.inference import qwen3_tts_model as official

    AutoConfig.register("qwen3_tts", official.Qwen3TTSConfig)
    AutoProcessor.register(official.Qwen3TTSConfig, official.Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(resolve_checkpoint(model), fix_mistral_regex=True)
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    limit = asyncio.Semaphore(concurrency)
    audio_dir = out_path.parent / f"{out_path.stem}_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        text_tokens = int(processor(text=case["input"], return_tensors="pt", padding=True)["input_ids"].shape[-1])
        body: dict[str, object] = {
            "model": model, "input": case["input"], "language": case["language"], "response_format": "wav",
            "stream": False, "seed": SCAN_SEED,
        }
        for field in ("task_type", "voice", "instructions"):
            if field in case:
                body[field] = case[field]
            else:
                pass
        record: dict[str, object] = {
            "id": case["id"], "set": case["set"], "label": label, "text_tokens": text_tokens,
            "budget": text_length_budget(text_tokens), "input": case["input"],
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
                if record["finish_reason"] == "length" or record["frames"] > record["budget"]:
                    record["audio"] = str(audio_dir / f"{case['id']}.wav")
                    Path(record["audio"]).write_bytes(response.content)
                else:
                    pass
            except (httpx.HTTPError, KeyError, ValueError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in rows))
    out_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    errors = sum("error" in record for record in records)
    lengths = sum(record.get("finish_reason") == "length" for record in records)
    print(f"{label}: {len(records)} requests, {errors} errors, {lengths} length stops", flush=True)


def score(results_path: Path, asr_port: int, concurrency: int) -> None:
    asr = load_router_asr(asr_port)
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    kept = [record for record in records if "audio" in record]

    def transcribe_one(record: dict[str, object]) -> None:
        language = language_of(record)
        reference = normalize_text(str(record["input"]), language)
        transcript = transcribe(asr, str(record["audio"]), language, "cpu")
        measures = process_words(reference, normalize_text(transcript, language))
        record["transcript"] = transcript
        record["error_rate"] = measures.wer
        record["target_said"] = measures.hits
        record["target_total"] = len(reference.split())

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(transcribe_one, kept))
    out_path = results_path.with_suffix(".scored.jsonl")
    out_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"{results_path.name}: {len(kept)} kept outputs scored into {out_path.name}", flush=True)


def compare(main_path: Path, branch_path: Path) -> None:
    def load(path: Path) -> dict[str, dict[str, object]]:
        return {
            record["id"]: record
            for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        }

    main_records, branch_records = load(main_path), load(branch_path)
    by_set: dict[str, list[str]] = {}
    for case_id, record in main_records.items():
        by_set.setdefault(str(record["set"]), []).append(case_id)
    print(
        "set | requests | errors main / branch | length stops main / branch | branch length stops' frames | "
        "branch natural stops past the budget | main's cuts kept on both: target characters said main / branch"
    )
    for name, ids in sorted(by_set.items()):
        main_set = [main_records[case_id] for case_id in ids]
        branch_set = [branch_records[case_id] for case_id in ids if case_id in branch_records]
        errors = [sum("error" in record for record in group) for group in (main_set, branch_set)]
        lengths = [[record for record in group if record.get("finish_reason") == "length"] for group in (main_set, branch_set)]
        past = [
            record for record in branch_set
            if record.get("finish_reason") == "stop" and record.get("frames", 0) > record["budget"]
        ]
        said_main = said_branch = total = paired = 0
        for record in lengths[0]:
            other = branch_records.get(record["id"], {})
            if "target_said" in record and "target_said" in other:
                said_main += int(record["target_said"])
                said_branch += int(other["target_said"])
                total += int(record["target_total"])
                paired += 1
            else:
                pass
        branch_frames = sorted(int(record["frames"]) for record in lengths[1])
        print(
            f"{name} | {len(main_set)} | {errors[0]} / {errors[1]} | {len(lengths[0])} / {len(lengths[1])} | "
            f"{branch_frames[:6]} | {len(past)} | {said_main} / {said_branch} of {total} ({paired} cases kept on both)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cases_parser = sub.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    cases_parser.add_argument("--custom-voice-model", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("cases", type=Path)
    scan_parser.add_argument("--base-url", required=True)
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument("--label", required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    scan_parser.add_argument("--concurrency", type=int, default=32)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("results", type=Path)
    score_parser.add_argument("--asr-port", type=int, required=True)
    score_parser.add_argument("--concurrency", type=int, default=64)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("main", type=Path)
    compare_parser.add_argument("branch", type=Path)
    args = parser.parse_args()
    if args.command == "cases":
        cases(args.out_dir, args.custom_voice_model)
    elif args.command == "scan":
        asyncio.run(scan(args.cases, args.base_url, args.model, args.label, args.out, args.concurrency))
    elif args.command == "score":
        score(args.results, args.asr_port, args.concurrency)
    else:
        compare(args.main, args.branch)


if __name__ == "__main__":
    main()
