"""Natural Qwen3-TTS utterance lengths against #3157's default budget, beyond the cells it was calibrated on.

usage: python3 budget_scan.py cases OUT_DIR                     (from an omni tree, so benchmarks/ imports)
       python3 budget_scan.py scan CASES.jsonl --base-url URL --model PATH --out RESULTS.jsonl [--concurrency 32]
       python3 budget_scan.py summarize RESULTS.jsonl [RESULTS.jsonl ...]

cases writes one JSONL per set: SeedTTS EN and ZH with each sample's own reference (Base), their targets for a
CustomVoice speaker with and without a slow-speech instruction, the same for a VoiceDesign description, FLEURS test
sentences in the eight other languages the checkpoints list (CustomVoice, plain and slow), and the cascade's short
spans built from the SeedTTS targets' last words. scan sends every case once with max_new_tokens 2048 (main's flat
default before #3157), seeded, non-streaming, and records the finish reason and codec frames from the response
headers; the target's token count is the official processor's, as #3157's budget counts it, and the budget is the
tree's text_length_budget. A natural stop (finish "stop") longer than its budget is speech #3157's default cuts.
"""

import argparse
import asyncio
import csv
import io
import json
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
FLEURS_LANGUAGES = {
    "ja_jp": ("Japanese", "ono_anna"),
    "ko_kr": ("Korean", "sohee"),
    "de_de": ("German", "ryan"),
    "fr_fr": ("French", "ryan"),
    "ru_ru": ("Russian", "ryan"),
    "pt_br": ("Portuguese", "ryan"),
    "es_419": ("Spanish", "ryan"),
    "it_it": ("Italian", "ryan"),
}
SLOW_INSTRUCTIONS = {
    "English": "Speak very slowly and calmly, with long pauses between phrases.",
    "Chinese": "请用非常缓慢、平静的语速说话，短语之间有较长的停顿。",
}
SLOW_INSTRUCTION_OTHER = "Speak very slowly and calmly, with long pauses between phrases."
VOICE_DESIGN_SLOW = (
    "A very old man with a deep, tired voice who speaks extremely slowly, pausing for a long time between words."
)
VOICE_DESIGN_PLAIN = "A young woman with a clear, warm voice speaking at a natural pace."


def write_cases(path: Path, cases: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"{path.name}: {len(cases)} cases")


def fleurs_sentences(language_code: str) -> list[tuple[str, str]]:
    url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language_code}/test.tsv"
    text = httpx.get(url, follow_redirects=True, timeout=120).raise_for_status().text
    sentences: dict[str, str] = {}
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        sentences.setdefault(row[0], row[2])
    return sorted(sentences.items())


def build_cases(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seedtts = {
        split: load_seedtts_samples(SEEDTTS_DATASET_ID, split=split) for split in ("en", "zh")
    }
    languages = {"en": "English", "zh": "Chinese"}
    write_cases(out_dir / "base_seedtts.jsonl", [
        {"id": f"{split}_{sample.sample_id}", "set": f"base seedtts {split}", "input": sample.target_text,
         "language": languages[split], "ref_audio": sample.ref_audio, "ref_text": sample.ref_text}
        for split, samples in seedtts.items() for sample in samples
    ])
    spans = []
    for split, samples in seedtts.items():
        for sample in samples:
            if split == "en":
                words = sample.target_text.split()
                tails = {count: " ".join(words[-count:]) for count in (1, 3) if len(words) > count}
            else:
                characters = sample.target_text.rstrip("。！？，、.!?,")
                tails = {3: characters[-3:] + "。"} if len(characters) > 3 else {}
            for count, tail in tails.items():
                spans.append({
                    "id": f"{split}_{sample.sample_id}_tail{count}", "set": f"base span {split} last {count}",
                    "input": tail, "language": languages[split], "ref_audio": sample.ref_audio,
                    "ref_text": sample.ref_text,
                })
    write_cases(out_dir / "base_spans.jsonl", spans)
    custom = []
    design = []
    for split, samples in seedtts.items():
        language = languages[split]
        speaker = "ryan" if split == "en" else "vivian"
        for sample in samples:
            for style, instructions in (("plain", None), ("slow", SLOW_INSTRUCTIONS[language])):
                custom.append({
                    "id": f"{split}_{sample.sample_id}_{style}", "set": f"customvoice seedtts {split} {style}",
                    "input": sample.target_text, "language": language, "voice": speaker,
                    **({"instructions": instructions} if instructions else {}),
                })
            for style, description in (("plain", VOICE_DESIGN_PLAIN), ("slow", VOICE_DESIGN_SLOW)):
                design.append({
                    "id": f"{split}_{sample.sample_id}_{style}", "set": f"voicedesign seedtts {split} {style}",
                    "input": sample.target_text, "language": language, "instructions": description,
                })
    for language_code, (language, speaker) in FLEURS_LANGUAGES.items():
        for sentence_id, sentence in fleurs_sentences(language_code):
            for style, instructions in (("plain", None), ("slow", SLOW_INSTRUCTION_OTHER)):
                custom.append({
                    "id": f"{language_code}_{sentence_id}_{style}", "set": f"customvoice fleurs {language_code} {style}",
                    "input": sentence, "language": language, "voice": speaker,
                    **({"instructions": instructions} if instructions else {}),
                })
    write_cases(out_dir / "customvoice.jsonl", custom)
    write_cases(out_dir / "voicedesign.jsonl", design)


async def scan(
    cases_path: Path, base_url: str, model: str, label: str, out_path: Path, concurrency: int
) -> None:
    # note (ratish): qwen-tts imports only after the transformers shim the server applies first.
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.inference import qwen3_tts_model as official

    AutoConfig.register("qwen3_tts", official.Qwen3TTSConfig)
    AutoProcessor.register(official.Qwen3TTSConfig, official.Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(resolve_checkpoint(model), fix_mistral_regex=True)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    audio_dir = out_path.parent / f"{out_path.stem}_over_budget"
    audio_dir.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        text_tokens = int(processor(text=case["input"], return_tensors="pt", padding=True)["input_ids"].shape[-1])
        body: dict[str, object] = {
            "model": model, "input": case["input"], "language": case["language"], "response_format": "wav",
            "stream": False, "seed": SCAN_SEED, "max_new_tokens": SCAN_MAX_NEW_TOKENS,
        }
        for field in ("voice", "instructions", "ref_audio", "ref_text"):
            if field in case:
                body[field] = case[field]
            else:
                pass
        record: dict[str, object] = {
            "id": case["id"], "set": f"{label} {case['set']}", "text_tokens": text_tokens,
            "budget_frames": text_length_budget(text_tokens),
        }
        async with limit:
            try:
                response = await client.post(f"{base_url}/v1/audio/speech", json=body)
                response.raise_for_status()
                record["finish_reason"] = response.headers.get("X-Finish-Reason")
                record["frames"] = int(response.headers["X-Completion-Tokens"])
                if record["finish_reason"] == "stop" and record["frames"] > record["budget_frames"]:
                    (audio_dir / f"{case['id']}.wav").write_bytes(response.content)
                else:
                    pass
            except (httpx.HTTPError, KeyError, ValueError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in cases))
    out_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summarize([out_path])


def summarize(paths: list[Path]) -> None:
    by_set: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            by_set.setdefault(str(record["set"]), []).append(record)
    print("set | requests | errors | natural stops | runaways at 2048 | natural stops past the budget | "
          "longest natural stop / its budget | median frames per token")
    for name, records in by_set.items():
        natural = [record for record in records if record.get("finish_reason") == "stop"]
        capped = [record for record in records if record.get("finish_reason") == "length"]
        errors = [record for record in records if "error" in record]
        cut = [record for record in natural if record["frames"] > record["budget_frames"]]
        worst = max(natural, key=lambda record: record["frames"] / record["budget_frames"], default=None)
        rates = sorted(record["frames"] / record["text_tokens"] for record in natural if record["text_tokens"])
        worst_text = (
            f"{worst['frames']} / {worst['budget_frames']} ({worst['frames'] / worst['budget_frames']:.2f}, "
            f"{worst['id']}, {worst['text_tokens']} tokens)"
            if worst is not None
            else "-"
        )
        rate_text = f"{rates[len(rates) // 2]:.2f}" if rates else "-"
        print(
            f"{name} | {len(records)} | {len(errors)} | {len(natural)} | {len(capped)} | {len(cut)} | "
            f"{worst_text} | {rate_text}"
        )
        for record in sorted(cut, key=lambda record: -record["frames"] / record["budget_frames"])[:10]:
            print(f"  cut: {record['id']} {record['text_tokens']} tokens, {record['frames']} frames, "
                  f"budget {record['budget_frames']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("cases", type=Path)
    scan_parser.add_argument("--base-url", required=True)
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    scan_parser.add_argument("--label", required=True)
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
