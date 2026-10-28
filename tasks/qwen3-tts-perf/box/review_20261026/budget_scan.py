"""Natural Qwen3-TTS utterance lengths per task in English, against #3157's default budget and vLLM-Omni's per-task
runaway ceilings.

usage: python3 budget_scan.py cases OUT_DIR                     (from an omni tree, so benchmarks/ imports)
       python3 budget_scan.py cases-wide OUT_DIR --custom-voice-model PATH   (every speaker, EN and ZH, VoiceDesign ZH)
       python3 budget_scan.py scan CASES.jsonl --base-url URL --model PATH --label L --out RESULTS.jsonl [--save-over]
       python3 budget_scan.py summarize RESULTS.jsonl [RESULTS.jsonl ...]
       python3 budget_scan.py check RESULTS.jsonl [RESULTS.jsonl ...] --asr-port PORT

--save-over keeps the audio of every natural stop past #3157's budget; check transcribes it with the Qwen3-ASR server
on PORT and calls it valid (WER at most 0.5 against the target), a loop (more insertions than target words) or
unclear, so the natural-length envelope comes from speech that says the target.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import soundfile
from jiwer import process_words
from transformers import AutoConfig, AutoProcessor

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.asr import load_router_asr, normalize_text, transcribe
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
# A saved output with WER at most this is the target spoken; one with more insertions than target words is a loop.
VALID_WER = 0.5
SECONDS_PER_FRAME = 0.08
# A 20 ms frame is speech when its RMS is within this of the clip's loudest frame; the cut keeps speech that ends
# within the tolerance of it.
SPEECH_FLOOR_DB = -35.0
SPEECH_END_TOLERANCE_S = 0.1


def write_cases(path: Path, cases: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"{path.name}: {len(cases)} cases")


def tails(text: str) -> dict[int, str]:
    words = text.split()
    return {count: " ".join(words[-count:]) for count in (1, 3) if len(words) > count}


def build_cases(out_dir: Path) -> None:
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    base = []
    custom = []
    design = []
    for sample in samples:
        # note (ratish): the loader's clips live in a temporary directory removed when this process exits.
        wav_path = out_dir / "wavs" / f"{sample.sample_id}.wav"
        wav_path.write_bytes(Path(sample.ref_audio).read_bytes())
        reference = {"ref_audio": str(wav_path), "ref_text": sample.ref_text}
        base.append({
            "id": sample.sample_id, "set": "base", "input": sample.target_text, "task_type": "Base", **reference,
        })
        for count, tail in tails(sample.target_text).items():
            base.append({
                "id": f"{sample.sample_id}_tail{count}", "set": f"base span {count}", "input": tail,
                "task_type": "Base", **reference,
            })
            custom.append({
                "id": f"{sample.sample_id}_vivian_slow_tail{count}", "set": f"customvoice span {count} vivian slow",
                "input": tail, "task_type": "CustomVoice", "voice": "vivian", "instructions": SLOW_INSTRUCTION,
            })
            design.append({
                "id": f"{sample.sample_id}_slow_tail{count}", "set": f"voicedesign span {count} slow",
                "input": tail, "task_type": "VoiceDesign", "instructions": VOICE_DESIGN_SLOW,
            })
        for speaker in ("ryan", "vivian"):
            for style, instructions in (("plain", None), ("slow", SLOW_INSTRUCTION)):
                custom.append({
                    "id": f"{sample.sample_id}_{speaker}_{style}", "set": f"customvoice {speaker} {style}",
                    "input": sample.target_text, "task_type": "CustomVoice", "voice": speaker,
                    **({"instructions": instructions} if instructions else {}),
                })
        for style, description in (("plain", VOICE_DESIGN_PLAIN), ("slow", VOICE_DESIGN_SLOW)):
            design.append({
                "id": f"{sample.sample_id}_{style}", "set": f"voicedesign {style}",
                "input": sample.target_text, "task_type": "VoiceDesign", "instructions": description,
            })
    write_cases(out_dir / "base.jsonl", base)
    write_cases(out_dir / "customvoice.jsonl", custom)
    write_cases(out_dir / "voicedesign.jsonl", design)


def build_wide_cases(out_dir: Path, custom_voice_model: str) -> None:
    """Every CustomVoice speaker (the checkpoint's talker_config.spk_id) on every SeedTTS EN and ZH sentence, plain;
    the slow instruction on every ZH sentence for ryan and vivian; VoiceDesign plain and slow on every ZH sentence.
    Each case carries its language."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(resolve_checkpoint(custom_voice_model)) / "config.json").read_text(encoding="utf-8"))
    speakers = sorted(config["talker_config"]["spk_id"])
    custom = []
    design = []
    for split, language in (("en", "English"), ("zh", "Chinese")):
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split=split):
            for speaker in speakers:
                custom.append({
                    "id": f"{split}_{sample.sample_id}_{speaker}", "set": f"customvoice {split} {speaker} plain",
                    "input": sample.target_text, "language": language, "task_type": "CustomVoice", "voice": speaker,
                })
            if split == "zh":
                for speaker in ("ryan", "vivian"):
                    custom.append({
                        "id": f"{split}_{sample.sample_id}_{speaker}_slow", "set": f"customvoice {split} {speaker} slow",
                        "input": sample.target_text, "language": language, "task_type": "CustomVoice",
                        "voice": speaker, "instructions": SLOW_INSTRUCTION,
                    })
                for style, description in (("plain", VOICE_DESIGN_PLAIN), ("slow", VOICE_DESIGN_SLOW)):
                    design.append({
                        "id": f"{split}_{sample.sample_id}_{style}", "set": f"voicedesign {split} {style}",
                        "input": sample.target_text, "language": language, "task_type": "VoiceDesign",
                        "instructions": description,
                    })
            else:
                pass
    print(f"speakers: {', '.join(speakers)}")
    write_cases(out_dir / "customvoice_wide.jsonl", custom)
    write_cases(out_dir / "voicedesign_wide.jsonl", design)


async def scan(
    cases_path: Path, base_url: str, model: str, label: str, out_path: Path, concurrency: int, save_over: bool
) -> None:
    # note (ratish): qwen-tts imports only after the transformers shim the server applies first.
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.inference import qwen3_tts_model as official

    AutoConfig.register("qwen3_tts", official.Qwen3TTSConfig)
    AutoProcessor.register(official.Qwen3TTSConfig, official.Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(resolve_checkpoint(model), fix_mistral_regex=True)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    limit = asyncio.Semaphore(concurrency)
    audio_dir = out_path.parent / f"{out_path.stem}_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        text_tokens = int(processor(text=case["input"], return_tensors="pt", padding=True)["input_ids"].shape[-1])
        body: dict[str, object] = {
            "model": model, "input": case["input"], "language": case.get("language", "English"), "response_format": "wav",
            "stream": False, "seed": SCAN_SEED, "max_new_tokens": SCAN_MAX_NEW_TOKENS,
        }
        for field in ("task_type", "voice", "instructions", "ref_text"):
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
            "text_tokens": text_tokens, "input": case["input"],
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
                if (
                    save_over
                    and record["finish_reason"] == "stop"
                    and record["frames"] > text_length_budget(text_tokens)
                ):
                    record["audio"] = str(audio_dir / f"{case['id']}.wav")
                    Path(record["audio"]).write_bytes(response.content)
                else:
                    pass
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


def language_of(record: dict[str, object]) -> str:
    """The ASR language of a record: zh for the wide cases' Chinese sets, en otherwise (10-26's sets are English)."""
    if " zh " in f" {record['set']} ":
        return "zh"
    else:
        return "en"


def check(paths: list[Path], asr_port: int, concurrency: int) -> None:
    """Transcribe every saved natural stop past #3157's budget and sort it into valid speech, a loop or unclear.
    Chinese is scored per character, as SeedTTS ZH scores it."""
    asr = load_router_asr(asr_port)
    records = [
        json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    saved = [record for record in records if "audio" in record]

    def transcribe_one(record: dict[str, object]) -> None:
        language = language_of(record)
        transcript = transcribe(asr, str(record["audio"]), language, "cpu")
        reference = normalize_text(str(record["input"]), language)
        measures = process_words(reference, normalize_text(transcript, language))
        record["transcript"] = transcript
        record["wer"] = measures.wer
        record["insertions"] = measures.insertions
        reference_words = len(reference.split())
        if measures.wer <= VALID_WER:
            record["verdict"] = "valid"
        elif measures.insertions > reference_words:
            record["verdict"] = "loop"
        else:
            record["verdict"] = "unclear"

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(transcribe_one, saved))
    out_path = paths[0].parent / "check.jsonl"
    out_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in saved), encoding="utf-8")
    print(
        "set | cut by #3157: valid / loop / unclear | cut by vLLM-Omni's ceiling: valid / loop / unclear | "
        "longest valid stop past #3157's budget, frames (tokens, frames per token)"
    )
    by_set: dict[str, list[dict[str, object]]] = {}
    for record in saved:
        by_set.setdefault(str(record["set"]), []).append(record)
    for name, flagged in sorted(by_set.items()):
        over_vllm = [
            record for record in flagged
            if record["frames"] > vllm_omni_ceiling(str(record["task"]), record["text_tokens"])
        ]
        valid = [record for record in flagged if record["verdict"] == "valid"]
        longest = max(valid, key=lambda record: record["frames"], default=None)
        counts = [
            " / ".join(str(sum(record["verdict"] == verdict for record in group)) for verdict in ("valid", "loop", "unclear"))
            for group in (flagged, over_vllm)
        ]
        longest_text = (
            f"{longest['frames']} ({longest['text_tokens']}, {longest['frames'] / longest['text_tokens']:.1f})"
            if longest is not None
            else "-"
        )
        print(f"{name} | {counts[0]} | {counts[1]} | {longest_text}")


def last_voiced_s(wav_path: str) -> float:
    """Time the last 20 ms frame ends whose RMS is within SPEECH_FLOOR_DB of the clip's loudest frame."""
    audio, rate = soundfile.read(wav_path, dtype="float32", always_2d=False)
    frame = int(rate * 0.02)
    frames = len(audio) // frame
    rms = np.sqrt(np.mean(audio[: frames * frame].reshape(frames, frame) ** 2, axis=1))
    voiced = np.flatnonzero(rms > max(rms.max() * 10 ** (SPEECH_FLOOR_DB / 20), 1e-4))
    return float((voiced[-1] + 1) * frame / rate) if voiced.size else 0.0


def speech(check_path: Path) -> None:
    """Whether each checked output past #3157's budget still speaks after the budget's cut, or only falls silent."""
    records = [json.loads(line) for line in check_path.read_text(encoding="utf-8").splitlines() if line]
    print(
        "set | cut by #3157: speech lost / silence only / loop | cut by vLLM-Omni's ceiling: speech lost / "
        "silence only / loop | latest speech end past #3157's cut, s (tokens)"
    )
    by_set: dict[str, list[dict[str, object]]] = {}
    for record in records:
        record["speech_end_s"] = last_voiced_s(str(record["audio"]))
        by_set.setdefault(str(record["set"]), []).append(record)

    def verdict(record: dict[str, object], cut_frames: int) -> str:
        if record["verdict"] == "loop":
            return "loop"
        elif record["speech_end_s"] > cut_frames * SECONDS_PER_FRAME + SPEECH_END_TOLERANCE_S:
            return "speech lost"
        else:
            return "silence only"

    for name, flagged in sorted(by_set.items()):
        counts = []
        for cut_of in (
            lambda record: text_length_budget(record["text_tokens"]),
            lambda record: vllm_omni_ceiling(str(record["task"]), record["text_tokens"]),
        ):
            over = [record for record in flagged if record["frames"] > cut_of(record)]
            verdicts = [verdict(record, cut_of(record)) for record in over]
            counts.append(" / ".join(str(verdicts.count(kind)) for kind in ("speech lost", "silence only", "loop")))
        lost = [
            record for record in flagged
            if verdict(record, text_length_budget(record["text_tokens"])) == "speech lost"
        ]
        worst = max(
            lost,
            key=lambda record: record["speech_end_s"] - text_length_budget(record["text_tokens"]) * SECONDS_PER_FRAME,
            default=None,
        )
        worst_text = (
            f"{worst['speech_end_s'] - text_length_budget(worst['text_tokens']) * SECONDS_PER_FRAME:.1f} "
            f"({worst['text_tokens']})"
            if worst is not None
            else "-"
        )
        print(f"{name} | {counts[0]} | {counts[1]} | {worst_text}")
    check_path.with_name("speech.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )


def truncated(check_path: Path, asr_port: int, concurrency: int) -> None:
    """Each checked output that speaks the target, cut where #3157's budget and vLLM-Omni's ceiling would stop it and
    transcribed again: a cut that drops target words loses natural speech."""
    asr = load_router_asr(asr_port)
    records = [
        json.loads(line) for line in check_path.read_text(encoding="utf-8").splitlines() if line
    ]
    spoken = [record for record in records if record["verdict"] != "loop"]
    cut_dir = check_path.parent / "cut_audio"
    cut_dir.mkdir(exist_ok=True)

    def transcribe_cut(record: dict[str, object]) -> None:
        audio, rate = soundfile.read(str(record["audio"]), dtype="float32", always_2d=False)
        language = language_of(record)
        reference = normalize_text(str(record["input"]), language)
        full_hits = process_words(reference, normalize_text(str(record["transcript"]), language)).hits
        for name, cut_frames in (
            ("3157", text_length_budget(record["text_tokens"])),
            ("vllm", vllm_omni_ceiling(str(record["task"]), record["text_tokens"])),
        ):
            if record["frames"] <= cut_frames:
                record[f"words_lost_{name}"] = 0
                continue
            else:
                pass
            cut_path = cut_dir / f"{Path(str(record['audio'])).stem}_{name}.wav"
            soundfile.write(cut_path, audio[: int(cut_frames * SECONDS_PER_FRAME * rate)], rate, subtype="PCM_16")
            cut_hits = process_words(
                reference, normalize_text(transcribe(asr, str(cut_path), language, "cpu"), language)
            ).hits
            record[f"words_lost_{name}"] = max(full_hits - cut_hits, 0)

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(transcribe_cut, spoken))
    print(
        "set | spoken outputs past #3157's budget | of which the cut drops target words (#3157) | words dropped "
        "| of which the cut drops target words (vLLM-Omni)"
    )
    by_set: dict[str, list[dict[str, object]]] = {}
    for record in spoken:
        by_set.setdefault(str(record["set"]), []).append(record)
    for name, group in sorted(by_set.items()):
        lossy_3157 = [record for record in group if record["words_lost_3157"] > 0]
        lossy_vllm = [record for record in group if record["words_lost_vllm"] > 0]
        print(
            f"{name} | {len(group)} | {len(lossy_3157)} | {sum(record['words_lost_3157'] for record in group)} "
            f"| {len(lossy_vllm)}"
        )
    check_path.with_name("truncated.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in spoken), encoding="utf-8"
    )


def reference_rate(results_path: Path, cases_path: Path, model: str) -> None:
    """For Base clones: the output's frames per target token against the reference's own frames per reference-text
    token (its audio at 12.5 frames a second), over natural stops, sentences and spans apart."""
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.inference import qwen3_tts_model as official

    AutoConfig.register("qwen3_tts", official.Qwen3TTSConfig)
    AutoProcessor.register(official.Qwen3TTSConfig, official.Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(resolve_checkpoint(model), fix_mistral_regex=True)
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line)
    }
    reference_frames_per_token: dict[str, float] = {}
    rows: dict[str, list[tuple[float, float]]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("finish_reason") != "stop":
            continue
        else:
            pass
        case = cases[record["id"]]
        if case["ref_audio"] not in reference_frames_per_token:
            reference_tokens = int(processor(text=case["ref_text"], return_tensors="pt")["input_ids"].shape[-1])
            reference_s = soundfile.info(case["ref_audio"]).duration
            reference_frames_per_token[case["ref_audio"]] = reference_s / SECONDS_PER_FRAME / reference_tokens
        else:
            pass
        rows.setdefault(str(record["set"]), []).append(
            (reference_frames_per_token[case["ref_audio"]], record["frames"] / record["text_tokens"])
        )
    print("set | natural stops | corr(reference rate, output rate) | output / reference rate p50 / p99 / max")
    for name, pairs in sorted(rows.items()):
        reference, output = (np.array(values) for values in zip(*pairs))
        ratio = np.sort(output / reference)
        print(
            f"{name} | {len(pairs)} | {np.corrcoef(reference, output)[0, 1]:.2f} | "
            f"{ratio[len(ratio) // 2]:.2f} / {ratio[int(len(ratio) * 0.99)]:.2f} / {ratio[-1]:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    wide_parser = commands.add_parser("cases-wide")
    wide_parser.add_argument("out_dir", type=Path)
    wide_parser.add_argument("--custom-voice-model", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("cases", type=Path)
    scan_parser.add_argument("--base-url", required=True)
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument("--label", required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    scan_parser.add_argument("--concurrency", type=int, default=32)
    scan_parser.add_argument("--save-over", action="store_true")
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("results", type=Path, nargs="+")
    check_parser = commands.add_parser("check")
    check_parser.add_argument("results", type=Path, nargs="+")
    check_parser.add_argument("--asr-port", type=int, required=True)
    check_parser.add_argument("--concurrency", type=int, default=16)
    speech_parser = commands.add_parser("speech")
    speech_parser.add_argument("check", type=Path)
    rate_parser = commands.add_parser("refrate")
    rate_parser.add_argument("results", type=Path)
    rate_parser.add_argument("cases", type=Path)
    rate_parser.add_argument("--model", required=True)
    truncated_parser = commands.add_parser("truncated")
    truncated_parser.add_argument("check", type=Path)
    truncated_parser.add_argument("--asr-port", type=int, required=True)
    truncated_parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.command == "cases":
        build_cases(args.out_dir)
    elif args.command == "cases-wide":
        build_wide_cases(args.out_dir, args.custom_voice_model)
    elif args.command == "scan":
        asyncio.run(
            scan(args.cases, args.base_url, args.model, args.label, args.out, args.concurrency, args.save_over)
        )
    elif args.command == "check":
        check(args.results, args.asr_port, args.concurrency)
    elif args.command == "speech":
        speech(args.check)
    elif args.command == "truncated":
        truncated(args.check, args.asr_port, args.concurrency)
    elif args.command == "refrate":
        reference_rate(args.results, args.cases, args.model)
    else:
        summarize(args.results)


if __name__ == "__main__":
    sys.exit(main())
