"""The streaming cascade's Qwen3-TTS requests replayed against one server: what a full-duplex conversation hears.

usage: python3 duplex_ab.py cases OUT_DIR --profiles DIR      (from an omni tree, so benchmarks/ and sglang_omni import)
       python3 duplex_ab.py serve CASES.jsonl --base-url URL --model PATH --label L --shape S --out DIR
           [--sets SET ...] [--concurrency 16] [--interrupt]
       python3 duplex_ab.py score DIR [DIR ...] --asr-port PORT [--sim-card CARD]
       python3 duplex_ab.py compare BASE_DIR OTHER_DIR [BASE_DIR OTHER_DIR ...]

cases: the synthesizer's request (`streaming_cascade/synthesizers.py` speak) in the three shapes the profiles send:
icl06, 0.6B Base cloning each SeedTTS EN sample's own reference with its transcript (the Nemotron profiles ship only
a placeholder reference); xvec17, 1.7B Base x-vector cloning of `voices/vivian.wav` (the vivian profiles); icl17zh,
1.7B Base cloning `reference_zh.wav` with its transcript (the Chinese clone profile). English targets are every
SeedTTS EN sentence, its last word and its last three words, sent with language English as spoken_text reads them;
Chinese targets are every SeedTTS ZH sentence, sent with language Chinese. The references are copied beside the cases.
serve: streaming PCM, voice default, references by server-side path, no seed, no max_new_tokens. Records the time to
the first audio byte, the audio, frames and finish reason from the stream outcome, and the opening and closing silence
(5 ms windows at -34 dBFS). With --interrupt each request is closed after its first audio chunk, as a barge-in
cancels the synthesizer's stream, and its outcome is read once the server has had time to abort it.
score: Qwen3-ASR transcripts, WER (English) or CER (Chinese) against the sent text, and WavLM similarity of every
natural stop to its reference.
compare: per shape and set, both arms side by side, then paired differences over the cases both arms finished.
"""

import argparse
import asyncio
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import soundfile
from jiwer import process_words

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import ensure_speaker_similarity_assets
from benchmarks.tasks.asr import load_router_asr, normalize_text, transcribe
from sglang_omni.models.streaming_cascade.numerals import spoken_text

REQUEST_TIMEOUT_S = 900.0
INTERRUPT_SETTLE_S = 2.0
PROGRESS_EVERY = 1000
SAMPLE_RATE_HZ = 24000
ONSET_WINDOW_SAMPLES = 120
ONSET_FLOOR_DBFS = -34.0
SIMILARITY_BATCH = 8
WER_FAILURE = 0.5
VIVIAN = "voices/vivian.wav"
CHINESE_CLONE = "profiles/qwen3-asr-qwen3-tts-base-clone/reference_zh.wav"
CHINESE_CLONE_TEXT = "嗯，如果说想去看美丽的风景啊，当然有啦。嗯，像之前电影当中的。"


def silence_edges_ms(samples: np.ndarray) -> tuple[float, float]:
    """Milliseconds before the first and after the last 5 ms window at ONSET_FLOOR_DBFS or louder."""
    windows = len(samples) // ONSET_WINDOW_SAMPLES
    if windows == 0:
        return 0.0, 0.0
    else:
        pass
    blocks = samples[: windows * ONSET_WINDOW_SAMPLES].reshape(windows, ONSET_WINDOW_SAMPLES)
    loud = np.flatnonzero(np.sqrt(np.mean(blocks**2, axis=1)) >= 10 ** (ONSET_FLOOR_DBFS / 20))
    window_ms = 1000 * ONSET_WINDOW_SAMPLES / SAMPLE_RATE_HZ
    if loud.size == 0:
        return windows * window_ms, windows * window_ms
    else:
        pass
    return float(loud[0] * window_ms), float((windows - 1 - loud[-1]) * window_ms)


def build_cases(out_dir: Path, profiles: Path) -> None:
    refs = out_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    vivian = refs / "vivian.wav"
    shutil.copy(profiles / VIVIAN, vivian)
    chinese_clone = refs / "reference_zh.wav"
    shutil.copy(profiles / CHINESE_CLONE, chinese_clone)
    english = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    chinese = load_seedtts_samples(SEEDTTS_DATASET_ID, split="zh")
    cases = []

    def add(shape: str, set_name: str, sample_id: str, text: str, lang: str, ref_audio: Path, ref_text: str | None) -> None:
        language = "English" if lang == "en" else "Chinese"
        cases.append({
            "id": f"{shape}_{set_name.replace(' ', '_')}_{sample_id}", "shape": shape, "set": set_name, "lang": lang,
            "language": language, "input": spoken_text(text, language)[0] if lang == "en" else text,
            "ref_audio": str(ref_audio), **({"ref_text": ref_text} if ref_text else {}),
        })

    for sample in english:
        own = refs / f"en_{sample.sample_id}.wav"
        own.write_bytes(Path(sample.ref_audio).read_bytes())
        words = sample.target_text.split()
        texts = {"en sentence": sample.target_text}
        for count in (1, 3):
            if len(words) > count:
                texts[f"en last{count}"] = " ".join(words[-count:])
            else:
                pass
        for set_name, text in texts.items():
            add("icl06", set_name, sample.sample_id, text, "en", own, sample.ref_text)
            add("xvec17", set_name, sample.sample_id, text, "en", vivian, None)
        add("icl17zh", "en sentence", sample.sample_id, sample.target_text, "en", chinese_clone, CHINESE_CLONE_TEXT)
    for sample in chinese:
        add("xvec17", "zh sentence", sample.sample_id, sample.target_text, "zh", vivian, None)
        add("icl17zh", "zh sentence", sample.sample_id, sample.target_text, "zh", chinese_clone, CHINESE_CLONE_TEXT)
    (out_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
    )
    counts = defaultdict(int)
    for case in cases:
        counts[(case["shape"], case["set"])] += 1
    print(f"{len(cases)} cases: " + ", ".join(f"{shape} {set_name} {n}" for (shape, set_name), n in sorted(counts.items())))


async def serve(cases_path: Path, base_url: str, model: str, label: str, shape: str, sets: list[str], out_dir: Path,
                concurrency: int, interrupt: bool) -> None:
    cases = [
        case for case in (json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line)
        if case["shape"] == shape and (not sets or case["set"] in sets)
    ]
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)
    progress = defaultdict(int)

    async def request(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        reference: dict[str, str] = {"audio_path": case["ref_audio"]}
        if "ref_text" in case:
            reference["text"] = case["ref_text"]
        else:
            pass
        body = {
            "model": model, "voice": "default", "input": case["input"], "response_format": "pcm", "stream": True,
            "language": case["language"], "references": [reference],
        }
        record: dict[str, object] = {
            "id": case["id"], "shape": case["shape"], "set": case["set"], "lang": case["lang"], "label": label,
            "input": case["input"], "ref_audio": case["ref_audio"], "concurrency": concurrency, "interrupt": interrupt,
        }
        async with limit:
            try:
                start = time.perf_counter()
                chunks = []
                async with client.stream("POST", f"{base_url}/v1/audio/speech", json=body) as response:
                    if response.status_code != 200:
                        await response.aread()
                        record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                        return record
                    else:
                        pass
                    speech_id = response.headers["X-SGLang-Omni-Speech-Id"]
                    worker = response.headers.get("X-SGLang-Omni-Worker")
                    async for chunk in response.aiter_bytes():
                        if chunk and "ttfa_ms" not in record:
                            record["ttfa_ms"] = 1000 * (time.perf_counter() - start)
                        else:
                            pass
                        chunks.append(chunk)
                        if interrupt and "ttfa_ms" in record:
                            break
                        else:
                            pass
                record["total_ms"] = 1000 * (time.perf_counter() - start)
                if interrupt:
                    await asyncio.sleep(INTERRUPT_SETTLE_S)
                else:
                    pass
                outcome = await client.get(
                    f"{base_url}/v1/audio/speech/{quote(speech_id, safe='')}",
                    headers={"x-sglang-omni-route-worker": worker} if worker else {},
                )
                record["outcome_status"] = outcome.status_code
                if outcome.status_code == 200:
                    data = outcome.json()
                    record["finish_reason"] = data.get("finish_reason")
                    record["frames"] = int((data.get("usage") or {}).get("completion_tokens") or 0)
                else:
                    pass
                pcm = b"".join(chunks)
                samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768
                record["duration_ms"] = 1000 * len(samples) / SAMPLE_RATE_HZ
                if not interrupt:
                    record["onset_ms"], record["tail_ms"] = silence_edges_ms(samples)
                    record["audio"] = str(out_dir / "audio" / f"{case['id']}.wav")
                    soundfile.write(record["audio"], samples, SAMPLE_RATE_HZ, subtype="PCM_16")
                else:
                    pass
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        record = await request(client, case)
        progress["done"] += 1
        progress["length"] += int(record.get("finish_reason") == "length")
        progress["errors"] += int("error" in record)
        if progress["done"] % PROGRESS_EVERY == 0 or progress["done"] == len(cases):
            print(
                f"{label} {shape} c{concurrency}{' interrupt' if interrupt else ''}: {progress['done']} of {len(cases)}"
                f" done, {progress['length']} length stops, {progress['errors']} errors",
                flush=True,
            )
        else:
            pass
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in cases))
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )


def score(dirs: list[Path], asr_port: int, sim_card: int | None) -> None:
    asr = load_router_asr(asr_port)
    scorer = None
    if sim_card is not None:
        assets = ensure_speaker_similarity_assets()
        scorer = WavLMSpeakerSimilarity(
            finetune_checkpoint=assets.finetune_checkpoint, wavlm_base=assets.wavlm_base, device=f"cuda:{sim_card}"
        )
    else:
        pass
    for out_dir in dirs:
        records = [
            json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line
        ]
        with_audio = [record for record in records if "audio" in record]

        def transcribe_one(record: dict[str, object]) -> None:
            lang = str(record["lang"])
            hypothesis = transcribe(asr, str(record["audio"]), lang, "cpu")
            measures = process_words(normalize_text(str(record["input"]), lang), normalize_text(hypothesis, lang))
            record["transcript"] = hypothesis
            record["wer"] = measures.wer
            record["word_errors"] = measures.substitutions + measures.deletions + measures.insertions
            record["reference_words"] = measures.substitutions + measures.deletions + measures.hits

        with ThreadPoolExecutor(64) as pool:
            list(pool.map(transcribe_one, with_audio))
        if scorer is not None:
            natural = [record for record in with_audio if record.get("finish_reason") == "stop"]
            for start in range(0, len(natural), SIMILARITY_BATCH):
                batch = natural[start : start + SIMILARITY_BATCH]
                similarities = scorer.score_batch(
                    [str(record["ref_audio"]) for record in batch], [str(record["audio"]) for record in batch]
                )
                for record, similarity in zip(batch, similarities):
                    record["similarity"] = float(similarity)
        else:
            pass
        (out_dir / "scored.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
        )
        print(f"scored {out_dir}: {len(with_audio)} clips", flush=True)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def paired(pairs: list[tuple[float, float]]) -> str:
    if len(pairs) < 2:
        return "-"
    else:
        pass
    deltas = [other - base for base, other in pairs]
    mean = statistics.fmean(deltas)
    half = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return f"{mean:+.2f} +- {half:.2f} (n {len(deltas)})"


def load(out_dir: Path) -> list[dict[str, object]]:
    source = out_dir / "scored.jsonl" if (out_dir / "scored.jsonl").exists() else out_dir / "results.jsonl"
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def arm_row(records: list[dict[str, object]]) -> str:
    errors = [record for record in records if "error" in record]
    finished = [record for record in records if "error" not in record]
    empty = sum(float(record.get("duration_ms", 0)) == 0 for record in finished)
    length = [record for record in finished if record.get("finish_reason") == "length"]
    natural = [record for record in finished if record.get("finish_reason") == "stop"]
    other = len(finished) - len(length) - len(natural)
    ttfa = [float(record["ttfa_ms"]) for record in finished if "ttfa_ms" in record]
    onset = [float(record["onset_ms"]) for record in natural if "onset_ms" in record]
    speech = [float(record["ttfa_ms"]) + float(record["onset_ms"]) for record in natural
              if "ttfa_ms" in record and "onset_ms" in record]
    frames = [int(record["frames"]) for record in natural if "frames" in record]
    scored = [record for record in natural if "word_errors" in record]
    errors_words = sum(int(record["word_errors"]) for record in scored)
    reference_words = sum(int(record["reference_words"]) for record in scored)
    failures = sum(float(record["wer"]) > WER_FAILURE for record in scored)
    similarity = [float(record["similarity"]) for record in natural if "similarity" in record]
    length_s = sum(float(record.get("duration_ms", 0)) for record in length) / 1000
    parts = [
        f"n {len(records)}", f"errors {len(errors)}", f"empty {empty}", f"length {len(length)} ({length_s:.0f} s)",
        f"other finish {other}",
        f"frames p50/p99 {quantile(frames, 0.5)}/{quantile(frames, 0.99)}" if frames else "frames -",
        f"TTFA ms p50/p90/p99 {quantile(ttfa, 0.5):.1f}/{quantile(ttfa, 0.9):.1f}/{quantile(ttfa, 0.99):.1f}"
        if ttfa else "TTFA -",
        f"onset ms p50/p95 {quantile(onset, 0.5):.0f}/{quantile(onset, 0.95):.0f}" if onset else "onset -",
        f"first speech ms p50/p95 {quantile(speech, 0.5):.0f}/{quantile(speech, 0.95):.0f}" if speech else "",
        f"corpus error {100 * errors_words / reference_words:.2f} %, over 50 % {failures}" if reference_words else "",
        f"SIM {statistics.fmean(similarity):.2f}" if similarity else "",
    ]
    return " | ".join(part for part in parts if part)


def compare(dirs: list[Path]) -> None:
    for base_dir, other_dir in zip(dirs[::2], dirs[1::2]):
        base = load(base_dir)
        other = load(other_dir)
        print(f"== {base_dir} against {other_dir}")
        groups: dict[tuple[str, str], list[list[dict[str, object]]]] = defaultdict(lambda: [[], []])
        for index, records in enumerate((base, other)):
            for record in records:
                groups[(str(record["shape"]), str(record["set"]))][index].append(record)
        for (shape, set_name), (base_records, other_records) in sorted(groups.items()):
            print(f"{shape} {set_name}")
            print(f"  {base_records[0]['label'] if base_records else '-'}: {arm_row(base_records)}")
            print(f"  {other_records[0]['label'] if other_records else '-'}: {arm_row(other_records)}")
            by_id = {str(record["id"]): record for record in other_records}
            both = [(record, by_id[str(record["id"])]) for record in base_records if str(record["id"]) in by_id]

            def pairs_of(key: str, natural_only: bool) -> list[tuple[float, float]]:
                return [
                    (float(first[key]), float(second[key])) for first, second in both
                    if key in first and key in second
                    and (not natural_only or (first.get("finish_reason") == "stop" and second.get("finish_reason") == "stop"))
                ]

            print(
                f"  paired (other - base): TTFA ms {paired(pairs_of('ttfa_ms', False))} | onset ms "
                f"{paired(pairs_of('onset_ms', True))} | WER {paired(pairs_of('wer', True))} | SIM "
                f"{paired(pairs_of('similarity', True))}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("out_dir", type=Path)
    cases_parser.add_argument("--profiles", type=Path, required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("cases", type=Path)
    serve_parser.add_argument("--base-url", required=True)
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--label", required=True)
    serve_parser.add_argument("--shape", required=True)
    serve_parser.add_argument("--sets", nargs="*", default=[])
    serve_parser.add_argument("--out", type=Path, required=True)
    serve_parser.add_argument("--concurrency", type=int, default=16)
    serve_parser.add_argument("--interrupt", action="store_true")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("dirs", type=Path, nargs="+")
    score_parser.add_argument("--asr-port", type=int, required=True)
    score_parser.add_argument("--sim-card", type=int)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "cases":
        build_cases(args.out_dir, args.profiles)
    elif args.command == "serve":
        asyncio.run(serve(args.cases, args.base_url, args.model, args.label, args.shape, args.sets, args.out,
                          args.concurrency, args.interrupt))
    elif args.command == "score":
        score(args.dirs, args.asr_port, args.sim_card)
    else:
        compare(args.dirs)


if __name__ == "__main__":
    sys.exit(main())
