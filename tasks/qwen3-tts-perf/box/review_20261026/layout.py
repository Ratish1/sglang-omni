"""Qwen3-TTS Base clones per server configuration on the cascade's requests (language English) and the SeedTTS client's
(no language, so auto): runaways, empty audio, opening and closing silence, WER and speaker similarity, against
official qwen-tts in the same prompt layout and language.

usage: python3 layout.py serve CASES.jsonl --base-url URL --model PATH --label L --language {auto,English} --out DIR
          [--modes icl xvec] [--max-new-tokens 300] [--concurrency 16]
       python3 layout.py score DIR [DIR ...] [--asr-port PORT] [--sim-card CARD]
       python3 layout.py summary OFFICIAL.json [OFFICIAL.json ...] --served DIR [DIR ...] [--max-new-tokens 300]
       python3 layout.py compare BASE_DIR DIR [DIR ...] [--max-new-tokens 300]

serve posts every case as a streaming PCM request (voice default, the reference as a server-side path with its text in
ICL mode, no seed, max_new_tokens the shared cap, or none at --max-new-tokens 0), keeps the audio as DIR/audio/<id>.wav, and takes the finish reason
and frames from the stream's outcome. Opening silence is the time to the first 5 ms window at -34 dBFS or louder
(#2357's onset), closing silence the time after the last such window. score transcribes every clip with the Qwen3-ASR
server on PORT (WER against the target after the Whisper English normalizer) and, with --sim-card, scores every
natural stop's WavLM speaker similarity against its own reference clip. summary prints one row per checkpoint,
configuration, language and set, official first; an official row without EOS returns one frame fewer than the cap.
compare pairs every arm with a base arm case by case.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import soundfile
from jiwer import process_words
from scipy.stats import binomtest, fisher_exact

from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import ensure_speaker_similarity_assets
from benchmarks.tasks.asr import load_router_asr, normalize_text, transcribe

REQUEST_TIMEOUT_S = 900.0
PROGRESS_EVERY = 1000
SAMPLE_RATE_HZ = 24000
ONSET_WINDOW_SAMPLES = 120
ONSET_FLOOR_DBFS = -34.0
SIMILARITY_BATCH = 8
WER_FAILURE = 0.5


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


async def serve(cases_path: Path, base_url: str, model: str, label: str, language: str, out_dir: Path,
                modes: list[str], max_new_tokens: int, concurrency: int) -> None:
    cases = [
        case for case in (json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line)
        if case["mode"] in modes
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
        body: dict[str, object] = {
            "model": model, "voice": "default", "input": case["input"], "response_format": "pcm", "stream": True,
            "references": [reference],
        }
        if max_new_tokens > 0:
            body["max_new_tokens"] = max_new_tokens
        else:
            pass
        if language != "auto":
            body["language"] = language
        else:
            pass
        record: dict[str, object] = {
            "id": case["id"], "set": case["set"], "mode": case["mode"], "label": label, "language": language,
            "input": case["input"], "ref_audio": case["ref_audio"],
        }
        async with limit:
            try:
                async with client.stream("POST", f"{base_url}/v1/audio/speech", json=body) as response:
                    if response.status_code != 200:
                        await response.aread()
                        record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                        return record
                    else:
                        pass
                    speech_id = response.headers["X-SGLang-Omni-Speech-Id"]
                    worker = response.headers.get("X-SGLang-Omni-Worker")
                    pcm = b"".join([chunk async for chunk in response.aiter_bytes()])
                outcome = await client.get(
                    f"{base_url}/v1/audio/speech/{quote(speech_id, safe='')}",
                    headers={"x-sglang-omni-route-worker": worker} if worker else {},
                )
                outcome.raise_for_status()
                data = outcome.json()
                record["finish_reason"] = data.get("finish_reason")
                record["frames"] = int((data.get("usage") or {}).get("completion_tokens"))
                samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768
                record["duration_ms"] = 1000 * len(samples) / SAMPLE_RATE_HZ
                record["onset_ms"], record["tail_ms"] = silence_edges_ms(samples)
                record["audio"] = str(out_dir / "audio" / f"{case['id']}.wav")
                soundfile.write(record["audio"], samples, SAMPLE_RATE_HZ, subtype="PCM_16")
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async def one(client: httpx.AsyncClient, case: dict[str, str]) -> dict[str, object]:
        record = await request(client, case)
        progress["done"] += 1
        progress["runaways"] += int(record.get("finish_reason") == "length")
        progress["errors"] += int("error" in record)
        if progress["done"] % PROGRESS_EVERY == 0 or progress["done"] == len(cases):
            print(
                f"{label} {language}: {progress['done']} of {len(cases)} done, {progress['runaways']} runaways, "
                f"{progress['errors']} errors",
                flush=True,
            )
        else:
            pass
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, case) for case in cases))
    (out_dir / "results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def score(dirs: list[Path], asr_port: int | None, sim_card: int | None) -> None:
    for out_dir in dirs:
        source = out_dir / "scored.jsonl" if (out_dir / "scored.jsonl").exists() else out_dir / "results.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        with_audio = [record for record in records if "audio" in record]
        if asr_port is not None:
            asr = load_router_asr(asr_port)

            def transcribe_one(record: dict[str, object]) -> None:
                hypothesis = transcribe(asr, str(record["audio"]), "en", "cpu")
                measures = process_words(normalize_text(str(record["input"]), "en"), normalize_text(hypothesis, "en"))
                record["transcript"] = hypothesis
                record["wer"] = measures.wer
                record["word_errors"] = measures.substitutions + measures.deletions + measures.insertions
                record["reference_words"] = measures.substitutions + measures.deletions + measures.hits

            with ThreadPoolExecutor(64) as pool:
                list(pool.map(transcribe_one, with_audio))
        else:
            pass
        if sim_card is not None:
            assets = ensure_speaker_similarity_assets()
            scorer = WavLMSpeakerSimilarity(
                finetune_checkpoint=assets.finetune_checkpoint, wavlm_base=assets.wavlm_base,
                device=f"cuda:{sim_card}",
            )
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
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        print(f"scored {out_dir.name}: {len(with_audio)} clips", flush=True)


def summary(official_paths: list[Path], served_dirs: list[Path], max_new_tokens: int) -> None:
    rows: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for path in official_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        size = "06" if "0.6B" in data["model"] else "17"
        layout = "whole" if data.get("whole_text") else "stream"
        for record in data["records"]:
            rows[(size, data.get("language", "English"), record["set"], f"official_{layout}")].append(
                {**record, "runaway": record["frames"] >= max_new_tokens - 1}
            )
    for out_dir in served_dirs:
        source = out_dir / "scored.jsonl" if (out_dir / "scored.jsonl").exists() else out_dir / "results.jsonl"
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            size, config = str(record["label"]).split("_", 1)
            record["runaway"] = record.get("frames", 0) >= max_new_tokens
            rows[(size, str(record["language"]), str(record["set"]), config)].append(record)
    print(
        "checkpoint | language | set | arm | n | runaways | Fisher p vs official same layout | empty | natural frames "
        "p50/p99 | onset ms p50/p95, share > 160 ms | closing silence ms p50/p95 | WER of natural stops "
        "(> 50 %) | SIM"
    )
    keys = sorted({(size, language, set_name) for size, language, set_name, _ in rows})
    for size, language, set_name in keys:
        arms = sorted(arm for key_size, key_language, key_set, arm in rows if (key_size, key_language, key_set) == (size, language, set_name))
        for arm in sorted(arms, key=lambda name: (not name.startswith("official"), name)):
            records = rows[(size, language, set_name, arm)]
            n = len(records)
            runaways = sum(bool(record["runaway"]) for record in records)
            empty = sum("error" in record for record in records)
            layout = "whole" if "whole" in arm else "stream"
            official = rows.get((size, language, set_name, f"official_{layout}"), [])
            if official and not arm.startswith("official"):
                official_runaways = sum(bool(record["runaway"]) for record in official)
                fisher = fisher_exact([[runaways, n - runaways], [official_runaways, len(official) - official_runaways]])[1]
                fisher_text = f"{fisher:.3f}"
            else:
                fisher_text = "-"
            natural = sorted(int(record["frames"]) for record in records if "frames" in record and not record["runaway"])
            onsets = sorted(float(record["onset_ms"]) for record in records if "onset_ms" in record and not record["runaway"])
            tails = sorted(float(record["tail_ms"]) for record in records if "tail_ms" in record and not record["runaway"])
            wers = [float(record["wer"]) for record in records if "wer" in record and not record["runaway"]]
            sims = [float(record["similarity"]) for record in records if "similarity" in record]

            def quantile(values: list[float], fraction: float) -> float:
                return values[min(len(values) - 1, int(len(values) * fraction))]

            print(
                f"{size} | {language} | {set_name} | {arm} | {n} | {runaways} ({100 * runaways / max(n, 1):.2f} %) | "
                f"{fisher_text} | {empty} | "
                + (f"{quantile(natural, 0.5)}/{quantile(natural, 0.99)}" if natural else "-") + " | "
                + (f"{quantile(onsets, 0.5):.0f}/{quantile(onsets, 0.95):.0f}, "
                   f"{100 * sum(onset > 160 for onset in onsets) / len(onsets):.1f} %" if onsets else "-") + " | "
                + (f"{quantile(tails, 0.5):.0f}/{quantile(tails, 0.95):.0f}" if tails else "-") + " | "
                + (f"{100 * sum(wers) / len(wers):.2f} % ({sum(wer > WER_FAILURE for wer in wers)})" if wers else "-")
                + " | " + (f"{sum(sims) / len(sims):.4f}" if sims else "-")
            )


def compare(base_dir: Path, other_dirs: list[Path], max_new_tokens: int) -> None:
    """Paired per case against base_dir: runaways (discordant pairs, exact binomial), empties, and over cases that stop
    naturally in both arms the onset, WER, failures and similarity, each with its paired mean difference and 95 % CI."""

    def load(out_dir: Path) -> dict[str, dict[str, object]]:
        source = out_dir / "scored.jsonl" if (out_dir / "scored.jsonl").exists() else out_dir / "results.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        return {str(record["id"]): record for record in records}

    def paired(values: list[tuple[float, float]]) -> str:
        if len(values) < 2:
            return "-"
        else:
            pass
        deltas = np.array([other - base for base, other in values])
        half = 1.96 * deltas.std(ddof=1) / np.sqrt(len(deltas))
        return (f"{np.mean([base for base, _ in values]):.4f} -> {np.mean([other for _, other in values]):.4f} "
                f"(d {deltas.mean():+.4f} +- {half:.4f})")

    base = load(base_dir)
    for other_dir in other_dirs:
        other = load(other_dir)
        print(f"== {base_dir.name} -> {other_dir.name}")
        for set_name in sorted({str(record["set"]) for record in base.values()}):
            ids = [case_id for case_id, record in base.items() if record["set"] == set_name and case_id in other]
            if not ids:
                continue
            else:
                pass

            def runaway(record: dict[str, object]) -> bool:
                return int(record.get("frames", 0)) >= max_new_tokens

            def natural(record: dict[str, object]) -> bool:
                return "error" not in record and not runaway(record)

            only_base = sum(runaway(base[i]) and not runaway(other[i]) for i in ids)
            only_other = sum(runaway(other[i]) and not runaway(base[i]) for i in ids)
            discordant = only_base + only_other
            p_value = binomtest(only_other, discordant).pvalue if discordant else 1.0
            both = [i for i in ids if natural(base[i]) and natural(other[i])]
            onsets = [(float(base[i]["onset_ms"]), float(other[i]["onset_ms"])) for i in both]
            wers = [(float(base[i]["wer"]), float(other[i]["wer"])) for i in both if "wer" in base[i] and "wer" in other[i]]
            failures = [(float(b > WER_FAILURE), float(o > WER_FAILURE)) for b, o in wers]
            sims = [
                (float(base[i]["similarity"]), float(other[i]["similarity"]))
                for i in both if "similarity" in base[i] and "similarity" in other[i]
            ]
            print(
                f"{set_name} | n {len(ids)} | runaways {sum(runaway(base[i]) for i in ids)} -> "
                f"{sum(runaway(other[i]) for i in ids)} (only base {only_base}, only other {only_other}, p {p_value:.3f}) | "
                f"empty {sum('error' in base[i] for i in ids)} -> {sum('error' in other[i] for i in ids)} | "
                f"both natural {len(both)} | onset ms {paired(onsets)} | WER {paired(wers)} | "
                f"failures {paired(failures)} | SIM {paired(sims)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("cases", type=Path)
    serve_parser.add_argument("--base-url", required=True)
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--label", required=True)
    serve_parser.add_argument("--language", choices=("auto", "English"), required=True)
    serve_parser.add_argument("--out", type=Path, required=True)
    serve_parser.add_argument("--modes", nargs="+", default=["icl", "xvec"])
    serve_parser.add_argument("--max-new-tokens", type=int, default=300)
    serve_parser.add_argument("--concurrency", type=int, default=16)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("dirs", type=Path, nargs="+")
    score_parser.add_argument("--asr-port", type=int)
    score_parser.add_argument("--sim-card", type=int)
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("official", type=Path, nargs="+")
    summary_parser.add_argument("--served", type=Path, nargs="+", required=True)
    summary_parser.add_argument("--max-new-tokens", type=int, default=300)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("others", type=Path, nargs="+")
    compare_parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(serve(args.cases, args.base_url, args.model, args.label, args.language, args.out, args.modes,
                          args.max_new_tokens, args.concurrency))
    elif args.command == "score":
        score(args.dirs, args.asr_port, args.sim_card)
    elif args.command == "summary":
        summary(args.official, args.served, args.max_new_tokens)
    else:
        compare(args.base, args.others, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())
