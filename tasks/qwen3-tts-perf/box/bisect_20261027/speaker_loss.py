"""A voice clone that loses its speaker: chosen parity cases drawn many times as the SeedTTS CI client sends them
(non-streaming wav, the reference as `references` with its transcript, no language, no seed, no max_new_tokens), each
draw's audio kept with its request id, finish reason and frames; score adds the WavLM speaker similarity the CI gate
uses to every draw of any number of such directories (ours, official) and prints per case its distribution and the
draws below 50.

usage: python3 speaker_loss.py serve CASES.jsonl --ids ID [ID ...] --base-url URL --model PATH --out DIR
          [--draws 100] [--concurrency 16]
       python3 speaker_loss.py score DIR [DIR ...] [--card 0]
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import ensure_speaker_similarity_assets

REQUEST_TIMEOUT_S = 900.0
COLLAPSED = 50.0
SIMILARITY_BATCH = 8


async def serve(cases_path: Path, ids: list[str], base_url: str, model: str, out_dir: Path, draws: int,
                concurrency: int) -> None:
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line)
    }
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, case: dict[str, str], draw: int) -> dict[str, object]:
        body = {
            "model": model, "input": case["input"], "response_format": "wav",
            "references": [{"audio_path": case["ref_audio"], "text": case["ref_text"]}],
        }
        record: dict[str, object] = {
            "id": case["id"], "draw": draw, "input": case["input"], "ref_audio": case["ref_audio"],
        }
        async with limit:
            try:
                response = await client.post(f"{base_url}/v1/audio/speech", json=body)
                if response.status_code != 200:
                    record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                    return record
                else:
                    pass
                record["request_id"] = response.headers.get("X-Request-Id")
                record["finish_reason"] = response.headers.get("X-Finish-Reason")
                record["frames"] = int(response.headers.get("X-Completion-Tokens", "-1"))
                record["audio"] = str(out_dir / "audio" / f"{case['id']}_{draw:04d}.wav")
                Path(str(record["audio"])).write_bytes(response.content)
            except httpx.HTTPError as error:
                record["error"] = f"{type(error).__name__}: {error}"
        return record

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
        records = await asyncio.gather(*(one(client, cases[case_id], draw) for case_id in ids for draw in range(draws)))
    (out_dir / "results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(f"served {len(records)} draws, {sum('error' in record for record in records)} errors", flush=True)


def score(dirs: list[Path], card: int) -> None:
    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint, wavlm_base=assets.wavlm_base, device=f"cuda:{card}"
    )
    for out_dir in dirs:
        records = [
            json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line
        ]
        with_audio = [record for record in records if "audio" in record]
        for start in range(0, len(with_audio), SIMILARITY_BATCH):
            batch = with_audio[start : start + SIMILARITY_BATCH]
            similarities = scorer.score_batch(
                [str(record["ref_audio"]) for record in batch], [str(record["audio"]) for record in batch]
            )
            for record, similarity in zip(batch, similarities):
                record["similarity"] = float(similarity)
        (out_dir / "scored.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in with_audio:
            by_case[str(record["id"])].append(record)
        print(f"{out_dir}: case | draws | SIM q10 / q50 / q90 | min | below {COLLAPSED:.0f} | finish reasons | frames q50")
        for case_id, draws in sorted(by_case.items()):
            values = np.array([float(draw["similarity"]) for draw in draws])
            reasons: dict[str, int] = defaultdict(int)
            for draw in draws:
                reasons[str(draw.get("finish_reason"))] += 1
            low = sorted((float(draw["similarity"]), int(draw["draw"])) for draw in draws if draw["similarity"] < COLLAPSED)
            print(
                f"{case_id} | {len(draws)} | {' / '.join(f'{value:.1f}' for value in np.quantile(values, (0.1, 0.5, 0.9)))} | "
                f"{values.min():.1f} | {len(low)} | {dict(reasons)} | "
                f"{np.median([int(draw.get('frames', -1)) for draw in draws]):.0f} | low draws {low[:12]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("cases", type=Path)
    serve_parser.add_argument("--ids", nargs="+", required=True)
    serve_parser.add_argument("--base-url", required=True)
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--out", type=Path, required=True)
    serve_parser.add_argument("--draws", type=int, default=100)
    serve_parser.add_argument("--concurrency", type=int, default=16)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("dirs", type=Path, nargs="+")
    score_parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(serve(args.cases, args.ids, args.base_url, args.model, args.out, args.draws, args.concurrency))
    else:
        score(args.dirs, args.card)


if __name__ == "__main__":
    main()
