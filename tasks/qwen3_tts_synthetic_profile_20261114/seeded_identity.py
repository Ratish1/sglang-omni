"""Seeded CustomVoice requests, one at a time, for byte identity between trees.

Per tree, one 1.7B CustomVoice server on card N. After one warm-up request, every input is sent seeded, Ryan,
PCM, streamed and whole, once without a language (the SeedTTS client's form, never eligible for the trim) and
once with English (eligible). Each request's audio hash is written to OUT/<label>.jsonl. With --compare, the
hashes of two labels are matched by request position and every mismatch is printed.

usage: python3 seeded_identity.py --tree TREE --label NAME --out DIR [--card 0]
       python3 seeded_identity.py --compare A B --out DIR
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy
from slot_wait_serving import INPUTS

SEED = 1234
REQUEST_TIMEOUT_S = 300
LANGUAGES = (None, "English")


def request_audio(url: str, text: str, language: str | None, stream: bool) -> dict:
    body = {
        "model": MODELS["customvoice"][0], "input": text, "voice": "Ryan", "response_format": "pcm",
        "stream": stream, "seed": SEED,
    }
    if language is not None:
        body["language"] = language
    else:
        pass
    audio = bytearray()
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
            if response.status_code != 200:
                return {"status": response.status_code}
            else:
                pass
            for piece in response.iter_bytes():
                audio.extend(piece)
    return {"status": 200, "samples": len(audio) // 2, "sha256": hashlib.sha256(bytes(audio)).hexdigest()[:16]}


def run_tree(tree: Path, label: str, card: int, port: int, out: Path) -> None:
    url = f"http://127.0.0.1:{port}"
    server = serve(tree, MODELS["customvoice"][0], card, port, out / f"{label}_serve.log", {})
    try:
        wait_healthy(port, server)
        request_audio(url, INPUTS[0], "English", True)
        with (out / f"{label}.jsonl").open("w") as handle:
            for language in LANGUAGES:
                for stream in (True, False):
                    for index, text in enumerate(INPUTS):
                        row = {"language": language, "stream": stream, "input": index, **request_audio(url, text, language, stream)}
                        handle.write(json.dumps(row) + "\n")
                        print(f"{time.strftime('%T')} {label} {json.dumps(row)}", flush=True)
    finally:
        stop(server)


def compare(first: str, second: str, out: Path) -> None:
    rows = {
        label: [json.loads(line) for line in (out / f"{label}.jsonl").read_text().splitlines()]
        for label in (first, second)
    }
    mismatches = [
        (a, b) for a, b in zip(rows[first], rows[second]) if a.get("sha256") != b.get("sha256") or a["status"] != 200
    ]
    for a, b in mismatches:
        print(f"MISMATCH {json.dumps(a)} against {json.dumps(b)}", flush=True)
    print(
        f"{time.strftime('%T')} seeded identity {first} against {second}: "
        f"{len(rows[first]) - len(mismatches)} of {len(rows[first])} byte-identical",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, default=29000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare is not None:
        compare(args.compare[0], args.compare[1], args.out)
    else:
        run_tree(args.tree, args.label, args.card, args.port, args.out)


if __name__ == "__main__":
    main()
