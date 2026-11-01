"""Which stage owns each SeedTTS sample's word errors, from one scored pass with reply_text.json.

For every sample: the thinker's reply text (run_bench.py gen records it) against the target text, and the ASR
transcript against the target, both through the scorer's own normalizer (benchmarks.tasks.asr.normalize_text) and
jiwer word measures. A sample whose reply differs from the target is a thinker text departure; a sample whose
reply equals the target but whose transcript does not is a speech departure (talker, code predictor, code2wav or
the scorer). Bad samples (WER above 0.5) are listed with both texts. Prints one summary and the bad rows.

usage: python3 bad_output_census.py en|zh PASS_DIR [PASS_DIR ...]   (each holds wer_results.json, reply_text.json)
"""

import json
import sys
from pathlib import Path

from jiwer import process_words

from benchmarks.tasks.asr import normalize_text

BAD_WER = 0.5


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    if not reference:
        return 0, 0
    else:
        pass
    if not hypothesis:
        return len(reference.split()), len(reference.split())
    else:
        pass
    measures = process_words(reference, hypothesis)
    errors = measures.substitutions + measures.deletions + measures.insertions
    return errors, len(reference.split())


def census(pass_dir: Path, lang: str) -> None:
    wer_rows = json.loads((pass_dir / "wer_results.json").read_text())["per_sample"]
    replies = json.loads((pass_dir / "reply_text.json").read_text())
    totals = {
        "samples": 0,
        "words": 0,
        "speech_errors": 0,
        "reply_departures": 0,
        "reply_errors": 0,
    }
    owners = {"thinker text": [0, 0], "speech": [0, 0]}
    bad_rows = []
    for row in wer_rows:
        reference = row["ref_norm"]
        reply = normalize_text(replies.get(row["id"], ""), lang)
        speech_errors = row["substitutions"] + row["deletions"] + row["insertions"]
        reply_errors, words = word_errors(reference, reply)
        totals["samples"] += 1
        totals["words"] += words
        totals["speech_errors"] += speech_errors
        totals["reply_errors"] += reply_errors
        owner = "thinker text" if reply_errors > 0 else "speech"
        if reply_errors > 0:
            totals["reply_departures"] += 1
        else:
            pass
        owners[owner][0] += speech_errors
        if row["wer"] > BAD_WER:
            owners[owner][1] += 1
            bad_rows.append(
                (row["id"], row["wer"], owner, reference, reply, row["hyp_norm"])
            )
        else:
            pass
    words = max(totals["words"], 1)
    print(
        f"== {pass_dir}: {totals['samples']} samples, WER {totals['speech_errors'] / words * 100:.3f} %, "
        f"reply departs from the target in {totals['reply_departures']} samples "
        f"(reply WER {totals['reply_errors'] / words * 100:.3f} %)"
    )
    for owner, (errors, bad) in owners.items():
        print(
            f"   {owner}: word errors {errors} ({errors / words * 100:.3f} %), bad samples {bad}"
        )
    for sample_id, wer, owner, reference, reply, transcript in bad_rows:
        print(f"   BAD {sample_id} wer {wer:.2f} owner {owner}")
        print(f"      target: {reference}")
        print(f"      reply:  {reply}")
        print(f"      heard:  {transcript[:300]}")


def main() -> None:
    lang = sys.argv[1]
    for pass_dir in sys.argv[2:]:
        census(Path(pass_dir), lang)


if __name__ == "__main__":
    main()
