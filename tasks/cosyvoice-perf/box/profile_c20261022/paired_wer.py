"""Per clip WER of two scored cells (wer_results.json), over the clips whose audio has the same length in both (the
same tokens under one seed, so only the vocoder route or its numerics differ): mean WER delta and its standard error,
clips worse and better, the clips above 50 % in either, and the largest deltas with their ids and transcripts.
usage: python3 paired_wer.py <cell dir A> <cell dir B>"""

import json
import os
import statistics
import sys
import wave


def per_clip(cell: str) -> dict[str, dict]:
    with open(os.path.join(cell, "wer_results.json")) as handle:
        return {row["id"]: row for row in json.load(handle)["per_sample"] if row.get("wer") is not None}


def frames(cell: str) -> dict[str, int]:
    lengths = {}
    for root, _, files in os.walk(cell):
        for name in files:
            if name.endswith(".wav"):
                with wave.open(os.path.join(root, name)) as wav:
                    lengths[name[: -len(".wav")]] = wav.getnframes()
    return lengths


def main() -> None:
    cell_a, cell_b = sys.argv[1], sys.argv[2]
    rows_a, rows_b = per_clip(cell_a), per_clip(cell_b)
    frames_a, frames_b = frames(cell_a), frames(cell_b)
    ids = sorted(clip for clip in set(rows_a) & set(rows_b) if frames_a.get(clip) == frames_b.get(clip))
    deltas = [(rows_b[clip]["wer"] - rows_a[clip]["wer"], clip) for clip in ids]
    values = [delta for delta, _ in deltas]
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    print(
        f"same length clips {len(ids)} of {len(set(rows_a) & set(rows_b))}: mean WER delta {100 * statistics.mean(values):+.3f} "
        f"points (se {100 * error:.3f}), worse {sum(value > 0 for value in values)}, better {sum(value < 0 for value in values)}, "
        f"equal {sum(value == 0 for value in values)}"
    )
    above = [clip for clip in ids if rows_a[clip]["wer"] > 0.5 or rows_b[clip]["wer"] > 0.5]
    print(f"above 50 % in either: {len(above)} {above[:8]}")
    for delta, clip in sorted(deltas, key=lambda item: -abs(item[0]))[:8]:
        print(
            f"  {delta * 100:+7.1f} {clip}: A {rows_a[clip]['wer'] * 100:.1f} % '{rows_a[clip].get('hyp_norm', '')[:70]}' | "
            f"B {rows_b[clip]['wer'] * 100:.1f} % '{rows_b[clip].get('hyp_norm', '')[:70]}'"
        )


if __name__ == "__main__":
    main()
