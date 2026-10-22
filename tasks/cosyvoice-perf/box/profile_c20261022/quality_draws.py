"""SeedTTS quality draws of scored pairs, pooled by cell family (c1f_s7 counts as c1f): per draw (pair, order, cell)
both arms' corpus WER and SIM and the second arm minus the first; per family the arms' means, the mean delta, its
standard deviation, standard error and t, and the draws worse and better. Then the vocoder route effect of each arm:
for every order that ran a streaming and a buffered seeded cell under one seed, the per clip WER and SIM of the
buffered audio minus the streaming audio over the clips whose two audios share a length (the same tokens), and the
second arm's route effect minus the first's. A seeded cell that ran under one seed in both orders is one draw, the
two orders averaged: the chain repeats its seeded audio exactly across boots.
usage: python3 quality_draws.py <run home> <pair> [<pair> ...]"""

import json
import statistics
import sys
import wave
from pathlib import Path

SEEDED = ("c1", "c1f", "b1f")


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def sim_rows(data):
    rows = data["per_sample"]
    key = next(key for key in rows[0] if "sim" in key.lower() and key != "id")
    return {row["id"]: float(row[key]) for row in rows if row.get(key) is not None}


def wer_rows(data):
    return {row["id"]: row["wer"] for row in data["per_sample"] if row.get("wer") is not None}


def frames(cell):
    lengths = {}
    for path in cell.rglob("*.wav"):
        with wave.open(str(path)) as audio:
            lengths[path.stem] = audio.getnframes()
    return lengths


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / len(values) ** 0.5
    return mean, sd, se, (mean / se if se else 0.0)


def main():
    home = Path(sys.argv[1])
    draws = {}
    routes = []
    print("| pair | order | cell | first WER | second WER | delta | first SIM | second SIM | delta |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sys.argv[2:]:
        for order in ("a", "b"):
            gpus = {"first": "gpu0", "second": "gpu1"} if order == "a" else {"first": "gpu1", "second": "gpu0"}
            cells = sorted(path.name for path in (home / name / order / gpus["first"]).iterdir() if path.is_dir())
            scored = {}
            for cell in cells:
                arms = {}
                for arm, gpu in gpus.items():
                    directory = home / name / order / gpu / cell
                    wer, sim = load(directory / "wer_results.json"), load(directory / "similarity_results.json")
                    if wer and sim:
                        arms[arm] = (directory, wer, sim)
                if len(arms) < 2:
                    continue
                scored[cell] = arms
                first_wer, second_wer = (100 * arms[arm][1]["summary"]["wer_corpus"] for arm in ("first", "second"))
                first_sim, second_sim = (arms[arm][2]["summary"]["speaker_similarity_mean"] for arm in ("first", "second"))
                family = cell.partition("_s")[0]
                key = (name, cell) if family in SEEDED else (name, order, cell)
                draws.setdefault(family, {}).setdefault(key, []).append((first_wer, second_wer, first_sim, second_sim))
                print(
                    f"| {name} | {order} | {cell} | {first_wer:.3f} | {second_wer:.3f} | {second_wer - first_wer:+.3f} | "
                    f"{first_sim:.2f} | {second_sim:.2f} | {second_sim - first_sim:+.2f} |"
                )
            for cell in scored:
                family, _, seed = cell.partition("_s")
                buffered = f"b1f_s{seed}" if seed else "b1f"
                if family != "c1f" or buffered not in scored:
                    continue
                effect = {}
                for arm in ("first", "second"):
                    stream_dir, stream_wer, stream_sim = scored[cell][arm]
                    buffer_dir, buffer_wer, buffer_sim = scored[buffered][arm]
                    stream_frames, buffer_frames = frames(stream_dir), frames(buffer_dir)
                    wer_s, wer_b = wer_rows(stream_wer), wer_rows(buffer_wer)
                    sim_s, sim_b = sim_rows(stream_sim), sim_rows(buffer_sim)
                    ids = [
                        clip for clip in wer_s.keys() & wer_b.keys() & sim_s.keys() & sim_b.keys()
                        if stream_frames.get(clip) == buffer_frames.get(clip)
                    ]
                    effect[arm] = (
                        len(ids),
                        100 * statistics.mean(wer_b[clip] - wer_s[clip] for clip in ids),
                        statistics.mean(sim_b[clip] - sim_s[clip] for clip in ids),
                    )
                routes.append((name, order, seed or "1234", effect))
    route_draws = {}
    for name, _, seed, effect in routes:
        route_draws.setdefault((name, seed), []).append(
            (effect["second"][1] - effect["first"][1], effect["second"][2] - effect["first"][2])
        )
    print()
    print("Per family, second arm minus first over every draw (WER in points):")
    print("| family | draws | first WER | second WER | delta mean | sd | se | t | worse, better | first SIM | second SIM | delta mean | sd | se | t | worse, better |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for family, keyed in sorted(draws.items()):
        rows = [tuple(statistics.mean(values) for values in zip(*group)) for group in keyed.values()]
        wer_deltas = [second - first for first, second, _, _ in rows]
        sim_deltas = [second - first for _, _, first, second in rows]
        wm, wsd, wse, wt = stats(wer_deltas)
        sm, ssd, sse, st = stats(sim_deltas)
        print(
            f"| {family} | {len(rows)} | {statistics.mean(r[0] for r in rows):.3f} | {statistics.mean(r[1] for r in rows):.3f} | "
            f"{wm:+.3f} | {wsd:.3f} | {wse:.3f} | {wt:+.2f} | {sum(d > 0 for d in wer_deltas)}, {sum(d < 0 for d in wer_deltas)} | "
            f"{statistics.mean(r[2] for r in rows):.2f} | {statistics.mean(r[3] for r in rows):.2f} | "
            f"{sm:+.3f} | {ssd:.3f} | {sse:.3f} | {st:+.2f} | {sum(d < 0 for d in sim_deltas)}, {sum(d > 0 for d in sim_deltas)} |"
        )
    if routes:
        print()
        print("Vocoder route effect, buffered minus streaming audio of the same tokens (clips, WER points, SIM):")
        print("| pair | order | seed | first: clips, WER, SIM | second: clips, WER, SIM | second minus first: WER, SIM |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name, order, seed, effect in routes:
            first, second = effect["first"], effect["second"]
            print(
                f"| {name} | {order} | {seed} | {first[0]}, {first[1]:+.3f}, {first[2]:+.3f} | "
                f"{second[0]}, {second[1]:+.3f}, {second[2]:+.3f} | {second[1] - first[1]:+.3f}, {second[2] - first[2]:+.3f} |"
            )
        per_draw = [tuple(statistics.mean(values) for values in zip(*group)) for group in route_draws.values()]
        wm, wsd, wse, wt = stats([wer for wer, _ in per_draw])
        sm, ssd, sse, st = stats([sim for _, sim in per_draw])
        print(f"route effect, second minus first over {len(per_draw)} draws: WER {wm:+.3f} (se {wse:.3f}, t {wt:+.2f}), SIM {sm:+.3f} (se {sse:.3f}, t {st:+.2f})")


if __name__ == "__main__":
    main()
