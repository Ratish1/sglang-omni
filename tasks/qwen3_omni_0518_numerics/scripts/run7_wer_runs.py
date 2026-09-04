import json
import os

S = "/private/tmp/claude-501/-Users-ratish-sglang-omni/755476ed-3ef4-4375-b034-a20d476b4162/scratchpad"
RUNS = {
    "run5": (
        f"{S}/run5/qwen3tts-cudnn-validation-20260903/ab",
        "{arm}_qwen3tts_c{c}_full",
    ),
    "run6": (
        f"{S}/run6/qwen3tts-predictor-capture-validation-20260903-final-961e608f5",
        "{arm}_c{c}",
    ),
    "run7": (
        f"{S}/run7/qwen3tts-startup-capture-validation-b6fdc94d7/ab-seed1234",
        "{arm}_c{c}",
    ),
}


def load(run, arm, c):
    base, pat = RUNS[run]
    d = os.path.join(base, pat.format(arm=arm, c=c))
    wer = json.load(open(d + "/wer_results.json"))
    speed = json.load(open(d + "/speed_results.json"))
    sim = None
    p = d + "/similarity_results.json"
    if os.path.exists(p):
        sim = json.load(open(p))
    return wer, speed, sim


def errs(s):
    return s["substitutions"] + s["deletions"] + s["insertions"]


def refw(s):
    return s["hits"] + s["substitutions"] + s["deletions"]


def sim_mean(sim):
    if sim is None:
        return None
    if isinstance(sim, dict):
        for k in ("summary",):
            if k in sim:
                for kk, v in sim[k].items():
                    if "mean" in kk or kk.startswith("sim"):
                        return (kk, v)
        return (list(sim.keys())[:5], None)
    return None


print("== corpus table")
data = {}
for run in RUNS:
    for arm in "AB":
        for c in (1, 16):
            try:
                wer, speed, sim = load(run, arm, c)
            except FileNotFoundError as e:
                print(run, arm, c, "missing", e)
                continue
            ps = wer["per_sample"]
            E = sum(errs(s) for s in ps)
            Wd = sum(refw(s) for s in ps)
            data[(run, arm, c)] = (wer, speed, sim)
            print(
                f"{run} {arm} c{c}: wer_corpus={wer['summary']['wer_corpus']*100:.5f}% errors={E} refwords={Wd} n_wrong_samples={sum(1 for s in ps if errs(s)>0)} sim={sim_mean(sim)} lat_mean={speed['summary'].get('latency_mean_s') or speed['summary'].get('mean_latency_s')}"
            )


def bymap(wer):
    return {s["id"]: s for s in wer["per_sample"]}


def pair(name, x, y, top=8):
    a, b = bymap(x), bymap(y)
    ids = [i for i in a if i in b]
    d = [(errs(b[i]) - errs(a[i]), i) for i in ids]
    diff = [t for t in d if t[0] != 0]
    net = sum(t[0] for t in diff)
    worse = sum(1 for t in diff if t[0] > 0)
    better = sum(1 for t in diff if t[0] < 0)
    print(
        f"\n== {name}: samples differing={len(diff)} of {len(ids)}, second worse in {worse}, better in {better}, net errors {net:+d}, sum|delta|={sum(abs(t[0]) for t in diff)}"
    )
    ident = sum(1 for i in ids if a[i]["hyp_norm"] == b[i]["hyp_norm"])
    print(f"   identical hypotheses: {ident} of {len(ids)}")
    for delta, i in sorted(diff, key=lambda t: -abs(t[0]))[:top]:
        print(
            f"   {delta:+d} {i}\n      ref: {a[i]['ref_norm']}\n      1st: {a[i]['hyp_norm']}\n      2nd: {b[i]['hyp_norm']}"
        )
    return diff


W = lambda k: data[k][0]
pair("run7 A c16 vs B c16", W(("run7", "A", 16)), W(("run7", "B", 16)))
pair(
    "run6 A c16 vs run7 A c16 (main vs main, fresh boots)",
    W(("run6", "A", 16)),
    W(("run7", "A", 16)),
)
pair(
    "run5 A c16 vs run6 A c16 (main vs main)",
    W(("run5", "A", 16)),
    W(("run6", "A", 16)),
)
pair("run6 A c16 vs run6 B c16", W(("run6", "A", 16)), W(("run6", "B", 16)), top=3)
pair("run6 B c16 vs run7 B c16", W(("run6", "B", 16)), W(("run7", "B", 16)), top=3)
pair("run7 A c1 vs run7 B c1", W(("run7", "A", 1)), W(("run7", "B", 1)), top=3)
pair("run6 A c1 vs run7 A c1", W(("run6", "A", 1)), W(("run7", "A", 1)), top=3)
pair("run5 A c1 vs run7 A c1", W(("run5", "A", 1)), W(("run7", "A", 1)), top=3)
pair(
    "run7 A c1 vs run7 A c16 (same code, c1 vs c16)",
    W(("run7", "A", 1)),
    W(("run7", "A", 16)),
    top=3,
)
print(
    "\n== c16 corpus WER of arms that replay identical kernels (all mains, run6 B, run7 B):"
)
for k in [
    ("run5", "A", 16),
    ("run6", "A", 16),
    ("run6", "B", 16),
    ("run7", "A", 16),
    ("run7", "B", 16),
]:
    ps = W(k)["per_sample"]
    print(
        "  ",
        k,
        f"{W(k)['summary']['wer_corpus']*100:.5f}%",
        "errors",
        sum(errs(s) for s in ps),
    )

print("\n== flip overlap at c16")


def diffset(x, y):
    a, b = bymap(x), bymap(y)
    return {i for i in a if i in b and errs(a[i]) != errs(b[i])}


ab7 = diffset(W(("run7", "A", 16)), W(("run7", "B", 16)))
main_pairs = [
    diffset(W(("run5", "A", 16)), W(("run6", "A", 16))),
    diffset(W(("run6", "A", 16)), W(("run7", "A", 16))),
    diffset(W(("run5", "A", 16)), W(("run7", "A", 16))),
]
main_union = set().union(*main_pairs)
bb = diffset(W(("run6", "B", 16)), W(("run7", "B", 16)))
print(
    f"run7 A vs B differing samples: {len(ab7)}; of these also differing between two main boots: {len(ab7 & main_union)}; also differing between the two B boots (run6 B vs run7 B): {len(ab7 & bb)}; in neither: {len(ab7 - main_union - bb)}"
)
print("never-before-flipping samples in run7 A vs B:", sorted(ab7 - main_union - bb))
print(
    f"union of samples that flip in any same-kernel c16 pair: {len(main_union | bb | ab7)} of 1088"
)
# ASR judge noise: identical WAVs (run7 c1) yet different transcripts
a1, b1 = bymap(W(("run7", "A", 1))), bymap(W(("run7", "B", 1)))
print(
    "run7 c1 identical WAVs, transcripts differing:",
    sum(1 for i in a1 if a1[i]["hyp_norm"] != b1[i]["hyp_norm"]),
)
