import json
import os
import statistics as st

S = "/private/tmp/claude-501/-Users-ratish-sglang-omni/755476ed-3ef4-4375-b034-a20d476b4162/scratchpad"
RUNS = {
    "run6": (
        f"{S}/run6/qwen3tts-predictor-capture-validation-20260903-final-961e608f5",
        "{arm}_c{c}",
    ),
    "run7": (
        f"{S}/run7/qwen3tts-startup-capture-validation-b6fdc94d7/ab-seed1234",
        "{arm}_c{c}",
    ),
}


def per_request(run, arm, c):
    base, pat = RUNS[run]
    sp = json.load(
        open(os.path.join(base, pat.format(arm=arm, c=c), "speed_results.json"))
    )
    return sp["per_request"]


for run in RUNS:
    A = per_request(run, "A", 1)
    B = per_request(run, "B", 1)
    assert [r["id"] for r in A] == [r["id"] for r in B]
    pairs = [(a, b) for a, b in zip(A, B)][1:]
    same_tokens = sum(
        1 for a, b in pairs if a["completion_tokens"] == b["completion_tokens"]
    )
    d = [b["latency_s"] - a["latency_s"] for a, b in pairs]
    tok = [a["completion_tokens"] for a, b in pairs]
    n = len(d)
    mean = st.mean(d)
    med = st.median(d)
    sd = st.pstdev(d)
    half = n // 2
    # least squares delta ~ intercept + slope * tokens
    mt = st.mean(tok)
    md = mean
    sxx = sum((t - mt) ** 2 for t in tok)
    sxy = sum((t - mt) * (x - md) for t, x in zip(tok, d))
    slope = sxy / sxx
    intercept = md - slope * mt
    per_tok = [
        (b["latency_s"] - a["latency_s"]) / a["completion_tokens"] for a, b in pairs
    ]
    print(f"{run} c1 pairs (first request excluded): n={n} same_tokens={same_tokens}")
    print(
        f"   B-A mean {mean*1000:+.2f} ms, median {med*1000:+.2f} ms, sd {sd*1000:.1f} ms, sem {sd/ n**0.5*1000:.2f} ms, B slower in {sum(1 for x in d if x>0)}"
    )
    print(
        f"   first half mean {st.mean(d[:half])*1000:+.2f} ms, second half mean {st.mean(d[half:])*1000:+.2f} ms"
    )
    q = n // 4
    print(
        "   quarter means ms:",
        [f"{st.mean(d[i*q:(i+1)*q])*1000:+.2f}" for i in range(4)],
    )
    print(
        f"   delta vs tokens: intercept {intercept*1000:+.2f} ms, slope {slope*1e6:+.1f} us/token, mean tokens {mt:.1f}; per-token delta mean {st.mean(per_tok)*1e6:+.1f} us"
    )
    print(
        f"   A mean {st.mean(a['latency_s'] for a,b in pairs)*1000:.1f} ms, B mean {st.mean(b['latency_s'] for a,b in pairs)*1000:.1f} ms; first request A {A[0]['latency_s']:.3f} B {B[0]['latency_s']:.3f}"
    )
    ttfp = [
        (b.get("audio_ttfp_s") or 0) - (a.get("audio_ttfp_s") or 0) for a, b in pairs
    ]
    print(f"   ttfp B-A mean {st.mean(ttfp)*1000:+.2f} ms")
