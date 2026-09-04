import json
import statistics as st

S = "/private/tmp/claude-501/-Users-ratish-sglang-omni/755476ed-3ef4-4375-b034-a20d476b4162/scratchpad"
R = f"{S}/run7/qwen3tts-startup-capture-validation-b6fdc94d7/ab-seed1234"
for arm in ("A", "B"):
    rows = json.load(open(f"{R}/{arm}_c1/speed_results.json"))["per_request"]
    seen_enc, seen_voc = set(), set()
    recs = []
    for i, r in enumerate(rows):
        enc = r["prompt_tokens"]
        voc = r["prompt_tokens"] + r["completion_tokens"]
        fe = enc not in seen_enc
        fv = voc not in seen_voc
        seen_enc.add(enc)
        seen_voc.add(voc)
        recs.append(
            (i, r["latency_s"], r["completion_tokens"], r["prompt_tokens"], fe, fv)
        )
    recs = recs[1:]  # drop the first request (capture, warm caches)
    base = [x for x in recs if not x[4] and not x[5]]

    # least squares latency = a + b*completion + c*prompt on already-seen requests
    def lstsq3(rows):
        # normal equations for latency = a + b*completion + c*prompt
        M = [[0.0] * 3 for _ in range(3)]
        v = [0.0] * 3
        for x in rows:
            f = (1.0, float(x[2]), float(x[3]))
            for i in range(3):
                v[i] += f[i] * x[1]
                for j in range(3):
                    M[i][j] += f[i] * f[j]
        for i in range(3):
            piv = M[i][i]
            for j in range(3):
                M[i][j] /= piv
            v[i] /= piv
            for k in range(3):
                if k != i:
                    fac = M[k][i]
                    for j in range(3):
                        M[k][j] -= fac * M[i][j]
                    v[k] -= fac * v[i]
        return v

    coef = lstsq3(base)

    def resid(x):
        return x[1] - (coef[0] + coef[1] * x[2] + coef[2] * x[3])

    groups = {
        "neither": base,
        "enc only": [x for x in recs if x[4] and not x[5]],
        "voc only": [x for x in recs if x[5] and not x[4]],
        "both": [x for x in recs if x[4] and x[5]],
    }
    print(
        f"{arm} c1: fit a={coef[0]*1000:.1f} ms, b={coef[1]*1000:.2f} ms/completion token, c={coef[2]*1000:.2f} ms/prompt token, n_base={len(base)}"
    )
    for name, g in groups.items():
        if not g:
            continue
        rs = [resid(x) * 1000 for x in g]
        print(
            f"   {name:9s} n={len(g):4d} residual mean {st.mean(rs):+7.1f} ms median {st.median(rs):+7.1f} ms sd {st.pstdev(rs):.1f}"
        )
    fs = [x for x in recs if x[4] or x[5]]
    print(
        f"   first-seen requests: {len(fs)} of {len(recs)}, in first 50: {sum(1 for x in fs if x[0] < 50)}, first 100: {sum(1 for x in fs if x[0] < 100)}"
    )
