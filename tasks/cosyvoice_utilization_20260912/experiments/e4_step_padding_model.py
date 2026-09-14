# Step padding rule model: replays a run's audio durations through the hop schedule and
# the E3 cost model to compare round time per admission rule. Usage: python e4_step_padding_model.py <speed_results.json>
import json
import random
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
D = [r["audio_duration_s"] for r in d["per_request"] if r["is_success"]]
FLOOR = 0.28
C = 80e-6  # E3: in-server call floor, per padded row frame


def rows_for():
    rows = []
    for s in D:
        c = int(s * 25)
        ps = int(random.uniform(3, 10) * 25)
        hop, off = 25, 0
        while off < c:
            rows.append(2 * (ps + min(off + hop + 3, c)))
            off += hop
            hop = min(hop * 2, 100)
    return rows


def run(rr, rule):
    left = list(rr)
    t = 0
    n = 0
    while left:
        step = []
        W = 0
        U = 0
        rest = []
        for f in left:
            if step and not rule(len(step) + 1, max(W, f), U + f, f, W):
                rest.append(f)
                continue
            step.append(f)
            W = max(W, f)
            U += f
        t += FLOOR + C * len(step) * W
        n += 1
        left = rest
    return t, n


BE = FLOOR / C
rules = {
    "no bound": lambda r, W, U, f, w: True,
    "cap 8000": lambda r, W, U, f, w: r * W <= 8000,
    "cap+pad 25": lambda r, W, U, f, w: r * W <= 8000 and (r * W - U) * 100 <= 25 * U,
    "marginal": lambda r, W, U, f, w: (r * W - U) <= BE,
}
random.seed(1)
rows = rows_for()
print(f"break-even padding frames per extra call: {BE:.0f}")
for scen, pick in (
    ("typical round", lambda: random.sample(rows, 16)),
    ("one runaway 4200 row", lambda: random.sample(rows, 15) + [4200]),
):
    print("==", scen)
    for name, rule in rules.items():
        res = [run(pick(), rule) for _ in range(3000)]
        print(
            f"  {name:11s} round s mean {st.mean(r[0] for r in res):.2f}  steps {st.mean(r[1] for r in res):.2f}"
        )
print("== marginal rule sensitivity to the budget (typical / runaway round s)")
for be in (1000, 1750, 3500, 7000, 14000):
    rule = lambda r, W, U, f, w, be=be: (r * W - U) <= be
    a = st.mean(run(random.sample(rows, 16), rule)[0] for _ in range(2000))
    b = st.mean(run(random.sample(rows, 15) + [4200], rule)[0] for _ in range(2000))
    print(f"  budget {be:5d}: {a:.2f} / {b:.2f}")
