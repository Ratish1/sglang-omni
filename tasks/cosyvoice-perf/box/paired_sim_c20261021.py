"""Paired speaker-similarity comparison of two arms of one cell: per clip, arm B's SIM minus
arm A's, over every clip and over the clips whose two arms share an audio length (the same
token count under one seed); the mean delta, its standard error and the counts. 10-21 copy
of the 0.5.22 program's paired_sim.py for the cosyvoice box.

usage: python3 paired_sim_c20261021.py <cell dir A> <cell dir B>
"""

import json
import os
import sys
import wave

import numpy as np

arm_a, arm_b = sys.argv[1], sys.argv[2]


def per_sample(cell):
    with open(os.path.join(cell, "similarity_results.json")) as handle:
        rows = json.load(handle)["per_sample"]
    keys = [key for key in rows[0] if "sim" in key.lower() and key != "id"]
    return {row["id"]: float(row[keys[0]]) for row in rows if row.get(keys[0]) is not None}, keys[0]


def frames(cell):
    lengths = {}
    for root, _, files in os.walk(cell):
        for name in files:
            if name.endswith(".wav"):
                with wave.open(os.path.join(root, name)) as wav:
                    lengths[name[: -len(".wav")]] = wav.getnframes()
    return lengths


sim_a, key = per_sample(arm_a)
sim_b, _ = per_sample(arm_b)
frames_a, frames_b = frames(arm_a), frames(arm_b)
ids = sorted(set(sim_a) & set(sim_b))
same_length = [id_ for id_ in ids if frames_a.get(id_) == frames_b.get(id_)]
for label, subset in (("all clips", ids), ("same length", same_length)):
    delta = np.array([sim_b[id_] - sim_a[id_] for id_ in subset])
    mean_a = np.mean([sim_a[id_] for id_ in subset])
    mean_b = np.mean([sim_b[id_] for id_ in subset])
    standard_error = delta.std(ddof=1) / np.sqrt(len(delta))
    print(
        f"{label}: n {len(subset)}, A {mean_a:.3f}, B {mean_b:.3f}, delta {delta.mean():+.3f} "
        f"(standard error {standard_error:.3f}), clips worse {int((delta < 0).sum())}, better {int((delta > 0).sum())}"
    )
print(f"score field: {key}")
