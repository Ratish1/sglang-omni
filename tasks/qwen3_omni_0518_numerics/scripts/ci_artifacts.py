# SPDX-License-Identifier: Apache-2.0
"""Download and compare Qwen3-Omni CI result artifacts across Omni CI runs.

Usage:
    python ci_artifacts.py download --out DIR RUN_ID [RUN_ID ...]
    python ci_artifacts.py compare --root DIR --pre RUN_ID,... --post RUN_ID,...
    python ci_artifacts.py compare-local --pre PATH,... --post PATH,...

download fetches the four Qwen3-Omni result artifacts (stages 5, 8, 9, 10)
for each run with gh. compare prints the per-attempt gate values and the
per-sample changes between the two run groups. compare-local takes result
JSON files produced by local benchmark runs (one file per attempt) and
applies the same per-sample comparison, so H100 ablation arms are read the
same way as CI runs.

Artifacts keep one directory per pytest attempt (pytest-0, pytest-1, ...).
The pytest-current directory duplicates the last attempt and is skipped.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import subprocess
import sys

ARTIFACTS = (
    "qwen3-omni-mmsu-results",
    "qwen3-omni-videomme-talker-results",
    "qwen3-omni-videoamme-results",
    "qwen3-omni-videoamme-talker-tp2-results",
)
REPO = "sgl-project/sglang-omni"


def download(out: str, runs: list[str]) -> None:
    for run in runs:
        for name in ARTIFACTS:
            dest = os.path.join(out, run, name)
            if os.path.isdir(dest):
                continue
            subprocess.run(
                ["gh", "run", "download", run, "--repo", REPO, "-n", name, "-D", dest],
                check=False,
                capture_output=True,
            )


def _attempts(
    root: str, run: str, artifact: str, pattern: str
) -> list[tuple[str, dict]]:
    out = []
    for d in sorted(
        glob.glob(os.path.join(root, run, artifact, "pytest-of-root", "pytest-[0-9]*"))
    ):
        files = [
            f
            for f in glob.glob(os.path.join(d, "**", pattern), recursive=True)
            if "current" not in f
        ]
        if files:
            out.append((f"{run}/{os.path.basename(d)}", json.load(open(files[0]))))
    return out


def _load_group(root: str, runs: list[str]) -> dict[str, list[tuple[str, dict]]]:
    g: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for run in runs:
        g["mmsu"] += _attempts(root, run, ARTIFACTS[0], "mmsu_results.json")
        g["mme"] += _attempts(root, run, ARTIFACTS[1], "videomme_results.json")
        g["amme"] += _attempts(root, run, ARTIFACTS[2], "videoamme_results.json")
        g["tp2"] += _attempts(root, run, ARTIFACTS[3], "videoamme_results.json")
    return g


def _load_local(paths: list[str]) -> dict[str, list[tuple[str, dict]]]:
    g: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        label = "/".join(p.rstrip("/").split("/")[-3:])
        if "speed_metrics" in d:
            g["mmsu"].append((label, d))
        elif d.get("config", {}).get("enable_audio"):
            g["mme"].append((label, d))
        else:
            g["amme"].append((label, d))
    return g


def _video_rows(items):
    rows = []
    for name, d in items:
        s, sp = d["summary"], d["speed"]
        rows.append(
            (
                name,
                s["accuracy"],
                sp["throughput_qps"],
                sp["latency_mean_s"],
                sp["output_tokens_mean"],
                sp.get("rtf_mean"),
            )
        )
    return rows


def _print_gate_table(label: str, rows, cols) -> None:
    print(f"\n=== {label} ===")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(x) for x in r))


def _video_preds(items):
    return [
        {
            x["sample_id"]: (
                x["predicted"],
                x["predicted"] == x["expected"],
                x["completion_tokens"],
            )
            for x in d["per_sample"]
        }
        for _, d in items
    ]


def _mmsu_preds(items):
    return [
        {
            x["sample_id"]: (x["predicted_choice"], x["is_correct"])
            for x in d["per_sample"]
        }
        for _, d in items
    ]


def _majority(values):
    return collections.Counter(values).most_common(1)[0][0]


def _compare_video(label: str, pre, post) -> None:
    if not pre or not post:
        return
    ppre, ppost = _video_preds(pre), _video_preds(post)
    samples = sorted(ppre[0].keys())
    print(
        f"\n=== {label}: per-sample changes (pre {len(ppre)} attempts, post {len(ppost)} attempts) ==="
    )
    print("sample | pred pre -> post | correct pre/post | tokens pre -> post")
    n_pred = n_tok = 0
    for s in samples:
        a = [p[s] for p in ppre if s in p]
        b = [p[s] for p in ppost if s in p]
        if not a or not b:
            continue
        pa, pb = _majority([x[0] for x in a]), _majority([x[0] for x in b])
        ta, tb = _majority([x[2] for x in a]), _majority([x[2] for x in b])
        ca = sum(x[1] for x in a) / len(a)
        cb = sum(x[1] for x in b) / len(b)
        if pa != pb:
            n_pred += 1
        if ta != tb:
            n_tok += 1
        if pa != pb or ta != tb or 0 < ca < 1 or 0 < cb < 1:
            print(f"{s} | {pa} -> {pb} | {ca:.2f}/{cb:.2f} | {ta} -> {tb}")
    print(
        f"prediction majority differs: {n_pred} of {len(samples)}, token-count majority differs: {n_tok} of {len(samples)}"
    )
    for name, group in (("pre", ppre), ("post", ppost)):
        unstable = sum(
            1 for s in samples if len({p[s][0] for p in group if s in p}) > 1
        )
        print(f"{name}: samples whose prediction differs across attempts: {unstable}")


def _compare_mmsu(pre, post) -> None:
    if not pre or not post:
        return
    ppre, ppost = _mmsu_preds(pre), _mmsu_preds(post)
    ids = list(ppre[0].keys())

    def rate(group, s):
        return sum(1 for p in group if p[s][1]) / len(group)

    flips = [(s, rate(ppre, s), rate(ppost, s)) for s in ids]
    systematic = [
        f
        for f in flips
        if (f[1] >= 0.9 and f[2] <= 0.1) or (f[1] <= 0.1 and f[2] >= 0.9)
    ]
    print(
        f"\n=== MMSU: samples correct in >=90% of one group's attempts and <=10% of the other's ==="
    )
    for f in systematic:
        print(f"{f[0]} pre {f[1]:.2f} post {f[2]:.2f}")
    print(
        f"count: {len(systematic)} (pre->wrong {sum(1 for f in systematic if f[1] > f[2])}, pre->right {sum(1 for f in systematic if f[1] < f[2])})"
    )
    for name, group in (("pre", ppre), ("post", ppost)):
        unstable = sum(1 for s in ids if 0 < rate(group, s) < 1)
        mean = statistics.mean(sum(p[s][1] for s in ids) / len(ids) for p in group)
        print(
            f"{name}: attempts {len(group)}, unstable samples {unstable}, mean accuracy {mean:.4f}"
        )


def _stage8_rows(items):
    rows = []
    for name, d in items:
        s, sp = d["summary"], d["speed"]
        ps = {x["sample_id"]: x for x in d["per_sample"]}
        top = sorted(ps.values(), key=lambda x: -(x["rtf"] or 0))[:2]
        rest = [x["rtf"] for x in ps.values() if x not in top]
        wer = (d.get("wer") or {}).get("summary", {}).get("wer_below_50_corpus")
        rows.append(
            (
                name,
                s["accuracy"],
                sp["rtf_mean"],
                sp["latency_mean_s"],
                sp["throughput_qps"],
                round(sum(rest) / len(rest), 3),
                [
                    (
                        t["sample_id"],
                        t["completion_tokens"],
                        round(t["audio_duration_s"], 2),
                        round(t["rtf"], 2),
                    )
                    for t in top
                ],
                wer,
            )
        )
    return rows


def compare(pre, post) -> None:
    for key, label in (
        ("mmsu", "stage 5 MMSU (gate 0.707)"),
        ("amme", "stage 9 Video-AMME FP8 (gate 0.64)"),
        ("tp2", "stage 10 FP8 TP2 (gate 0.5)"),
    ):
        for gname, g in (("pre", pre), ("post", post)):
            items = g.get(key, [])
            if not items:
                continue
            if key == "mmsu":
                rows = [
                    (
                        n,
                        d["summary"]["overall_accuracy"],
                        d["speed_metrics"]["throughput_qps"],
                        d["speed_metrics"]["latency_mean_s"],
                    )
                    for n, d in items
                ]
                _print_gate_table(
                    f"{label} [{gname}]",
                    rows,
                    ["attempt", "acc", "qps", "latency_mean_s"],
                )
            else:
                _print_gate_table(
                    f"{label} [{gname}]",
                    _video_rows(items),
                    [
                        "attempt",
                        "acc",
                        "qps",
                        "latency_mean_s",
                        "out_tokens_mean",
                        "rtf_mean",
                    ],
                )
    for gname, g in (("pre", pre), ("post", post)):
        if g.get("mme"):
            _print_gate_table(
                f"stage 8 Video-MME talker bf16 (rtf gate 1.12, acc 0.6) [{gname}]",
                _stage8_rows(g["mme"]),
                [
                    "attempt",
                    "acc",
                    "rtf_mean",
                    "latency_mean_s",
                    "qps",
                    "rtf_mean_without_top2",
                    "top2 (sample, tokens, audio_s, rtf)",
                    "wer",
                ],
            )
    _compare_video("stage 9 Video-AMME", pre.get("amme", []), post.get("amme", []))
    _compare_video("stage 8 Video-MME talker", pre.get("mme", []), post.get("mme", []))
    _compare_video("stage 10 Video-AMME TP2", pre.get("tp2", []), post.get("tp2", []))
    _compare_mmsu(pre.get("mmsu", []), post.get("mmsu", []))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("runs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("--root", required=True)
    c.add_argument("--pre", required=True)
    c.add_argument("--post", required=True)
    cl = sub.add_parser("compare-local")
    cl.add_argument("--pre", required=True)
    cl.add_argument("--post", required=True)
    a = p.parse_args(argv)
    if a.cmd == "download":
        download(a.out, a.runs)
    elif a.cmd == "compare":
        compare(
            _load_group(a.root, a.pre.split(",")),
            _load_group(a.root, a.post.split(",")),
        )
    else:
        compare(_load_local(a.pre.split(",")), _load_local(a.post.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
