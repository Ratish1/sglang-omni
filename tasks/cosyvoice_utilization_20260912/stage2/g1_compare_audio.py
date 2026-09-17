#!/usr/bin/env python3
"""G1: is one arm's emitted audio byte identical to another's.

Takes directories written by g1_stream_identity.sh. Checks that the runs asked
the same thing (same ordered corpus, same client contract, all complete), then
compares the generated audio file of every sample byte for byte.

Not every request on this box is reproducible: a few diverge from their first
audio sample between two boots of one revision, so identity against a single
baseline confuses a real change with that. --control takes a second run of the
baseline revision. Samples the two baseline runs disagree on are unstable, are
named in the report and carry no verdict; the verdict is identity on the rest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IGNORED_CONFIG = {"output_dir", "model", "base_url"}


def read_json(path: Path):
    return json.loads(path.read_text())


def audio_digests(run: Path) -> dict[str, tuple[str, int]]:
    directory = run / "seedtts" / "measured" / "audio"
    if not directory.is_dir():
        raise SystemExit(f"{run} holds no measured/audio directory")
    digests = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            digests[path.name] = (hashlib.sha256(raw).hexdigest(), len(raw))
    if not digests:
        raise SystemExit(f"{directory} is empty")
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--control", type=Path, help="a second run of the baseline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    paths = [args.baseline, args.candidate]
    labels = ["baseline", "candidate"]
    if args.control is not None:
        paths.append(args.control)
        labels.append("control")
    runs = [read_json(path / "seedtts" / "experiment.json") for path in paths]
    for label, path, run in zip(labels, paths, runs, strict=True):
        if run["status"] != "complete":
            raise SystemExit(f"{label} {path} is {run['status']}, not complete")
        if run["ordered_inputs_sha256"] != runs[0]["ordered_inputs_sha256"]:
            raise SystemExit(f"{label} did not send the baseline's ordered corpus")
    contracts = [
        {k: v for k, v in run["client_config"].items() if k not in IGNORED_CONFIG}
        for run in runs
    ]
    for label, contract in zip(labels, contracts, strict=True):
        if contract != contracts[0]:
            differing = sorted(k for k in contract if contract[k] != contracts[0].get(k))
            raise SystemExit(f"{label} client contract differs: " + ", ".join(differing))
    if contracts[0].get("seed") is None:
        raise SystemExit("G1 compares seeded runs; neither run set a seed")

    digests = [audio_digests(path) for path in paths]
    for label, digest in zip(labels, digests, strict=True):
        if set(digest) != set(digests[0]):
            raise SystemExit(f"{label} wrote a different set of sample files")
    baseline, candidate = digests[0], digests[1]
    unstable = (
        sorted(name for name in baseline if baseline[name] != digests[2][name])
        if args.control is not None
        else []
    )
    gated = [name for name in sorted(baseline) if name not in set(unstable)]
    differing = [name for name in gated if baseline[name] != candidate[name]]
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "control": None if args.control is None else str(args.control),
        "baseline_head": (args.baseline / "head.txt").read_text().strip(),
        "candidate_head": (args.candidate / "head.txt").read_text().strip(),
        "ordered_inputs_sha256": runs[0]["ordered_inputs_sha256"],
        "seed": contracts[0]["seed"],
        "samples": len(baseline),
        "unstable": unstable,
        "gated": len(gated),
        "identical": len(gated) - len(differing),
        "differing": {
            name: {"baseline": baseline[name], "candidate": candidate[name]}
            for name in differing
        },
        "verdict": "pass" if not differing else "fail",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    raise SystemExit(0 if not differing else 1)


if __name__ == "__main__":
    main()
