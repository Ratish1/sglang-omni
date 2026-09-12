#!/usr/bin/env python3
"""Compare matching profiler-off English SeedTTS artifacts; no benchmark execution."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    a, b = [read(p / "experiment.json") for p in (args.baseline, args.candidate)]
    for label, run in (("baseline", a), ("candidate", b)):
        if run["client_config"].get("lang") != "en":
            parser.error(f"{label} must use the English split")
        if run["profiled"] or run["status"] != "complete":
            parser.error(f"{label} must be a complete profiler-off run")
    if a["ordered_inputs_sha256"] != b["ordered_inputs_sha256"]:
        parser.error("Ordered input identity differs")
    ignored = {"output_dir", "model", "server_config"}
    ca = {k: v for k, v in a["client_config"].items() if k not in ignored}
    cb = {k: v for k, v in b["client_config"].items() if k not in ignored}
    if ca != cb:
        parser.error(
            "Client contracts differ: " + ", ".join(k for k in ca if ca[k] != cb.get(k))
        )
    results = [
        read(p / "measured" / "speed_results.json")
        for p in (args.baseline, args.candidate)
    ]
    if [r["id"] for r in results[0]["per_request"]] != [
        r["id"] for r in results[1]["per_request"]
    ]:
        parser.error("Native result order differs")
    metrics = {}
    for name, av in results[0]["summary"].items():
        bv = results[1]["summary"].get(name)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            finite = math.isfinite(av) and math.isfinite(bv)
            metrics[name] = {
                "baseline": av if math.isfinite(av) else str(av),
                "candidate": bv if math.isfinite(bv) else str(bv),
                "delta": bv - av if finite else None,
                "ratio": bv / av if finite and av else None,
            }
    out = {
        "ordered_inputs_sha256": a["ordered_inputs_sha256"],
        "sample_count": a["sample_count"],
        "full_split": a["full_split"] and b["full_split"],
        "metrics": metrics,
        "model_labels": [a["client_config"]["model"], b["client_config"]["model"]],
        "note": "Ratios are descriptive, not acceptance decisions. Match server/model identities separately; evaluate English WER, SIM, UTMOS and latency/continuity gates. Repeat paired runs for variance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
