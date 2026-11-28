"""Count the questions whose raw response text and completion token count are one value over a set of passes.

Usage: python3 response_identity.py RUN_DIR LABEL=GLOB[,GLOB...] [LABEL=GLOB[,GLOB...] ...]
Each GLOB names videomme or videoamme result files (videomme_results.json, videoamme_results.json, or a talker pass's
json) relative to RUN_DIR. Every label is read alone, and every pair of labels is read together.
"""

import glob
import itertools
import json
import os
import sys


def responses(path):
    with open(path) as file:
        data = json.load(file)
    rows = data.get("results") or data.get("per_sample") or data.get("samples")
    out = {}
    for row in rows:
        key = row.get("question_id") or row.get("id") or row.get("sample_id")
        text = (
            row.get("response")
            or row.get("raw_response")
            or row.get("prediction")
            or row.get("output")
        )
        out[key] = (text, row.get("completion_tokens") or row.get("output_tokens"))
    return out


def read(runs):
    keys = set.intersection(*(set(run) for run in runs.values()))
    same_text = sum(
        1 for key in keys if len({run[key][0] for run in runs.values()}) == 1
    )
    same_tokens = sum(
        1 for key in keys if len({run[key][1] for run in runs.values()}) == 1
    )
    return len(runs), len(keys), same_text, same_tokens


def main():
    os.chdir(sys.argv[1])
    groups = {}
    for spec in sys.argv[2:]:
        label, patterns = spec.split("=", 1)
        paths = sorted(
            path for pattern in patterns.split(",") for path in glob.glob(pattern)
        )
        groups[label] = {path: responses(path) for path in paths}
    for label, runs in groups.items():
        if not runs:
            print(f"{label}: no result files")
            continue
        passes, questions, same_text, same_tokens = read(runs)
        print(
            f"{label}: {passes} passes, one text on {same_text} of {questions}, one token count on {same_tokens}"
        )
    for (left, left_runs), (right, right_runs) in itertools.combinations(
        groups.items(), 2
    ):
        if left_runs and right_runs:
            passes, questions, same_text, same_tokens = read(
                {**left_runs, **right_runs}
            )
            print(
                f"{left} with {right}: {passes} passes, one text on {same_text} of {questions}, one token count on {same_tokens}"
            )


if __name__ == "__main__":
    main()
