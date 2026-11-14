"""Per question Video-MME answers of two arms: input identity, answer flips within and across arms, unparsed responses.

usage: python3 videomme_answer_diff.py <run dir> main=boot,boot pr=boot,boot [--show N]

A pass is every videomme_results.json or videoamme_results.json under a boot. Input identity is the prompt token count per question,
which covers the frames the preprocessor produced. A flip is a question whose predicted letter differs
between passes; flips inside one arm bound what batching alone moves.
"""

import argparse
import collections
import json
from pathlib import Path


def passes(run_dir: Path, boots: list[str]) -> list[tuple[str, dict]]:
    found = []
    for boot in boots:
        for path in sorted((run_dir / boot).rglob("video*me_results.json")):
            label = f"{boot}/{path.parent.relative_to(run_dir / boot)}"
            found.append((label, json.loads(path.read_text(encoding="utf-8"))))
    return found


def by_question(result: dict) -> dict[str, dict]:
    return {sample["question_id"]: sample for sample in result["per_sample"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("arms", nargs=2)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()
    arms = []
    for spec in args.arms:
        name, boots = spec.split("=")
        arms.append((name, passes(args.run_dir, boots.split(","))))

    for name, arm_passes in arms:
        print(f"{name}: {len(arm_passes)} passes")
        for label, result in arm_passes:
            summary, speed = result["summary"], result["speed"]
            print(
                f"  {label:40s} acc {summary['accuracy']:.3f} correct {summary['correct']:2d} "
                f"unparsed {summary['unparseable']} max_token_hits {speed['max_token_hits']} failed {summary['failed']}"
            )

    tokens = collections.defaultdict(set)
    for _, arm_passes in arms:
        for _, result in arm_passes:
            for qid, sample in by_question(result).items():
                tokens[qid].add(sample["prompt_tokens"])
    differing = {qid: counts for qid, counts in tokens.items() if len(counts) > 1}
    print(
        f"\ninput identity: {len(tokens) - len(differing)} of {len(tokens)} questions have one prompt token count over every pass of both arms"
    )
    for qid, counts in sorted(differing.items()):
        print(f"  {qid}: {sorted(counts)}")

    expected = {}
    letters = {}
    for name, arm_passes in arms:
        for label, result in arm_passes:
            for qid, sample in by_question(result).items():
                expected[qid] = sample["expected"]
                letters.setdefault(qid, {}).setdefault(name, []).append(
                    sample["predicted"] or "-"
                )

    (base_name, _), (pr_name, _) = arms
    stable_same, flips_base, flips_pr, stable_diff = [], [], [], []
    for qid in sorted(letters):
        base_set, pr_set = set(letters[qid][base_name]), set(letters[qid][pr_name])
        if len(base_set) == 1 and base_set == pr_set:
            stable_same.append(qid)
        elif len(base_set) == 1 and len(pr_set) == 1:
            stable_diff.append(qid)
        else:
            if len(base_set) > 1:
                flips_base.append(qid)
            else:
                pass
            if len(pr_set) > 1:
                flips_pr.append(qid)
            else:
                pass

    print(
        f"\nanswers: {len(stable_same)} questions give one letter on every pass of both arms"
    )
    print(
        f"  stable in each arm but different between arms: {len(stable_diff)} {stable_diff}"
    )
    print(f"  flip across {base_name}'s own passes: {len(flips_base)} {flips_base}")
    print(f"  flip across {pr_name}'s own passes: {len(flips_pr)} {flips_pr}")
    moving = sorted(set(stable_diff) | set(flips_base) | set(flips_pr))
    print("\nper moving question, expected, then each arm's letters by pass:")
    for qid in moving:
        print(
            f"  {qid} exp {expected[qid]}  {base_name} {''.join(letters[qid][base_name])}  "
            f"{pr_name} {''.join(letters[qid][pr_name])}"
        )

    if args.show:
        print("\nraw responses of the moving questions, first pass of each arm:")
        for qid in moving[: args.show]:
            for name, arm_passes in arms:
                sample = by_question(arm_passes[0][1])[qid]
                text = " ".join(sample["raw_response"].split())
                print(
                    f"--- {qid} {name} predicted {sample['predicted']} expected {sample['expected']} "
                    f"completion_tokens {sample['completion_tokens']}\n{text[:700]}"
                )


if __name__ == "__main__":
    main()
