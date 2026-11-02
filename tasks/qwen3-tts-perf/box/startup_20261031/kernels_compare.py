# Block 76 reduction: codec_kernels.py JSONs of two arms. Per key the replay time of each run and the kernel counts; the
# kernels aligned by name stem in order (difflib), with the difference summed per stem over the arm's runs; the launches
# whose grid or block differ.
# usage: python3 kernels_compare.py <dir> <main run,main run> <pr run,pr run> [key ...]
import difflib
import json
import re
import statistics
import sys


def stem(name):
    return re.sub(r"_\d+(\.\w+)?$", "", name)


def main():
    directory, main_runs, pr_runs = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    wanted = sys.argv[4:]
    mains = [json.load(open(f"{directory}/{run}.json"))["keys"] for run in main_runs]
    prs = [json.load(open(f"{directory}/{run}.json"))["keys"] for run in pr_runs]
    for key in mains[0]:
        main_ms = [run[key]["ms"] for run in mains]
        pr_ms = [run[key]["ms"] for run in prs]
        main_kernels = mains[0][key]["kernels"]
        pr_kernels = prs[0][key]["kernels"]
        sums = lambda runs: [sum(item["us"] for item in run[key]["kernels"]) for run in runs]
        print(
            f"{key}: replay main {main_ms} pr {pr_ms} ms ({100 * (statistics.mean(pr_ms) / statistics.mean(main_ms) - 1):+.2f} %); "
            f"kernels {len(main_kernels)} / {len(pr_kernels)}; kernel us summed {[round(v, 1) for v in sums(mains)]} / {[round(v, 1) for v in sums(prs)]}"
        )
        if wanted and key not in wanted:
            continue
        else:
            pass
        median_us = lambda runs, index: statistics.median(run[key]["kernels"][index]["us"] for run in runs)
        matcher = difflib.SequenceMatcher(a=[stem(k["name"]) for k in main_kernels], b=[stem(k["name"]) for k in pr_kernels], autojunk=False)
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(a1 - a0):
                    m, p = main_kernels[a0 + offset], pr_kernels[b0 + offset]
                    mu, pu = median_us(mains, a0 + offset), median_us(prs, b0 + offset)
                    if m["grid"] != p["grid"] or m["block"] != p["block"] or abs(pu - mu) >= 0.5:
                        print(f"   {a0 + offset:>3} {stem(m['name'])[:70]:<70} grid {m['grid']} -> {p['grid']} block {m['block']} -> {p['block']} {mu:.2f} -> {pu:.2f} us")
                    else:
                        pass
            else:
                print(f"   {tag} main[{a0}:{a1}] {[(stem(k['name'])[:50], k['grid'], k['block'], k['us']) for k in main_kernels[a0:a1]]}")
                print(f"   {tag} pr[{b0}:{b1}] {[(stem(k['name'])[:50], k['grid'], k['block'], k['us']) for k in pr_kernels[b0:b1]]}")


if __name__ == "__main__":
    main()
