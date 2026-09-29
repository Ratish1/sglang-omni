"""Compare the speaker embeddings and reference codes two servers served, clip by clip.

Each directory holds one file per reference, named by the reference file's content
hash, written by the probe module under OMNI_DUMP_REFERENCE.

usage: python reference_dump_diff.py DIR_A DIR_B
"""

from __future__ import annotations

import os
import statistics
import sys

import torch


def main() -> None:
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    keys = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
    print(
        f"clips: {len(os.listdir(a_dir))} in A, {len(os.listdir(b_dir))} in B, {len(keys)} common"
    )
    cosines, max_abs, codes_equal, rows = [], [], 0, []
    for key in keys:
        a = torch.load(os.path.join(a_dir, key))
        b = torch.load(os.path.join(b_dir, key))
        ea, eb = a["embedding"].double().flatten(), b["embedding"].double().flatten()
        cosine = float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
        cosines.append(cosine)
        max_abs.append(float((ea - eb).abs().max()))
        same_codes = (
            a["codes"] is not None
            and b["codes"] is not None
            and torch.equal(a["codes"], b["codes"])
        )
        codes_equal += same_codes
        rows.append(
            (
                cosine,
                key,
                same_codes,
                int(a["codes"].shape[0]) if a["codes"] is not None else 0,
            )
        )
    print(
        f"embedding cosine mean {statistics.fmean(cosines):.7f}, min {min(cosines):.7f}"
    )
    print(
        f"embedding max abs diff mean {statistics.fmean(max_abs):.5f}, max {max(max_abs):.5f}"
    )
    print(f"reference codes identical in {codes_equal} of {len(keys)}")
    for cosine, key, same_codes, frames in sorted(rows)[:5]:
        print(
            f"  lowest cosine {cosine:.7f} codes identical {same_codes} frames {frames} {key[:40]}"
        )


if __name__ == "__main__":
    main()
