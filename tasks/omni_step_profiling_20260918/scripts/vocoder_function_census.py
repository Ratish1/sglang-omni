"""The vocoder's attributed device time by function, from a pipeline census G section.

Reads census_Gall.txt (pipeline_census.py --sections G --top 2000) and groups the vocoder
owners' op rows by the source line that launched them: the decoder transformer (the
incremental attention, rope and K/V in incremental_codec.py, and HF's norm, layer scale and
MLP), conv kernels and their glue, the arena gather and scatter, the quantizer decode,
SnakeBeta, and the conv bias adds of PyTorch's cuDNN path. The line ranges are those of
incremental_codec.py and modeling_qwen3_tts_tokenizer_v2.py at main b97f66d98 to ddbb1c779.

usage: python vocoder_function_census.py census_Gall.txt
"""

from __future__ import annotations

import collections
import re
import sys


def category(op: str, where: str, kernel: str) -> str:
    path, _, line_text = where.rpartition(":")
    line = int(line_text) if line_text.isdigit() else -1
    if "codec_state_arena" in path:
        return "arena gather/scatter"
    elif "modeling_qwen3_tts_tokenizer_v2" in path:
        if 660 <= line <= 830:
            return "quantizer decode"
        elif 350 <= line <= 420:
            return "transformer (HF norm, scale, mlp)"
        elif 570 <= line <= 620:
            return "snake (HF)"
        else:
            return f"HF other {line}"
    elif "incremental_codec.py" in path:
        if line == 895:
            return "quantizer decode"
        elif 363 <= line <= 527:
            return "transformer (incremental attention, rope, kv)"
        elif 528 <= line <= 556:
            return "convnext"
        elif 557 <= line <= 576:
            return "residual add"
        elif 234 <= line <= 322:
            return "conv glue (cat, pad, clone, overlap, bias)"
        elif 147 <= line <= 233:
            return "depthwise conv (NCL)"
        else:
            return f"incremental_codec other {line}"
    elif "channels_last_conv" in path:
        if "elementwise" in kernel:
            return "conv bias add (cuDNN path)"
        else:
            return "conv"
    elif "snake_beta" in path:
        return "snake"
    elif "linear" in path or "functional" in path:
        return "linear"
    else:
        return f"other {path.split('/')[-1]}:{line}"


def main() -> None:
    rows = []
    in_op_section = False
    with open(sys.argv[1]) as census:
        lines = census.readlines()
    for text in lines:
        if "by op, library line and kernel" in text:
            in_op_section = True
            continue
        else:
            pass
        if not in_op_section:
            continue
        else:
            pass
        match = re.match(
            r"\s+(voc\.\w+)\s+([\d.]+) ms\s+[\d.]+%\s+(\S+)\s+(\S+)\s+(.*)", text
        )
        if match:
            rows.append(
                (
                    match.group(1),
                    float(match.group(2)),
                    match.group(3),
                    match.group(4),
                    match.group(5),
                )
            )
        else:
            pass
    totals: collections.Counter[str] = collections.Counter()
    by_owner: collections.Counter[tuple[str, str]] = collections.Counter()
    for owner, ms, op, where, kernel in rows:
        name = category(op, where, kernel)
        totals[name] += ms
        by_owner[(name, owner)] += ms
    attributed = sum(totals.values())
    print(
        f"vocoder attributed device time {attributed:.0f} ms over {len(rows)} op rows"
    )
    for name, ms in totals.most_common():
        initial = by_owner[(name, "voc.initial")]
        followup = by_owner[(name, "voc.followup")]
        print(
            f"  {ms:7.0f} ms  {100 * ms / attributed:5.1f}%  {name}"
            f"  (initial {initial:.0f}, follow-up {followup:.0f})"
        )


if __name__ == "__main__":
    main()
