"""Summarize thinker_slice_probe records: every prefill chunk with video rows, and whether its rows equal the encoder's.

usage: python3 thinker_slice_read.py '<prefix>.*.jsonl' [...]

Prints the chunks checked, how many shared their step with other requests, how many started at a boundary other than
a multiple of the chunk size (so co-batching placed it), and the count of chunks whose video embedding or DeepStack rows
differ from the encoder rows the prompt fixes; each differing chunk is listed.
"""

import collections
import glob
import json
import sys

CHUNK = 8192


def main() -> None:
    records = []
    for pattern in sys.argv[1:]:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                records += [json.loads(line) for line in handle]
    video = [r for r in records if r.get("video_rows")]
    print(f"{len(records)} request steps, {len(video)} with video rows")
    print(f"  requests: {len({r['rid'] for r in video})}")
    print(
        f"  shared the step with other requests: {sum(r['batch_reqs'] > 1 for r in video)}"
    )
    print(
        f"  started off a {CHUNK} boundary: {sum(r['prefix'] % CHUNK != 0 for r in video)}"
    )
    print(
        f"  ended off a {CHUNK} boundary and before the prompt end: "
        f"{sum((r['prefix'] + r['length']) % CHUNK != 0 and r['prefix'] + r['length'] < r['prompt_len'] for r in video)}"
    )
    print(f"  video rows checked: {sum(r['video_rows'] for r in video)}")
    embed = collections.Counter(r["video_embed_equal"] for r in video)
    deepstack = collections.Counter(r.get("video_deepstack_equal") for r in video)
    print(f"  embedding rows equal to the encoder's: {dict(embed)}")
    print(f"  DeepStack rows equal to the encoder's: {dict(deepstack)}")
    for r in video:
        if r["video_embed_equal"] is False or r.get("video_deepstack_equal") is False:
            print(f"  DIFFERS {r}")
        else:
            pass
    sizes = collections.Counter(r["batch_reqs"] for r in video)
    print(f"  requests in the step: {dict(sorted(sizes.items()))}")


if __name__ == "__main__":
    main()
