"""The Qwen3-TTS reference code batcher as serving builds it (the cached speech tokenizer, the serving graph ladder),
on the tree on PYTHONPATH: a long-reference cell's warm list encoded first, as a server's warmup would, then its run
list in order; per run reference its frames and the wall time of the batcher's encode (submit to codes, after the
batcher's own synchronize), and the run's median, mean and count over 30 ms.

usage: PYTHONPATH=<tree> python3 reference_batcher_bench.py --card 3 --warm <warm.lst> --run <run.lst>
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile

from sglang_omni.models.qwen3_tts import request_builders
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def references(meta: Path) -> list[Path]:
    return [meta.parent / line.split("|")[2] for line in meta.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=f"cuda:{args.card}", dtype="bfloat16", attn_implementation=None)
    batcher = request_builders.Qwen3TTSRefCodeBatcher(
        tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    )
    hop = batcher.hop
    try:
        for path in references(args.warm):
            waveform, sample_rate = soundfile.read(path, dtype="float32")
            batcher.encode(np.asarray(waveform), sample_rate)
        times = []
        for path in references(args.run):
            waveform, sample_rate = soundfile.read(path, dtype="float32")
            start = time.perf_counter()
            codes = batcher.encode(np.asarray(waveform), sample_rate)
            times.append(1000 * (time.perf_counter() - start))
            print(f"{path.name}: {codes.shape[0]} frames (hop {hop}), {times[-1]:.1f} ms")
        print(
            f"run of {len(times)}: median {statistics.median(times):.1f} ms, mean {statistics.mean(times):.1f} ms, "
            f"over 30 ms {sum(value > 30 for value in times)}"
        )
    finally:
        batcher.close()


if __name__ == "__main__":
    main()
