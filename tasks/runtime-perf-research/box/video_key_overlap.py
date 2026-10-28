"""Whether hashing a video's frames overlaps the HF processor call, per Video-MME CI video, CPU only.

usage: OMP_NUM_THREADS=16 PYTHONPATH=<tree> python3 video_key_overlap.py --out overlap.json [--videos N] [--repeats 3]

Each video is loaded once as the preprocessor loads it (load_video_path at the CI's settings); then, repeated and the
fastest kept so the host's other load drops out: the video cache key alone, the processor call alone, the two in
sequence (main's order) and the two at once (the key on a thread while the processor runs).
"""

import argparse
import json
import statistics
import threading
import time

from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.preprocessing.video import compute_video_cache_key, load_video_path

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"


def fastest(function, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1e3)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--videos", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    preprocessor = Qwen3OmniPreprocessor(MODEL, max_seq_len=32768)
    processor = preprocessor.processor
    rows = []
    for sample in load_videomme_samples(args.videos, repo_id=VIDEOMME_CI):
        frames, fps = load_video_path(sample.video_path, fps=2.0, max_frames=128, max_pixels=401408)
        prompt = processor.apply_chat_template(
            preprocessor.build_multimodal_messages(
                [{"role": "user", "content": sample.question}], num_images=0, num_audios=0, num_videos=1
            ),
            add_generation_prompt=True,
            tokenize=False,
        )

        def key() -> None:
            compute_video_cache_key([frames])

        def process() -> None:
            processor(
                text=prompt,
                videos=[frames],
                add_special_tokens=False,
                return_tensors="pt",
                videos_kwargs={"fps": fps, "max_frames": 128, "max_pixels": 401408, "device": "cpu"},
            )

        def in_sequence() -> None:
            key()
            process()

        def at_once() -> None:
            hasher = threading.Thread(target=key)
            hasher.start()
            process()
            hasher.join()

        rows.append(
            {
                "id": sample.sample_id,
                "key": fastest(key, args.repeats),
                "processor": fastest(process, args.repeats),
                "in_sequence": fastest(in_sequence, args.repeats),
                "at_once": fastest(at_once, args.repeats),
            }
        )
    for name in ("key", "processor", "in_sequence", "at_once"):
        values = [row[name] for row in rows]
        print(f"{name}: p50 {statistics.median(values):.1f} ms mean {statistics.mean(values):.1f}")
    saved = [row["in_sequence"] - row["at_once"] for row in rows]
    print(f"in sequence minus at once: p50 {statistics.median(saved):.1f} ms mean {statistics.mean(saved):.1f}")
    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(rows, out)


if __name__ == "__main__":
    main()
