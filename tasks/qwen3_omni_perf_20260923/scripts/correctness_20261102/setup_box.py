"""A fresh lease's downloads, so a check never waits on one: every model and dataset the correctness checks read.

usage: PYTHONPATH=TREE python3 setup_box.py   (one line per item; "setup done" at the end)
"""

from huggingface_hub import snapshot_download

from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.dataset.videomme import load_videomme_samples


def main() -> None:
    for model in ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-ASR-1.7B"):
        print(model, snapshot_download(model), flush=True)
    print("seedtts en", len(load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", None, split="en")), flush=True)
    print("mmsu-ci-2000", len(load_mmsu_samples(max_samples=None, repo_id=DATASETS["mmsu-ci-2000"])), flush=True)
    print("mmmu-ci-50", len(load_mmmu_samples(repo_id=DATASETS["mmmu-ci-50"])), flush=True)
    print("mmmu", len(load_mmmu_samples()), flush=True)
    print("videomme-ci-50", len(load_videomme_samples(max_samples=None, repo_id=DATASETS["videomme-ci-50"])), flush=True)
    print("setup done", flush=True)


if __name__ == "__main__":
    main()
