"""Every mmsu-ci-2000 clip through each tree's serving loader, against HF's reference load, bit for bit.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/mmsu_audio_bits.py --trees upw,fle --out /data/pr2800/<run>
1. The samples from main's benchmark loader: the files the client sends as paths (mmsu.py writes each clip's bytes to
   <id>.mp3 whatever their format).
2. Per tree, in that tree's own code (this file in its --tree mode, cwd and PYTHONPATH the tree): load_audio_path(path,
   16000), which ensure_audio_list_async calls for a local path, then the Qwen3-Omni processor's input_features and
   audio token count for that array, called as the preprocessor calls it (audio only, add_special_tokens=False).
3. In the same process: the reference, librosa.load(path, sr=16000) as qwen_omni_utils.process_audio_info loads a
   local path, through the same processor; and SGLang's own load_audio(path, 16000).
Per clip: the stored rate and channels, each array's length, whether the served array and the served features equal the
reference's bit for bit, the largest absolute differences, and the audio token counts. Writes <out>/<tree>.jsonl and
<out>/summary.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REPO_ID = "zhaochenyang20/mmsu-ci-2000"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_PROMPT = "<|audio_start|><|audio_pad|><|audio_end|>"
WORKERS = 12

processor = None
thinker_config = None


def write_samples(path: Path) -> None:
    from benchmarks.dataset.mmsu import load_mmsu_samples

    samples = load_mmsu_samples(repo_id=REPO_ID)
    path.write_text(json.dumps([dict(sample_id=s.sample_id, audio_path=s.audio_path) for s in samples]))


def start_worker() -> None:
    global processor, thinker_config
    from transformers import AutoConfig, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    thinker_config = AutoConfig.from_pretrained(MODEL).thinker_config


def features(array):
    inputs = processor(text=AUDIO_PROMPT, audio=[array], add_special_tokens=False, return_tensors="pt")
    audio_tokens = int((inputs["input_ids"][0] == processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")).sum())
    return inputs, audio_tokens


def positions_digest(inputs) -> str:
    import hashlib

    from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions

    positions, delta = compute_mrope_positions(
        inputs["input_ids"][0],
        {"audio_feature_lengths": inputs["feature_attention_mask"].sum(-1)},
        thinker_config,
    )
    digest = hashlib.sha256(positions.to(dtype=positions.dtype).contiguous().numpy().tobytes()).hexdigest()[:16]
    return f"{digest} {positions.dtype} delta {delta.flatten().tolist()}"


def largest_difference(a, b) -> float:
    import numpy as np

    length = min(a.shape[-1], b.shape[-1])
    if length == 0:
        return 0.0
    else:
        pass
    return float(np.max(np.abs(a[..., :length].astype(np.float64) - b[..., :length].astype(np.float64))))


def compare(sample: dict) -> dict:
    import librosa
    import numpy as np
    import soundfile
    from sglang.srt.utils.common import load_audio

    from sglang_omni.preprocessing.audio import load_audio_path

    path = sample["audio_path"]
    info = soundfile.info(path)
    served = load_audio_path(path, target_sr=16000)
    reference = librosa.load(path, sr=16000)[0]
    sglang = load_audio(path, sr=16000)
    served_inputs, served_tokens = features(served)
    reference_inputs, reference_tokens = features(reference)
    served_features = served_inputs["input_features"]
    reference_features = reference_inputs["input_features"]
    return dict(
        sample_id=sample["sample_id"],
        stored_rate=int(info.samplerate),
        channels=int(info.channels),
        format=info.format,
        served_length=int(served.shape[0]),
        reference_length=int(reference.shape[0]),
        sglang_length=int(sglang.shape[0]),
        served_dtype=str(served.dtype),
        served_equal=bool(served.dtype == reference.dtype and np.array_equal(served, reference)),
        served_difference=largest_difference(served, reference),
        sglang_equal=bool(np.array_equal(sglang.astype(np.float32), reference)),
        sglang_difference=largest_difference(sglang, reference),
        features_equal=bool(torch_equal(served_features, reference_features)),
        features_difference=largest_difference(served_features.numpy(), reference_features.numpy()),
        served_audio_tokens=served_tokens,
        reference_audio_tokens=reference_tokens,
        reference_positions=positions_digest(reference_inputs),
    )


def torch_equal(a, b) -> bool:
    import torch

    return a.shape == b.shape and torch.equal(a, b)


def run_tree(samples_path: Path, out_path: Path) -> None:
    samples = json.loads(samples_path.read_text())
    with Pool(WORKERS, initializer=start_worker) as pool:
        rows = pool.map(compare, samples, chunksize=8)
    out_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def summarize(rows: list[dict]) -> list[str]:
    resampled = [row for row in rows if row["stored_rate"] != 16000]
    native = [row for row in rows if row["stored_rate"] == 16000]
    lines = [
        f"  {len(rows)} clips; stored rates {dict(Counter(row['stored_rate'] for row in rows))}; channels "
        f"{dict(Counter(row['channels'] for row in rows))}; formats {dict(Counter(row['format'] for row in rows))}"
    ]
    for label, group in (("resampled", resampled), ("stored at 16 kHz", native)):
        if not group:
            continue
        else:
            pass
        differences = sorted(row["served_difference"] for row in group)
        lines.append(
            f"  {label} ({len(group)}): served array equal to librosa.load {sum(row['served_equal'] for row in group)}, "
            f"features equal {sum(row['features_equal'] for row in group)}, audio tokens equal "
            f"{sum(row['served_audio_tokens'] == row['reference_audio_tokens'] for row in group)}, lengths equal "
            f"{sum(row['served_length'] == row['reference_length'] for row in group)}; largest array difference "
            f"median {differences[len(differences) // 2]:.6f} max {differences[-1]:.6f}; SGLang's load_audio equal "
            f"{sum(row['sglang_equal'] for row in group)}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", help="arm,arm: the trees whose loaders are compared (pair_check arms)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tree", action="store_true", help="internal: compare in the current tree's code")
    parser.add_argument("--write-samples", action="store_true", help="internal: write the samples in this tree")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.tree:
        run_tree(args.samples, args.result)
        return
    elif args.write_samples:
        write_samples(args.samples)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.trees.split(",")
    trees = {arm: pair_check.tree(pair_check.ARMS[arm][0]) for arm in arms}
    samples_path = args.out / "samples.json"
    # note (ratish): CPU only, so no process opens a context on the cards other workloads hold.
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--write-samples", "--out", str(args.out), "--samples",
         str(samples_path)],
        cwd=trees[arms[0]], env={**cpu_env, "PYTHONPATH": str(trees[arms[0]])}, check=True,
    )
    pair_check.log(args.out, f"{len(json.loads(samples_path.read_text()))} samples")
    processes = {
        arm: subprocess.Popen(
            pair_check.pinned(
                card,
                [sys.executable, str(Path(__file__).resolve()), "--tree", "--out", str(args.out), "--samples",
                 str(samples_path), "--result", str(args.out / f"{arm}.jsonl")],
            ),
            cwd=tree, env={**cpu_env, "PYTHONPATH": str(tree)},
            stdout=open(args.out / f"{arm}.log", "w"), stderr=subprocess.STDOUT,
        )
        for card, (arm, tree) in enumerate(trees.items())
    }
    lines = []
    for arm, process in processes.items():
        rc = process.wait()
        pair_check.log(args.out, f"{arm} rc {rc}")
        if rc != 0:
            continue
        else:
            pass
        rows = [json.loads(line) for line in (args.out / f"{arm}.jsonl").read_text().splitlines()]
        lines += [f"{arm} ({trees[arm].name}):", *summarize(rows)]
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        pair_check.log(args.out, line)
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
