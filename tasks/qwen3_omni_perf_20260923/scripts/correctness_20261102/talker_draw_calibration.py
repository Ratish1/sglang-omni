"""Do the talker's served first-code draws follow the distribution HF's sampler defines (correctness plan row 13, L5)?

For every captured request (SGLANG_OMNI_TALKER_CAPTURE_DIR) and frame, the captured hidden goes through the checkpoint's
codec_head and HF's processor order: repetition penalty over the frame's earlier first codes, the top 1,024 ids but
EOS suppressed, temperature, top-k. If the draw follows that distribution, the randomized probability integral
transform u = cdf(code - 1) + v * p(code), v uniform, is uniform on [0, 1] and independent across frames. Reported: the
Kolmogorov-Smirnov test of u, its autocorrelation at lags 1 to 5 inside a request (a draw correlated across frames would
pass the marginal test and still repeat), the mean probability of the drawn code against its expectation, and the same
test split by frame position (first frames, later frames). This reads the whole serving path, seed plumbing included.

usage: python3 talker_draw_calibration.py OUT.json CAPTURE_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from scipy import stats
from transformers import AutoConfig

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEED = 20261102


def codec_head_weight(snapshot: str) -> torch.Tensor:
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
    name = "talker.codec_head.weight"
    with safe_open(Path(snapshot) / index["weight_map"][name], "pt") as handle:
        return handle.get_tensor(name).float()


def served_probs(logits, first_codes, suppressed, sampling) -> torch.Tensor:
    logits = logits.clone()
    penalty = float(sampling["repetition_penalty"])
    for frame in range(1, logits.shape[0]):
        seen = first_codes[:frame].unique()
        row = logits[frame, seen]
        logits[frame, seen] = torch.where(row > 0, row / penalty, row * penalty)
    logits[:, suppressed] = float("-inf")
    probs = torch.softmax(logits / float(sampling["temperature"]), dim=-1)
    kept = probs.topk(int(sampling["top_k"]), dim=-1)
    filtered = torch.zeros_like(probs).scatter_(-1, kept.indices, kept.values)
    return filtered / filtered.sum(-1, keepdim=True)


def ks_row(values: np.ndarray) -> dict:
    result = stats.kstest(values, "uniform")
    return {
        "count": int(values.size),
        "ks_statistic": float(result.statistic),
        "ks_p": float(result.pvalue),
        "mean": float(values.mean()),
    }


def main() -> None:
    out = Path(sys.argv[1])
    paths = sorted(Path(sys.argv[2]).glob("*.pt"))
    snapshot = snapshot_download(MODEL)
    config = AutoConfig.from_pretrained(snapshot).talker_config
    vocab = config.text_config.vocab_size
    suppressed = torch.tensor(
        [i for i in range(vocab - 1024, vocab) if i != config.codec_eos_token_id]
    )
    head = codec_head_weight(snapshot)
    generator = torch.Generator().manual_seed(SEED)
    all_u = []
    first_u = []
    later_u = []
    lag_pairs = {lag: ([], []) for lag in range(1, 6)}
    drawn_probability = []
    expected_probability = []
    top1_fraction = []
    for path in paths:
        record = torch.load(path, map_location="cpu")
        hidden = record["hidden"].float()
        first_codes = record["codes"][:, 0]
        logits = hidden @ head.T
        probs = served_probs(logits, first_codes, suppressed, record["sampling"])
        steps = torch.arange(first_codes.numel())
        cdf = probs.cumsum(-1)
        below = torch.where(
            first_codes > 0, cdf[steps, (first_codes - 1).clamp_min(0)], torch.zeros(steps.numel())
        )
        drawn = probs[steps, first_codes]
        uniform = torch.rand(steps.numel(), generator=generator)
        values = (below + uniform * drawn).clamp(0.0, 1.0).numpy()
        all_u.append(values)
        first_u.append(values[:10])
        later_u.append(values[10:])
        for lag, (left, right) in lag_pairs.items():
            if values.size > lag:
                left.append(values[:-lag])
                right.append(values[lag:])
        drawn_probability.append(drawn.numpy())
        expected_probability.append((probs * probs).sum(-1).numpy())
        top1_fraction.append((first_codes == probs.argmax(-1)).float().numpy())
    report = {
        "requests": len(paths),
        "all_frames": ks_row(np.concatenate(all_u)),
        "first_10_frames": ks_row(np.concatenate(first_u)),
        "later_frames": ks_row(np.concatenate(later_u)),
        "lag_correlation": {
            str(lag): float(np.corrcoef(np.concatenate(left), np.concatenate(right))[0, 1])
            for lag, (left, right) in lag_pairs.items()
        },
        "drawn_probability_mean": float(np.concatenate(drawn_probability).mean()),
        "expected_drawn_probability_mean": float(np.concatenate(expected_probability).mean()),
        "drawn_top1_fraction": float(np.concatenate(top1_fraction).mean()),
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
