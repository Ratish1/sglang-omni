"""Whether qwen-tts's create_voice_clone_prompt(x_vector_only_mode=True) gives the speaker embedding our server computes
for the same reference: for the first --count references the probe build dumped (reference path, embedding, dtype),
load the checkpoint with qwen-tts (sglang-omni's transformers compatibility patches first) in the dumps' dtype, compute
each reference's embedding, and report exact equality, the max absolute difference and the cosine against the dump.

usage: PYTHONPATH=TREE python3 qwen_tts_embedding_compare.py --model PATH --dump-dir DIR --device cuda:2 [--count 50]
"""

import argparse
from pathlib import Path

import torch

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--count", type=int, default=50)
    options = parser.parse_args()
    dumps = [torch.load(path) for path in sorted(Path(options.dump_dir).glob("*.pt"))[: options.count]]
    dtype = getattr(torch, dumps[0]["dtype"].removeprefix("torch."))
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(options.model, device_map=options.device, dtype=dtype)
    print(f"{options.model}: dumps in {dumps[0]['dtype']}, qwen-tts loaded in {dtype}, {len(dumps)} references", flush=True)
    exact = 0
    worst_difference = 0.0
    worst_cosine = 1.0
    for dump in dumps:
        (item,) = model.create_voice_clone_prompt(ref_audio=dump["ref_audio"], x_vector_only_mode=True)
        computed = item.ref_spk_embedding.detach().float().cpu()
        served = dump["embedding"]
        difference = (computed - served).abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(computed, served, dim=0).item()
        exact += int(torch.equal(computed, served))
        worst_difference = max(worst_difference, difference)
        worst_cosine = min(worst_cosine, cosine)
    print(
        f"bit identical {exact} of {len(dumps)}, max abs difference {worst_difference:.3e}, lowest cosine "
        f"{worst_cosine:.6f}, embedding length {served.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
