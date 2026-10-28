"""Whether qwen-tts's create_voice_clone_prompt(x_vector_only_mode=True) gives the speaker embedding our server computed
for the same reference. Two steps, two environments:

stage:   in the image's python, copy every SeedTTS EN reference clip into --wav-dir under its own file name.
compare: in a venv holding only qwen-tts 0.1.1 and its pinned dependencies, for the first --count references the probe
         build dumped (reference path, embedding, dtype), load the checkpoint with qwen-tts in the dumps' dtype, compute
         the embedding of the same clip (by file name in --wav-dir), and report exact equality, the max absolute
         difference and the cosine against the dump.

usage: PYTHONPATH=TREE python3 qwen_tts_embedding_compare.py stage --wav-dir DIR
       VENV/bin/python qwen_tts_embedding_compare.py compare --model PATH --dump-dir DIR --wav-dir DIR --device cuda:2
"""

import argparse
import shutil
from pathlib import Path

import torch


def stage(wav_dir: Path) -> None:
    from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
    from benchmarks.dataset.seedtts import load_seedtts_samples

    wav_dir.mkdir(parents=True, exist_ok=True)
    references = {sample.ref_audio for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")}
    for reference in references:
        shutil.copy(reference, wav_dir / Path(reference).name)
    print(f"staged {len(references)} reference clips in {wav_dir}", flush=True)


def compare(options: argparse.Namespace) -> None:
    import qwen_tts
    import transformers
    from qwen_tts import Qwen3TTSModel

    dumps = []
    for path in sorted(Path(options.dump_dir).glob("*.pt")):
        dump = torch.load(path)
        if dump["ref_audio"].startswith("/"):
            dumps.append(dump)
        else:
            pass
        if len(dumps) == options.count:
            break
        else:
            pass
    dtype = getattr(torch, dumps[0]["dtype"].removeprefix("torch."))
    model = Qwen3TTSModel.from_pretrained(options.model, device_map=options.device, dtype=dtype)
    print(
        f"{options.model}: qwen-tts {qwen_tts.__version__ if hasattr(qwen_tts, '__version__') else '0.1.1'}, "
        f"transformers {transformers.__version__}, torch {torch.__version__}, dumps in {dumps[0]['dtype']}, "
        f"loaded in {dtype}, {len(dumps)} references",
        flush=True,
    )
    exact = 0
    worst_difference = 0.0
    worst_cosine = 1.0
    for dump in dumps:
        clip = Path(options.wav_dir) / Path(dump["ref_audio"]).name
        (item,) = model.create_voice_clone_prompt(ref_audio=str(clip), x_vector_only_mode=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "compare"))
    parser.add_argument("--model", default="")
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=100)
    options = parser.parse_args()
    if options.mode == "stage":
        stage(Path(options.wav_dir))
    else:
        compare(options)


if __name__ == "__main__":
    main()
