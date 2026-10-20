"""The torch speech tokenizer V3 against ONNX Runtime (10-20, L3 of DESIGN_PR_STACK_20261020.md 3): codes of every
seed-tts EN reference from ONNX Runtime's CUDA session (main's options), from the torch model at the reference's own
width (TF32 matmuls off and on), and from the CUDA graphs at the padded width. Per pair of methods: references and
tokens differing, length mismatches. Also the graphs' capture time and memory, and each method's time per call on an
idle card. ONNX Runtime differs from itself across processes in 0.05 to 0.06 % of tokens (queue h).
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<tree>:<onnx dir> python3 tokenizer_parity_c20261020.py <out json>"""

import itertools
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime
import torch
import whisper

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3.request_builders import load_prompt_audio
from sglang_omni.models.fun_cosyvoice3.speech_tokenizer import (
    SpeechTokenizerCudaGraphs,
    conv_frames,
    load_speech_tokenizer,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def main() -> None:
    out = sys.argv[1]
    device = torch.device("cuda", 0)
    model_path = os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx")
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    mels = [
        whisper.log_mel_spectrogram(torch.from_numpy(load_prompt_audio(sample.ref_audio).reshape(1, -1)), n_mels=128)
        for sample in samples
    ]
    report: dict[str, object] = {"references": len(mels), "frames_max": max(mel.shape[2] for mel in mels)}

    session = onnxruntime.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
    )
    names = [item.name for item in session.get_inputs()]
    codes: dict[str, list[list[int]]] = {"ort": []}
    seconds: dict[str, list[float]] = {"ort": []}
    for mel in mels:
        started = time.perf_counter()
        result = session.run(None, {names[0]: mel.numpy(), names[1]: np.array([mel.shape[2]], dtype=np.int32)})[0]
        seconds["ort"].append(time.perf_counter() - started)
        codes["ort"].append(result.flatten().tolist())
    del session

    free_before = torch.cuda.mem_get_info(device)[0]
    model = load_speech_tokenizer(model_path, device)
    free_after_load = torch.cuda.mem_get_info(device)[0]
    with torch.inference_mode():
        for allow_tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            name = f"eager_tf32={allow_tf32}"
            codes[name], seconds[name] = [], []
            for mel in mels:
                started = time.perf_counter()
                frames = torch.tensor([mel.shape[2]], device=device)
                result = model(mel.to(device), frames)[0, : conv_frames(conv_frames(mel.shape[2]))].tolist()
                seconds[name].append(time.perf_counter() - started)
                codes[name].append(result)
    for allow_tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.cuda.synchronize(device)
        free_before_capture = torch.cuda.mem_get_info(device)[0]
        started = time.perf_counter()
        graphs = SpeechTokenizerCudaGraphs(model, device)
        report[f"capture_s_tf32={allow_tf32}"] = round(time.perf_counter() - started, 2)
        report[f"graphs_gib_tf32={allow_tf32}"] = round((free_before_capture - torch.cuda.mem_get_info(device)[0]) / 2**30, 3)
        name = f"graph_tf32={allow_tf32}"
        codes[name], seconds[name] = [], []
        for mel in mels:
            started = time.perf_counter()
            codes[name].append(graphs.tokenize(mel))
            seconds[name].append(time.perf_counter() - started)
        del graphs
        torch.cuda.empty_cache()
    report["weights_gib"] = round((free_before - free_after_load) / 2**30, 3)
    report["ms_per_call"] = {
        name: {"mean": round(1000 * statistics.mean(values), 2), "p50": round(1000 * statistics.median(values), 2)}
        for name, values in seconds.items()
    }
    comparisons = {}
    for first, second in itertools.combinations(codes, 2):
        total = sum(len(row) for row in codes[first])
        comparisons[f"{first} vs {second}"] = {
            "references_differing": sum(a != b for a, b in zip(codes[first], codes[second])),
            "tokens_differing": sum(
                sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)) for a, b in zip(codes[first], codes[second])
            ),
            "tokens": total,
            "length_mismatches": sum(len(a) != len(b) for a, b in zip(codes[first], codes[second])),
        }
    report["comparisons"] = comparisons
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
