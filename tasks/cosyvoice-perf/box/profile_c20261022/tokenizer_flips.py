"""Per reference speech tokens of every seed-tts EN sample from three tokenizers: ONNX Runtime's CUDA session with
main's options, the served CUDA graphs (L3) and the eager torch model in fp32 (TF32 matmuls off). Writes, per sample
id, the tokens each pair of methods disagrees on, for joining with per clip WER and SIM.
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<tree with L3>:<CosyVoice dirs> python3 tokenizer_flips.py <out json>"""

import json
import os
import sys

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


def differing(first, second):
    return sum(x != y for x, y in zip(first, second)) + abs(len(first) - len(second))


def main() -> None:
    device = torch.device("cuda", 0)
    model_path = os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx")
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    mels = [
        whisper.log_mel_spectrogram(torch.from_numpy(load_prompt_audio(sample.ref_audio).reshape(1, -1)), n_mels=128)
        for sample in samples
    ]
    session = onnxruntime.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
    )
    names = [item.name for item in session.get_inputs()]
    ort = [
        session.run(None, {names[0]: mel.numpy(), names[1]: np.array([mel.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        for mel in mels
    ]
    del session
    model = load_speech_tokenizer(model_path, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        fp32 = [
            model(mel.to(device), torch.tensor([mel.shape[2]], device=device))[0, : conv_frames(conv_frames(mel.shape[2]))].tolist()
            for mel in mels
        ]
    graphs = SpeechTokenizerCudaGraphs(model, device)
    served = [graphs.tokenize(mel) for mel in mels]
    rows = {
        sample.sample_id: {
            "tokens": len(fp32_codes),
            "ort_vs_graph": differing(ort_codes, graph_codes),
            "ort_vs_fp32": differing(ort_codes, fp32_codes),
            "graph_vs_fp32": differing(graph_codes, fp32_codes),
        }
        for sample, ort_codes, fp32_codes, graph_codes in zip(samples, ort, fp32, served)
    }
    with open(sys.argv[1], "w") as handle:
        json.dump(rows, handle, indent=1)
    for key in ("ort_vs_graph", "ort_vs_fp32", "graph_vs_fp32"):
        print(key, sum(row[key] for row in rows.values()), "tokens in", sum(row[key] > 0 for row in rows.values()), "references")


if __name__ == "__main__":
    main()
