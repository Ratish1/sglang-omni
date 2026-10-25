"""#3132 SIM forensics (10-25 evening): the speech tokenizer token by token, against an exact reference.

Every distinct seed-tts EN reference clip, loaded and turned into whisper's log mel exactly as the server's
CosyVoice3ReferenceEncodeHook and SpeechTokenizerV3.extract_speech_token do, is tokenized by:
  ort_cuda  main's SpeechTokenizerV3 on CUDA, as served (ONNX Runtime CUDA provider, TF32, HEURISTIC conv search);
  ort_cpu   the same class on ONNX Runtime's CPU provider, the published graph in fp32;
  fp64      the torch port of the same graph's weights (speech_tokenizer.py of the L3 tree) in float64, the exact
            reference;
  fp32      that port in fp32 with TF32 off; tf32  with TF32 matmuls and convs on.
The fp64 run also records each token's FSQ margin: over its 8 dimensions, the smallest distance of
|tanh(z) x FSQ_SCALE| to the 0.5 rounding boundary; a token can change under a rounding error only below its margin.
Saves, per reference content key (the server's cosyvoice3_reference_input_key, as the conditioning probe logs it),
the sample ids, every method's codes and the margins, so runs in other processes or on other cards compare offline
(tokenizer_margins_report.py).
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<main tree> python3 tokenizer_margins.py --l3-tree <tree> --out <file.pt>
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import torch
import whisper

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cosyvoice3_reference_input_key,
    load_prompt_audio,
)
from sglang_omni.models.fun_cosyvoice3.utils import SpeechTokenizerV3
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
SAMPLE_RATE = 16000


def load_port(l3_tree):
    spec = importlib.util.spec_from_file_location(
        "l3_speech_tokenizer", os.path.join(l3_tree, "sglang_omni/models/fun_cosyvoice3/speech_tokenizer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


@torch.inference_mode()
def port_codes(port, model, mel, device, dtype):
    frames = mel.shape[2]
    codes = model(mel.to(device=device, dtype=dtype), torch.tensor([frames], device=device))
    return codes[0, : port.conv_frames(port.conv_frames(frames))].tolist()


@torch.inference_mode()
def fp64_codes_and_margins(port, model, mel, device):
    captured = {}
    handle = model.project_down.register_forward_hook(lambda module, inputs, output: captured.setdefault("z", output))
    try:
        codes = port_codes(port, model, mel, device, torch.float64)
    finally:
        handle.remove()
    positions = len(codes)
    scaled = torch.tanh(captured["z"][0, :positions]) * port.FSQ_SCALE
    margins = (scaled.abs() - 0.5).abs().min(dim=-1).values
    return codes, margins.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3-tree", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    model_path = os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx")
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    references = {}
    for sample in samples:
        key = str(cosyvoice3_reference_input_key(sample.ref_audio))
        if key not in references:
            audio = load_prompt_audio(sample.ref_audio).reshape(1, -1)
            references[key] = {"samples": [], "audio": audio,
                               "mel": whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=128)}
        else:
            pass
        references[key]["samples"].append(sample.sample_id)
    print(f"{len(samples)} samples, {len(references)} distinct references", flush=True)
    timings = {}
    started = time.perf_counter()
    for name, tokenizer_device in (("ort_cuda", "cuda"), ("ort_cpu", "cpu")):
        tokenizer = SpeechTokenizerV3(model_path, device=tokenizer_device)
        for reference in references.values():
            reference[name] = tokenizer.extract_speech_token(reference["audio"], SAMPLE_RATE)[0].tolist()
        timings[name] = time.perf_counter() - started
        started = time.perf_counter()
        del tokenizer
    port = load_port(args.l3_tree)
    model = port.load_speech_tokenizer(model_path, device)
    set_tf32(False)
    model = model.double()
    for reference in references.values():
        reference["fp64"], reference["margins"] = fp64_codes_and_margins(port, model, reference["mel"], device)
    model = model.float()
    for name, tf32 in (("fp32", False), ("tf32", True)):
        set_tf32(tf32)
        for reference in references.values():
            reference[name] = port_codes(port, model, reference["mel"], device, torch.float32)
    set_tf32(False)
    timings["port"] = time.perf_counter() - started
    for reference in references.values():
        reference.pop("audio")
        reference.pop("mel")
    torch.save({"references": references, "card": os.environ.get("CUDA_VISIBLE_DEVICES"), "timings": timings,
                "versions": {"torch": torch.__version__, "onnxruntime": __import__("onnxruntime").__version__}},
               args.out)
    tokens = sum(len(reference["fp64"]) for reference in references.values())
    for name in ("ort_cuda", "ort_cpu", "fp32", "tf32"):
        wrong = sum(sum(a != b for a, b in zip(reference[name], reference["fp64"]))
                    + abs(len(reference[name]) - len(reference["fp64"])) for reference in references.values())
        print(f"{name} against fp64: {wrong} of {tokens} tokens", flush=True)
    print(f"saved {args.out} ({timings})", flush=True)


if __name__ == "__main__":
    main()
