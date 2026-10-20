# SPDX-License-Identifier: Apache-2.0
"""Is the speech tokenizer deterministic? The accuracy run (acc_reference_c20261011.py, frontend) found the
prompt speech tokens of two SeedTTS references differing between omni's served request and CosyVoice's
frontend, in one tree's run and not in the other's. Per reference of the first 16 SeedTTS English samples,
the v3 tokenizer (speech_tokenizer_v3.onnx) runs REPEATS times in each configuration:
  omni      SpeechTokenizerV3 (CUDA EP, cudnn_conv_algo_search HEURISTIC, utils.py:61-71)
  cosyvoice CosyVoice's frontend session (CUDA EP, default EXHAUSTIVE search, cli/frontend.py:45-47)
  cpu       the same model on the CPU EP
and prints the distinct outputs per configuration and the positions where each differs from the CPU's.

    python3 tok_determinism_c20261011.py
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime
import torch
import whisper

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.request_builders import load_prompt_audio

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REPEATS = 10


def session(path: str, providers: list) -> onnxruntime.InferenceSession:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(path, sess_options=option, providers=providers)


def tokens(sess: onnxruntime.InferenceSession, audio: np.ndarray) -> tuple[int, ...]:
    feat = whisper.log_mel_spectrogram(torch.from_numpy(audio.reshape(1, -1)), n_mels=128)
    out = sess.run(
        None,
        {
            sess.get_inputs()[0].name: feat.numpy(),
            sess.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0]
    return tuple(int(token) for token in out.flatten())


def main() -> None:
    path = os.path.join(stages.resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx")
    sessions = {
        "omni": session(path, [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]),
        "cosyvoice": session(path, ["CUDAExecutionProvider"]),
        "cpu": session(path, ["CPUExecutionProvider"]),
    }
    seen = set()
    for sample in load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")[:16]:
        if sample.ref_audio in seen:
            continue
        seen.add(sample.ref_audio)
        audio = load_prompt_audio(sample.ref_audio)
        cpu = tokens(sessions["cpu"], audio)
        line = [f"{os.path.basename(sample.ref_audio)[:28]:28s} {len(cpu):4d} tokens"]
        for name, sess in sessions.items():
            outputs = [tokens(sess, audio) for _ in range(REPEATS)]
            distinct = sorted(set(outputs), key=outputs.index)
            diffs = [sum(a != b for a, b in zip(output, cpu)) for output in distinct]
            line.append(f"{name}: {len(distinct)} distinct, positions off cpu {diffs}")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
