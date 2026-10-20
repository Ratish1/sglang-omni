# SPDX-License-Identifier: Apache-2.0
"""L3 forensics (10-20): served requests hit the token cap with the speech tokenizer replayed from CUDA graphs, while
the same model matched ONNX Runtime on every reference alone. Put this directory first on PYTHONPATH with
TOKENIZER_PROBE=<log path>. At construction it logs torch's default dtype, the model's parameter dtype and the matmul
precision; per call it logs the graph's codes against the same model run eagerly at the mel's own width and against
an ONNX Runtime CPU session of the same file, as mismatching token counts. No runtime code changes."""

import importlib.abc
import importlib.util
import json
import os
import sys
import threading

TARGET = "sglang_omni.models.fun_cosyvoice3.utils"


def patch(module) -> None:
    import numpy as np
    import onnxruntime
    import torch

    from sglang_omni.models.fun_cosyvoice3 import speech_tokenizer

    path = os.environ["TOKENIZER_PROBE"]
    write_lock = threading.Lock()

    def log(record: dict) -> None:
        with write_lock, open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")

    graphs_init = speech_tokenizer.SpeechTokenizerCudaGraphs.__init__

    def graphs_init_logged(self, model, device):
        log(
            {
                "event": "construct",
                "default_dtype": str(torch.get_default_dtype()),
                "parameter_dtype": str(next(model.parameters()).dtype),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "thread": threading.current_thread().name,
            }
        )
        graphs_init(self, model, device)
        self.probe_model = model

    speech_tokenizer.SpeechTokenizerCudaGraphs.__init__ = graphs_init_logged
    tokenizer_init = module.SpeechTokenizerV3.__init__

    def tokenizer_init_logged(self, model_path, *args, **kwargs):
        tokenizer_init(self, model_path, *args, **kwargs)
        self.probe_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    module.SpeechTokenizerV3.__init__ = tokenizer_init_logged
    tokenize = speech_tokenizer.SpeechTokenizerCudaGraphs.tokenize
    owners: dict[int, object] = {}

    def tokenize_checked(self, mel):
        codes = tokenize(self, mel)
        frames = mel.shape[2]
        model = self.probe_model
        with torch.inference_mode():
            eager = model(mel.to(next(model.parameters()).device), torch.tensor([frames], device=next(model.parameters()).device))
        eager = eager[0, : len(codes)].tolist()
        session = owners.get(id(self))
        reference = None
        if session is not None:
            names = [item.name for item in session.get_inputs()]
            reference = session.run(None, {names[0]: mel.numpy(), names[1]: np.array([frames], dtype=np.int32)})[0]
            reference = reference.flatten().tolist()
        log(
            {
                "event": "tokenize",
                "frames": frames,
                "codes": len(codes),
                "graph_vs_eager": sum(a != b for a, b in zip(codes, eager)),
                "graph_vs_ort": None if reference is None else sum(a != b for a, b in zip(codes, reference)) + abs(len(codes) - len(reference)),
                "eager_vs_ort": None if reference is None else sum(a != b for a, b in zip(eager, reference)),
                "stream": torch.cuda.current_stream().cuda_stream,
                "thread": threading.current_thread().name,
            }
        )
        return codes

    speech_tokenizer.SpeechTokenizerCudaGraphs.tokenize = tokenize_checked
    extract = module.SpeechTokenizerV3.extract_speech_token

    def extract_checked(self, audio, sample_rate=16000):
        if self.cuda_graphs is not None:
            owners[id(self.cuda_graphs)] = self.probe_session
        else:
            pass
        return extract(self, audio, sample_rate)

    module.SpeechTokenizerV3.extract_speech_token = extract_checked
    print(f"tokenizer probe patched {module.__name__}", file=sys.stderr, flush=True)


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != TARGET or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("TOKENIZER_PROBE"):
    sys.meta_path.insert(0, PatchOnImport())
