"""Hash every tensor the Qwen3-Omni preprocessor hands downstream, one JSON line per request.

Enabled by OMNI_PREPROC_HASH=<output prefix>. A watcher thread waits for the preprocessor module to load, then wraps
Qwen3OmniPreprocessor.finalize_state. Each call writes the request id, a hash of its input ids (the key that names the
question across servers), and the shape and sha256 of every tensor in its multimodal and encoder inputs, to
<prefix>.<pid>.jsonl. The wrapper reads only; the state it returns is the original's. OMNI_PREPROC_DUMP_KEYS=<key,key>
also saves those requests' input ids, attention mask and multimodal inputs to <prefix>.<key>.pt.
"""

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time

PREFIX = os.environ.get("OMNI_PREPROC_HASH")
DUMP_KEYS = set(filter(None, os.environ.get("OMNI_PREPROC_DUMP_KEYS", "").split(",")))
MODULE = "sglang_omni.models.qwen3_omni.components.preprocessor"


def tensor_entries(value, path: str, out: dict) -> None:
    torch = sys.modules["torch"]
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
        out[path] = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "sha": hashlib.sha256(raw).hexdigest()[:24],
        }
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            tensor_entries(value[key], f"{path}.{key}", out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensor_entries(item, f"{path}[{index}]", out)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            tensor_entries(getattr(value, field.name), f"{path}.{field.name}", out)
    else:
        pass


def install(module) -> None:
    lock = threading.Lock()
    path = f"{PREFIX}.{os.getpid()}.jsonl"
    original = module.Qwen3OmniPreprocessor.finalize_state

    def finalize_state(self, payload, **kwargs):
        torch = sys.modules["torch"]
        tensors: dict = {}
        tensor_entries(kwargs.get("full_mm_inputs"), "mm", tensors)
        tensor_entries(kwargs.get("encoder_inputs"), "enc", tensors)
        input_ids = kwargs["input_ids"].detach().cpu().contiguous().reshape(-1)
        record = {
            "request_id": payload.request_id,
            "time": time.time(),
            "thread": threading.current_thread().name,
            "key": hashlib.sha256(
                input_ids.view(torch.uint8).numpy().tobytes()
            ).hexdigest()[:24],
            "prompt_tokens": int(input_ids.numel()),
            "tensors": tensors,
        }
        with lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if record["key"] in DUMP_KEYS:
            torch.save(
                {
                    "input_ids": kwargs["input_ids"],
                    "attention_mask": kwargs["attention_mask"],
                    "full_mm_inputs": kwargs["full_mm_inputs"],
                    "record": record,
                },
                f"{PREFIX}.{record['key']}.pt",
            )
        else:
            pass
        return original(self, payload, **kwargs)

    module.Qwen3OmniPreprocessor.finalize_state = finalize_state


def watch() -> None:
    while MODULE not in sys.modules or not hasattr(
        sys.modules[MODULE], "Qwen3OmniPreprocessor"
    ):
        time.sleep(0.2)
    install(sys.modules[MODULE])


if PREFIX:
    threading.Thread(target=watch, name="preproc-hash-probe", daemon=True).start()
else:
    pass
