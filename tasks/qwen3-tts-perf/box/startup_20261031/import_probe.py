# Startup item 2 and 5 probe (10-31): in one fresh process, the seconds of each import step a Qwen3-TTS stage process
# takes, in order (the CLI, the Qwen3-TTS stage modules, SGLang's model registry, then every omni model class the
# engine's register_omni_model imports), the live object count after each, and the seconds of one full gc.collect()
# at the end, the collection SGLang's freeze_gc runs before and after each CUDA graph capture.
# usage: python3 import_probe.py
import gc
import importlib
import importlib.util
import time
from pathlib import Path


def step(name, action):
    began = time.perf_counter()
    action()
    print(f"{name}: {time.perf_counter() - began:.2f} s, objects {len(gc.get_objects())}", flush=True)


def omni_model_imports():
    # the registry's entries read as text, so the probe imports nothing before its steps
    source = Path(importlib.util.find_spec("sglang_omni").origin).parent / "model_runner" / "sglang_model_runner.py"
    paths = [line.split('"')[3].split(":")[0] for line in source.read_text().splitlines() if '": "sglang_omni.' in line]
    for path in paths:
        began = time.perf_counter()
        try:
            importlib.import_module(path)
        except Exception as error:
            print(f"  {path} failed: {error}", flush=True)
        else:
            pass
        print(f"  {path}: {time.perf_counter() - began:.2f} s", flush=True)
    importlib.import_module("sglang_omni.models.ming_omni.registration")


began = time.perf_counter()
step("sglang_omni.cli", lambda: importlib.import_module("sglang_omni.cli"))
step("qwen3_tts stage modules", lambda: [importlib.import_module(name) for name in (
    "sglang_omni.models.qwen3_tts.engine_builder", "sglang_omni.models.qwen3_tts.streaming_vocoder",
    "sglang_omni.models.qwen3_tts.stages")])
step("sglang.srt.models.registry", lambda: importlib.import_module("sglang.srt.models.registry"))
step("omni model classes", omni_model_imports)
for attempt in range(3):
    step(f"gc.collect {attempt}", gc.collect)
gc.freeze()
step("gc.collect after gc.freeze", gc.collect)
print(f"total {time.perf_counter() - began:.2f} s")
