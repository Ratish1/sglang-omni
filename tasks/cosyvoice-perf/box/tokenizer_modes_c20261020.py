"""The CUDA speech tokenizer alone, per session mode, over every seed-tts EN reference (10-20): where a call's time
goes (the CPU log mel against session.run, and the run lock's wait) and what each mode costs in device memory, one
thread and eight. Each mode runs in a fresh process (its own ORT arena), torch on one CPU thread as in the engine's
process (OMP_NUM_THREADS=1). Modes:
  main         main's session: per thread streams, no lock, no warm run
  unified      prerequisite 1: one stream, the run lock, a 30 s warm run
  unified_nospin   prerequisite 1 with session.intra_op.allow_spinning=0
  unified_1thread  prerequisite 1 with intra_op_num_threads=1
  pool2, pool8     main's session called through a pool of 2 or 8 worker threads the tokenizer owns, so each worker
                   keeps its stream and handles, warmed by that many concurrent 30 s runs (later on 10-20)
Each mode's tokens are compared with main's.
usage: CUDA_VISIBLE_DEVICES=<card> [TOKENIZER_MODES=a,b] python3 tokenizer_modes_c20261020.py <out dir> [mode]"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODES = ("main", "unified", "unified_nospin", "unified_1thread", "pool2", "pool8")
POOL_WORKERS = {"pool2": 2, "pool8": 8}
THREADS = 8
MAX_SECONDS = 30


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def run_mode(out: Path, mode: str) -> None:
    import numpy as np
    import onnxruntime
    import torch
    import whisper

    from benchmarks.dataset.seedtts import load_seedtts_samples
    from sglang_omni.models.fun_cosyvoice3.request_builders import load_prompt_audio
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    torch.set_num_threads(1)
    torch.cuda.init()
    free_start = torch.cuda.mem_get_info()[0]
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1 if mode == "unified_1thread" else 16
    if mode == "unified_nospin":
        option.add_session_config_entry("session.intra_op.allow_spinning", "0")
    else:
        pass
    cuda_options = {"cudnn_conv_algo_search": "HEURISTIC"}
    if mode.startswith("unified"):
        cuda_options["use_ep_level_unified_stream"] = "1"
    else:
        pass
    session = onnxruntime.InferenceSession(
        os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx"),
        sess_options=option,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )
    lock = threading.Lock() if mode.startswith("unified") else None
    workers = POOL_WORKERS.get(mode)
    pool = ThreadPoolExecutor(max_workers=workers) if workers else None
    names = [item.name for item in session.get_inputs()]

    def session_run(feeds: dict[str, np.ndarray]) -> np.ndarray:
        return session.run(None, feeds)[0]

    def tokenize(audio: np.ndarray) -> tuple[list[int], float, float, float]:
        started = time.perf_counter()
        feat = whisper.log_mel_spectrogram(torch.from_numpy(audio.reshape(1, -1)), n_mels=128)
        feeds = {names[0]: feat.numpy(), names[1]: np.array([feat.shape[2]], dtype=np.int32)}
        mel_done = time.perf_counter()
        if lock is not None:
            lock.acquire()
        else:
            pass
        locked = time.perf_counter()
        try:
            if pool is not None:
                result = pool.submit(session_run, feeds).result()
            else:
                result = session_run(feeds)
        finally:
            if lock is not None:
                lock.release()
            else:
                pass
        ended = time.perf_counter()
        return result.flatten().tolist(), mel_done - started, locked - mel_done, ended - locked

    after_session = torch.cuda.mem_get_info()[0]
    if mode.startswith("unified"):
        tokenize(np.zeros(MAX_SECONDS * 16000, dtype=np.float32))
    elif pool is not None:
        barrier = threading.Barrier(workers)
        longest = np.zeros((1, MAX_SECONDS * 16000), dtype=np.float32)
        feat = whisper.log_mel_spectrogram(torch.from_numpy(longest), n_mels=128)
        feeds = {names[0]: feat.numpy(), names[1]: np.array([feat.shape[2]], dtype=np.int32)}

        def warm_one(_: int) -> None:
            barrier.wait()
            session_run(feeds)

        list(pool.map(warm_one, range(workers)))
    else:
        pass
    after_warm = torch.cuda.mem_get_info()[0]
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    audios = [load_prompt_audio(sample.ref_audio) for sample in samples]
    tokenize(audios[0])
    report: dict[str, object] = {"mode": mode, "references": len(audios)}
    tokens = []
    for threads in (1, THREADS):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(tokenize, audios))
        wall = time.perf_counter() - started
        if threads == 1:
            tokens = [result[0] for result in results]
        else:
            pass
        mel, wait, run = ([1000 * result[index] for result in results] for index in (1, 2, 3))
        total = [m + w + r for m, w, r in zip(mel, wait, run)]
        report[f"threads={threads}"] = {
            "calls_per_s": round(len(audios) / wall, 1),
            "mel_ms_mean": round(statistics.mean(mel), 2),
            "wait_ms_mean": round(statistics.mean(wait), 2),
            "wait_ms_p95": round(quantile(wait, 0.95), 2),
            "run_ms_mean": round(statistics.mean(run), 2),
            "run_ms_p50": round(quantile(run, 0.5), 2),
            "run_ms_p95": round(quantile(run, 0.95), 2),
            "call_ms_mean": round(statistics.mean(total), 2),
            "call_ms_p95": round(quantile(total, 0.95), 2),
            "free_gib_after": round(torch.cuda.mem_get_info()[0] / 2**30, 3),
        }
    report["session_gib"] = round((free_start - after_session) / 2**30, 3)
    report["warm_gib"] = round((after_session - after_warm) / 2**30, 3)
    report["served_growth_gib"] = round((after_warm - torch.cuda.mem_get_info()[0]) / 2**30, 3)
    (out / f"{mode}.tokens.json").write_text(json.dumps(tokens))
    (out / f"{mode}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2:
        run_mode(out, sys.argv[2])
        return
    else:
        pass
    modes = os.environ.get("TOKENIZER_MODES", ",".join(MODES)).split(",")
    for mode in modes:
        rc = subprocess.run([sys.executable, __file__, str(out), mode], env={**os.environ, "OMP_NUM_THREADS": "1"}).returncode
        print(f"mode {mode} rc {rc}", flush=True)
    reference = json.loads((out / "main.tokens.json").read_text())
    for mode in modes:
        report = json.loads((out / f"{mode}.json").read_text())
        same = json.loads((out / f"{mode}.tokens.json").read_text()) == reference
        print(json.dumps({**report, "tokens_equal_main": same}))


if __name__ == "__main__":
    main()
