"""Independent PR #3553 checks; official reference files are pinned to 1116bea."""

import argparse
import concurrent.futures
import gc
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import httpx
import soundfile
import torch
from torch import nn

from sglang_omni.models.yue2.acoustic import AcousticChunk
from sglang_omni.models.yue2.layers import TokenCache, Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.runtime import Yue2Runtime

MODEL = "m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1"
VAE = "m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126"
DIRECTORY = Path("/data/pr3553-review-20261114")


def reference_module(filename: str):
    directory = Path(__file__).parent / "reference"
    if "pr3553_reference" not in sys.modules:
        specification = importlib.util.spec_from_file_location(
            "pr3553_reference",
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        module = importlib.util.module_from_spec(specification)
        sys.modules["pr3553_reference"] = module
        specification.loader.exec_module(module)
    return importlib.import_module(f"pr3553_reference.{Path(filename).stem}")


@torch.inference_mode()
def numerical() -> None:
    torch.manual_seed(42)
    results = {}
    for layers in (1, 2):
        model = (
            Yue2Model(
                Yue2ModelConfig(
                    hidden_size=16,
                    num_hidden_layers=layers,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    intermediate_size=32,
                    vocab_size=100,
                    latent_dim=64,
                    max_latent_frames=128,
                )
            )
            .cuda()
            .eval()
        )
        calls = []
        attention = model.model.layers[0].self_attn
        original = attention.attend

        def count_attend(*arguments, **keywords):
            calls.append(1)
            return original(*arguments, **keywords)

        attention.attend = count_attend
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        complete = TokenCache(capacity=4)
        model(tokens, complete)
        incremental = TokenCache(capacity=4)
        model(tokens[:, :3], incremental)
        model(tokens[:, 3:], incremental)
        errors = [
            float((left - right).abs().max())
            for left, right in zip(
                complete.keys + complete.values, incremental.keys + incremental.values
            )
        ]
        results[f"cache_{layers}_layers"] = {
            "attention_calls": len(calls),
            "max_absolute_error": max(errors),
        }
        assert max(errors) < 1e-5
    del model, complete, incremental
    gc.collect()
    torch.cuda.empty_cache()
    runtime = Yue2Runtime.from_pretrained(
        MODEL, VAE, torch.device("cuda:0"), vae_core_frames=512, vae_halo_frames=16
    )
    official = reference_module("modeling_yue2.py")
    configuration = SimpleNamespace(**runtime.model.config.model_dump())
    with torch.device("meta"):
        reference = nn.Module()
        reference.config = configuration
        reference.model = official.Backbone(configuration)
        reference.llm2vae = nn.Linear(
            configuration.hidden_size, configuration.latent_dim
        )
        reference.vae2llm = nn.Linear(
            configuration.latent_dim, configuration.hidden_size
        )
        reference.time_embedder = official.TimestepEmbedder(configuration.hidden_size)
        reference.latent_pos_embed = official.AudioPositionEmbedding(
            configuration.max_latent_frames, configuration.hidden_size
        )
    reference.load_state_dict(runtime.model.state_dict(), assign=True, strict=True)
    reference._shift_t_value = MethodType(
        official.YuE2ForCausalLM._shift_t_value, reference
    )
    reference.eval()
    prefix = [151643, 1, 2, 151847, 5, 6, 151848, 151851]
    codec = [151853 + index for index in range(16)]
    conditioning = prefix + codec + [151852]
    latent = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    chunk = AcousticChunk(runtime.model, conditioning, latent.cpu().float())
    nar = reference_module("nar.py")
    cached_reference = nar.CachedNAR(
        reference, nar.Chunk(conditioning, latent.cpu().float())
    )
    tokens = torch.tensor(
        [conditioning + [184621] + [184623] * 16 + [184622]], device="cuda"
    )
    ar_mask = torch.arange(tokens.shape[1], device="cuda")[None] < len(conditioning)
    nar_mask = ~ar_mask
    content_mask = nar_mask.clone()
    content_mask[:, len(conditioning)] = False
    content_mask[:, -1] = False
    for raw_time in (0.0, 3.0):
        actual = chunk.predict_velocity(
            latent, torch.tensor(raw_time, dtype=torch.bfloat16, device="cuda")
        )
        expected = official.YuE2ForCausalLM.nar_velocity(
            reference, tokens, ar_mask, nar_mask, content_mask, latent, raw_time
        )
        relative = float(
            (actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt()
        )
        results[f"dense_nar_time_{raw_time}"] = {
            "relative_rms": relative,
            "max_absolute_error": float((actual - expected).abs().max()),
        }
        cached_expected = cached_reference.velocity(latent, raw_time)
        results[f"cached_nar_time_{raw_time}"] = {
            "relative_rms": float(
                (actual.float() - cached_expected.float()).square().mean().sqrt()
                / cached_expected.float().square().mean().sqrt()
            ),
            "max_absolute_error": float((actual - cached_expected).abs().max()),
        }
        print(json.dumps(results, indent=2), flush=True)
        torch.testing.assert_close(actual, cached_expected, rtol=0, atol=0)
    reference_vae = reference_module("modeling_vae.py")
    with torch.device("meta"):
        decoder = reference_vae.OobleckDecoder(
            **runtime.vae.config.decoder_config.model_dump()
        )
    decoder.load_state_dict(runtime.vae.decoder.state_dict(), assign=True)
    decoder.eval()
    latents = torch.randn(1, 64, 33, device="cuda") * 0.1
    full = decoder(latents).cpu().clamp(-1, 1)
    tiled = runtime.vae.decode_tiled(latents.cpu(), 13, 16, lambda: False)
    results["vae_full_vs_tiled"] = {
        "shape": list(tiled.shape),
        "max_absolute_error": float((full - tiled).abs().max()),
    }
    torch.testing.assert_close(full, tiled, rtol=1e-4, atol=2e-6)
    print(json.dumps(results, indent=2), flush=True)


def request_body(seed: int, cot: str = "off") -> dict:
    return {
        "model": MODEL,
        "input": "[Verse]\nMorning light is shining through",
        "instructions": "Gentle acoustic pop, warm female vocals",
        "response_format": "wav",
        "seed": seed,
        "max_new_tokens": 16,
        "stage_params": {
            "synthesis": {"cot": cot, "abc_max_tokens": 8, "ode_steps": 2}
        },
    }


def serving(
    priority: bool,
    policy: str,
    kv_cache_bytes: str | None = None,
    prefill_max_requests: int | None = None,
    natural_stop: bool = False,
    pressure: bool = False,
    events: bool = False,
) -> None:
    port = 18353
    arguments = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--yue2_acoustic.factory.vae_path",
        VAE,
        "--yue2_ar.factory.max_concurrency",
        "4",
    ]
    if priority:
        arguments += ["--yue2_ar.engine.enable_priority_scheduling", "true"]
    if policy != "default":
        arguments += ["--yue2_ar.engine.schedule_policy", policy]
    if kv_cache_bytes is not None:
        arguments += ["--yue2_ar.engine.kv_cache_bytes", kv_cache_bytes]
    if prefill_max_requests is not None:
        arguments += [
            "--yue2_ar.engine.prefill_max_requests",
            str(prefill_max_requests),
        ]
    label = (
        "priority"
        if priority
        else (
            "bytes"
            if kv_cache_bytes is not None
            else ("prefill" if prefill_max_requests is not None else policy)
        )
    )
    if natural_stop:
        label = "natural-stop"
    if events:
        label = "events"
    if pressure:
        label = "pressure"
        arguments += [
            "--yue2_ar.engine.max_total_tokens",
            "4096",
            "--yue2_ar.engine.max_running_requests",
            "16",
        ]
    log = (DIRECTORY / f"server-{label}.log").open("w")
    process = subprocess.Popen(
        arguments, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    base = f"http://127.0.0.1:{port}"
    results = {"priority": priority, "policy": policy, "pid": process.pid}
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited {process.returncode}; see server-{label}.log"
                )
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Server startup timed out")

        def send(body: dict) -> dict:
            started = time.monotonic()
            try:
                response = httpx.post(
                    f"{base}/v1/audio/speech",
                    json=body,
                    timeout=600 if natural_stop or pressure else 30,
                )
            except httpx.TimeoutException:
                return {"status": "timeout", "seconds": time.monotonic() - started}
            result = {
                "status": response.status_code,
                "seconds": time.monotonic() - started,
            }
            if response.status_code == 200:
                audio, rate = soundfile.read(
                    io.BytesIO(response.content), always_2d=True
                )
                result.update(
                    shape=list(audio.shape),
                    rate=rate,
                    finish=response.headers.get("X-Finish-Reason"),
                )
            else:
                result["body"] = response.text[:1500]
            return result

        if events:
            response = httpx.post(
                f"{base}/start_request_profile",
                json={
                    "run_id": "pr3553-events",
                    "event_dir": str(DIRECTORY / "events"),
                },
                timeout=30,
            )
            response.raise_for_status()
            time.sleep(1)
            results["full"] = send(request_body(72, "full"))
            results["off"] = send(request_body(73, "off"))
            response = httpx.post(f"{base}/stop_request_profile", json={}, timeout=30)
            response.raise_for_status()
            return
        if pressure:
            bodies = [request_body(seed) for seed in range(60, 66)]
            for body in bodies:
                body["max_new_tokens"] = 512
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                results["pressure"] = list(executor.map(send, bodies))
            return
        body = request_body(42)
        if natural_stop:
            body["max_new_tokens"] = 9000
            body["stage_params"]["synthesis"] = {"cot": "full", "ode_steps": 32}
        results["single"] = send(body)
        if natural_stop:
            return
        if results["single"]["status"] == 200:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results["concurrent"] = list(
                    executor.map(
                        send,
                        [
                            request_body(43, "full"),
                            request_body(44, "melody"),
                            request_body(45),
                            request_body(46),
                        ],
                    )
                )
            invalid = request_body(42)
            invalid.pop("instructions")
            results["missing_style"] = send(invalid)
            invalid = request_body(42)
            invalid["max_new_tokens"] = 25000
            results["over_context"] = send(invalid)
            try:
                cancelled = request_body(47)
                cancelled["max_new_tokens"] = 2000
                httpx.post(f"{base}/v1/audio/speech", json=cancelled, timeout=0.2)
            except httpx.TimeoutException:
                results["disconnect"] = "timeout sent"
            results["after_disconnect"] = send(request_body(48))
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        log.close()
        results["server_exit"] = process.returncode
        if events:
            intervals = {}
            for path in (DIRECTORY / "events").glob("events_*.jsonl"):
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event["event_name"] in {"model_path_start", "model_path_end"}:
                        row = intervals.setdefault(
                            event["request_id"], {"stage": event["stage"], "events": []}
                        )
                        row["events"].append(event["event_name"])
            results["intervals"] = intervals
        (DIRECTORY / f"http-{label}.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["numerical", "serving"])
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--policy", default="default")
    parser.add_argument("--kv-cache-bytes")
    parser.add_argument("--prefill-max-requests", type=int)
    parser.add_argument("--natural-stop", action="store_true")
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--events", action="store_true")
    options = parser.parse_args()
    DIRECTORY.mkdir(exist_ok=True, parents=True)
    if options.mode == "numerical":
        numerical()
    else:
        serving(
            options.priority,
            options.policy,
            options.kv_cache_bytes,
            options.prefill_max_requests,
            options.natural_stop,
            options.pressure,
            options.events,
        )
