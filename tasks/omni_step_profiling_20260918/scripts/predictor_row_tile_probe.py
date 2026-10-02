"""Row tile probe for the fused predictor layers: what each batch size compiles to and costs.

For every batch: the Qwen3-TTS predictor at the checkpoint's shapes (5 layers, the test
fixtures of tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py), plain and fused, runs
the opening pair (2 rows per request) and one single-token pass; records success or the
exception, the median time of each pass, and whether request 0's fused outputs and cache
rows equal the batch-1 run bit for bit. Then every compiled kernel's BLOCK_M, registers,
spills and shared memory, against the device's opt-in shared memory per block.

usage: PYTHONPATH=<tree> python predictor_row_tile_probe.py 1,8,16,32,64,65,128 [tts|omni]
"""

from __future__ import annotations

import statistics
import sys

import torch


def time_ms(fn, inputs) -> float:
    times = []
    for tokens in inputs:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(tokens)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times[3:])


def graph_step_ms(talker, batch: int, pair, single, max_fused_rows=None) -> float:
    """One served predictor step, the opening pair then 14 single-token passes,
    captured in a CUDA graph; the median replay time. With max_fused_rows, a fused
    talker runs a pass plain when its rows exceed it."""
    fused = talker.predictor_fused_layers

    def run(tokens, cache_len):
        rows = tokens.shape[0] * tokens.shape[1]
        if max_fused_rows is not None and rows > max_fused_rows:
            talker.predictor_fused_layers = None
        else:
            talker.predictor_fused_layers = fused
        talker.predictor_forward_tokens(
            token_embeds=tokens, batch_size=batch, cache_len=cache_len
        )
        talker.predictor_fused_layers = fused

    def step():
        run(pair, 0)
        for cache_len in range(2, 16):
            run(single, cache_len)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    times = []
    for _ in range(23):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times[3:])


def kernel_resources(predictor_kernels) -> None:
    device = torch.cuda.current_device()
    optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    print(f"device opt-in shared memory per block: {optin} bytes")
    for name in (
        "norm_qkv_rope_store_kernel",
        "gemv_add_kernel",
        "norm_gate_up_silu_kernel",
    ):
        fn = getattr(predictor_kernels, name)
        rows = []
        for entry in getattr(fn, "device_caches", {}).values():
            for kernel in entry[0].values():
                try:
                    kernel._init_handles()
                except Exception:  # noqa: BLE001
                    pass
                constants = {}
                for key, value in kernel.src.constants.items():
                    index = key[0] if isinstance(key, tuple) else key
                    constants[
                        fn.arg_names[index] if isinstance(index, int) else index
                    ] = value
                rows.append(
                    (
                        constants.get("BLOCK_M"),
                        constants.get("N"),
                        getattr(kernel, "n_regs", "not loaded"),
                        getattr(kernel, "n_spills", "not loaded"),
                        kernel.metadata.shared,
                    )
                )
        for block_m, n, regs, spills, shared in sorted(
            rows, key=lambda r: (r[0], r[1] or 0)
        ):
            print(
                f"  {name} BLOCK_M={block_m} N={n}: regs {regs}, spills {spills}, "
                f"shared {shared}"
            )


def main() -> None:
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.runtime_context import get_context

    import sglang_omni.models.qwen3_tts.sglang_model as tts_model
    from sglang_omni.models.qwen3_omni.components import predictor_kernels
    from sglang_omni.vendor.sglang.models import apply_qk_norm
    from tests.unit_test.qwen3_tts import test_predictor_cuda_graph as tts

    tts_model.apply_qk_norm = apply_qk_norm
    batches = [int(b) for b in sys.argv[1].split(",")]
    assert batches[0] == 1, "batch 1 is the bit reference"
    device = torch.device("cuda")
    torch.manual_seed(0)
    layers = [tts.real_shape_layer(device) for _ in range(tts.FUSED_LAYERS)]
    final_norm = tts.real_shape_norm(tts.FUSED_HIDDEN, device)
    generator = torch.Generator(device=device).manual_seed(1)
    largest = max(batches)
    pair = torch.randn(largest, 2, tts.FUSED_HIDDEN, device=device, generator=generator)
    single = torch.randn(
        largest, 1, tts.FUSED_HIDDEN, device=device, generator=generator
    )
    pair, single = pair.to(tts.DTYPE), single.to(tts.DTYPE)
    model = sys.argv[2] if len(sys.argv) > 2 else "tts"
    if model == "omni":
        from tests.unit_test.qwen3_omni import test_predictor_kernels as omni

        def build(batch, fused):
            sized = {"max_bs": batch} if batch > omni.MAX_BS else {}
            talker = omni.build_talker(device, seed=0, **sized)
            return omni.fuse(talker) if fused else talker

    else:

        def build(batch, fused):
            tts.FUSED_MAX_BS = batch
            return tts.real_shape_talker(device, layers, final_norm, fused=fused)

    reference = None
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        for batch in batches:
            line = [f"batch {batch:4d} (pair rows {2 * batch:4d})"]
            for path in ("plain", "fused"):
                try:
                    talker = build(batch, path == "fused")

                    def run_pair(tokens):
                        return talker.predictor_forward_tokens(
                            token_embeds=tokens, batch_size=batch, cache_len=0
                        )

                    def run_single(tokens):
                        return talker.predictor_forward_tokens(
                            token_embeds=tokens, batch_size=batch, cache_len=2
                        )

                    out_pair = run_pair(pair[:batch].clone()).clone()
                    out_single = run_single(single[:batch].clone()).clone()
                    k_row = talker.predictor_k_cache[:, :1, :3].clone()
                    v_row = talker.predictor_v_cache[:, :1, :3].clone()
                    torch.cuda.synchronize()
                    pair_ms = time_ms(
                        run_pair, [pair[:batch].clone() for _ in range(33)]
                    )
                    single_ms = time_ms(
                        run_single, [single[:batch].clone() for _ in range(33)]
                    )
                    step_ms = graph_step_ms(
                        talker, batch, pair[:batch].clone(), single[:batch].clone()
                    )
                    line.append(
                        f"{path} pair {pair_ms:.3f} ms single {single_ms:.3f} ms "
                        f"graph step {step_ms:.3f} ms"
                    )
                    if path == "fused":
                        rule_ms = graph_step_ms(
                            talker,
                            batch,
                            pair[:batch].clone(),
                            single[:batch].clone(),
                            max_fused_rows=64,
                        )
                        line.append(
                            f"fused within 64 rows: graph step {rule_ms:.3f} ms"
                        )
                    else:
                        pass
                    if path == "fused":
                        rows = (out_pair[:1], out_single[:1], k_row, v_row)
                        if reference is None:
                            reference = rows
                        else:
                            same = all(
                                torch.equal(a, b) for a, b in zip(rows, reference)
                            )
                            line.append(f"request 0 bit equal to batch 1: {same}")
                    else:
                        pass
                except Exception as error:  # noqa: BLE001
                    message = str(error).strip().splitlines()
                    line.append(
                        f"{path} FAILED {type(error).__name__}: "
                        f"{message[-1][:160] if message else ''}"
                    )
                    torch.cuda.synchronize()
            print(" | ".join(line), flush=True)
    kernel_resources(predictor_kernels)


if __name__ == "__main__":
    main()
