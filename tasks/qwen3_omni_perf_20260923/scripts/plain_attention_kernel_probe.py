"""The code predictor's plain-path attention, main's SDPA call against the branch's in-place
decode_attention, in one process on the same inputs.

Inputs as the plain path holds them: q a slice of the packed qkv rows, the caches slot major
with the talker's max batch, sliced to the batch. Per signature: graph-timed us for both, and
each one's relative L2 to an fp32 SDPA. Per layer and frame: the opening pair plus the 14
single-token steps at ends 3 to 16.

usage: python plain_attention_kernel_probe.py BRANCH_TREE
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
SLOTS = 17
MAX_BATCH = 64
QKV_WIDTH = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def load(tree: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "predictor_layers_branch", tree / "sglang_omni/utils/predictor_layers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["predictor_layers_branch"] = module
    spec.loader.exec_module(module)
    return module


def graph_microseconds(fn, replays: int = 200) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    graph.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / replays


def sdpa_as_main(q, k_cache, v_cache, batch, tokens, end, upcast):
    """talker.py's plain path on main: heads-major views of q and the cached slots."""
    queries = q.reshape(batch, tokens, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    keys = k_cache[:batch, :end].transpose(1, 2)
    values = v_cache[:batch, :end].transpose(1, 2)
    if upcast:
        queries, keys, values = queries.float(), keys.float(), values.float()
    else:
        pass
    attended = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, is_causal=tokens > 1, enable_gqa=True
    )
    return attended.transpose(1, 2).reshape(batch * tokens, NUM_Q_HEADS * HEAD_DIM)


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference).norm() / reference.norm()).item()


def main() -> None:
    branch = load(pathlib.Path(sys.argv[1]))
    generator = torch.Generator(device="cuda").manual_seed(7)
    k_cache = torch.randn(
        MAX_BATCH, SLOTS, NUM_KV_HEADS, HEAD_DIM, device="cuda", generator=generator
    ).to(torch.bfloat16)
    v_cache = torch.randn(
        MAX_BATCH, SLOTS, NUM_KV_HEADS, HEAD_DIM, device="cuda", generator=generator
    ).to(torch.bfloat16)
    print(
        f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()} "
        f"{torch.cuda.get_device_name()}"
    )
    print("batch tokens end | us sdpa triton | relL2 sdpa triton")
    for batch in (1, 4, 16, 32, 64):
        totals = {"sdpa": 0.0, "triton": 0.0}
        signatures = [(2, 2)] + [(1, end) for end in range(3, SLOTS)]
        for tokens, end in signatures:
            rows = batch * tokens
            qkv = torch.randn(rows, QKV_WIDTH, device="cuda", generator=generator).to(
                torch.bfloat16
            )
            q = qkv[:, : NUM_Q_HEADS * HEAD_DIM]
            out_triton = torch.empty(
                rows, NUM_Q_HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            holder = {}

            def run_sdpa() -> None:
                holder["sdpa"] = sdpa_as_main(
                    q, k_cache, v_cache, batch, tokens, end, upcast=False
                )

            def run_triton() -> None:
                branch.decode_attention(
                    q=q,
                    tokens_per_row=tokens,
                    k_cache=k_cache[:batch],
                    v_cache=v_cache[:batch],
                    end=end,
                    attention_output=out_triton,
                )

            sdpa_us = graph_microseconds(run_sdpa)
            triton_us = graph_microseconds(run_triton)
            run_sdpa()
            run_triton()
            reference = sdpa_as_main(
                q, k_cache, v_cache, batch, tokens, end, upcast=True
            )
            torch.cuda.synchronize()
            totals["sdpa"] += sdpa_us
            totals["triton"] += triton_us
            print(
                f"{batch:5d} {tokens:6d} {end:3d} | {sdpa_us:6.2f} {triton_us:6.2f} | "
                f"{relative_error(holder['sdpa'], reference):.5f} "
                f"{relative_error(out_triton, reference):.5f}"
            )
        print(
            f"batch {batch:2d} per layer and frame: sdpa {totals['sdpa']:6.1f} us, "
            f"triton {totals['triton']:6.1f} us"
        )


if __name__ == "__main__":
    main()
