"""decode_attention before and after its q row stride became a constexpr, from two trees.

1. The fused path's launch (q contiguous, so Q_ROW_STRIDE equals q heads * head dim): the PTX
   both trees compile for each tokens_per_row must be identical, so Qwen3-TTS and the fused
   layers keep the kernel they run.
2. The plain path's q, a strided slice of the packed qkv rows: the old tree copies it
   contiguous first (as the plain path did), the new tree reads it in place. Outputs must be
   bit-identical; both are graph-timed at the predictor's batch sizes.

usage: python decode_attention_stride_probe.py OLD_TREE NEW_TREE
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
QKV_WIDTH = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def load(tree: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name, tree / "sglang_omni/utils/predictor_layers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compiled_ptx(module) -> list[str]:
    kernel = module.decode_attention_kernel
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        compiled = [
            value
            for device_cache in caches.values()
            for value in device_cache[0].values()
        ]
    else:
        compiled = [
            value for cache in kernel.cache.values() for value in cache.values()
        ]
    return [value.asm["ptx"] for value in compiled]


def normalized(ptx: str) -> list[str]:
    return [
        line
        for line in ptx.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]


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


def main() -> None:
    old_tree, new_tree = map(pathlib.Path, sys.argv[1:3])
    old, new = load(old_tree, "predictor_layers_old"), load(
        new_tree, "predictor_layers_new"
    )
    generator = torch.Generator(device="cuda").manual_seed(5)
    for tokens in (1, 2):
        rows = 4 * tokens
        q = torch.randn(
            rows, NUM_Q_HEADS * HEAD_DIM, device="cuda", generator=generator
        ).to(torch.bfloat16)
        cache = torch.randn(
            rows // tokens,
            SLOTS,
            NUM_KV_HEADS,
            HEAD_DIM,
            device="cuda",
            generator=generator,
        ).to(torch.bfloat16)
        out_old, out_new = torch.empty_like(q), torch.empty_like(q)
        old.decode_attention(
            q=q,
            tokens_per_row=tokens,
            k_cache=cache,
            v_cache=cache,
            end=tokens + 3,
            attention_output=out_old,
            shape=old.PredictorLayerShape(
                hidden_size=1024,
                head_dim=HEAD_DIM,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                intermediate_size=3072,
                split_qkv=1,
                split_hidden=1,
            ),
        )
        new.decode_attention(
            q=q,
            tokens_per_row=tokens,
            k_cache=cache,
            v_cache=cache,
            end=tokens + 3,
            attention_output=out_new,
        )
        torch.cuda.synchronize()
        print(
            f"fused-path launch tokens={tokens}: outputs bit-identical {torch.equal(out_old, out_new)}"
        )
    old_ptx, new_ptx = compiled_ptx(old), compiled_ptx(new)
    print(f"compiled variants old {len(old_ptx)} new {len(new_ptx)}")
    for index, (old_text, new_text) in enumerate(zip(sorted(old_ptx), sorted(new_ptx))):
        print(
            f"variant {index}: PTX identical {normalized(old_text) == normalized(new_text)} ({len(normalized(old_text))} lines)"
        )

    print(
        "plain path: old copies q out of the qkv rows then reads it, new reads the rows in place"
    )
    for batch in (1, 4, 16, 32, 64):
        for tokens, end in ((2, 2), (1, 9), (1, 16)):
            rows = batch * tokens
            qkv = torch.randn(rows, QKV_WIDTH, device="cuda", generator=generator).to(
                torch.bfloat16
            )
            q = qkv[:, : NUM_Q_HEADS * HEAD_DIM]
            k_cache = torch.randn(
                batch, SLOTS, NUM_KV_HEADS, HEAD_DIM, device="cuda", generator=generator
            ).to(torch.bfloat16)
            v_cache = torch.randn(
                batch, SLOTS, NUM_KV_HEADS, HEAD_DIM, device="cuda", generator=generator
            ).to(torch.bfloat16)
            out_old = torch.empty(
                rows, NUM_Q_HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            out_new = torch.empty_like(out_old)
            shape = old.PredictorLayerShape(
                hidden_size=1024,
                head_dim=HEAD_DIM,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                intermediate_size=3072,
                split_qkv=1,
                split_hidden=1,
            )

            def copy_then_read() -> None:
                old.decode_attention(
                    q=q.contiguous(),
                    tokens_per_row=tokens,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    end=end,
                    attention_output=out_old,
                    shape=shape,
                )

            def read_in_place() -> None:
                new.decode_attention(
                    q=q,
                    tokens_per_row=tokens,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    end=end,
                    attention_output=out_new,
                )

            copy_us = graph_microseconds(copy_then_read)
            place_us = graph_microseconds(read_in_place)
            copy_then_read()
            read_in_place()
            torch.cuda.synchronize()
            print(
                f"batch {batch:2d} tokens {tokens} end {end:2d}: copy+read {copy_us:6.2f} us, in place {place_us:6.2f} us, bit-identical {torch.equal(out_old, out_new)}"
            )


if __name__ == "__main__":
    main()
