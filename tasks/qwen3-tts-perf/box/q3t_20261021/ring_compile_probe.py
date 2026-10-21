"""Can a compiled vocoder decode hold the K/V ring step (state_in_place_20261021/PLAN.md 2, risk 1)?

usage: python3 ring_compile_probe.py --card 2

One decoder layer's ring step at Qwen3-TTS's shapes (16 q and kv heads, head dim 64, ring 79 frames, 8 rows at
width 8): the rope kernel rotates q and k and stores k and v in the rows' ring rows, then FA3 attends each row's
window through the ring's page table. Runs it eagerly, under torch.compile(fullgraph=True), and the compiled step
inside a CUDA graph, and prints whether each runs and how far each is from the eager output.
"""

import argparse
import traceback

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.rope import apply_rope_inplace_with_kvcache

HEADS = 16
HEAD_DIM = 64
RETAINED = 71
RING = 79


def ring_inputs(device: torch.device, rows: int, width: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(7)
    tokens = rows * width
    slots = torch.arange(rows, device=device) * 2 + 1
    starts = torch.tensor([0, 1, 40, 70, 71, 72, 150, 400][:rows], device=device)
    frame_positions = starts[:, None] + torch.arange(width, device=device)
    window_starts = (starts - RETAINED).clamp(min=0)
    window_positions = window_starts[:, None] + torch.arange(RING, device=device)
    angles = frame_positions.reshape(tokens, 1).float() * (
        1.0 / 10000 ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM)
    )
    return {
        "q": torch.randn(tokens, HEADS, HEAD_DIM, device=device, generator=generator).bfloat16(),
        "k": torch.randn(tokens, HEADS, HEAD_DIM, device=device, generator=generator).bfloat16(),
        "v": torch.randn(tokens, HEADS * HEAD_DIM, device=device, generator=generator).bfloat16(),
        "key_ring": torch.randn(
            (2 * rows + 1) * RING, HEADS * HEAD_DIM, device=device, generator=generator
        ).bfloat16(),
        "value_ring": torch.randn(
            (2 * rows + 1) * RING, HEADS * HEAD_DIM, device=device, generator=generator
        ).bfloat16(),
        "cos_sin": torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous(),
        "positions": torch.arange(tokens, device=device),
        "write_rows": (slots[:, None] * RING + frame_positions % RING).reshape(-1),
        "page_table": (slots[:, None] * RING + window_positions % RING).to(torch.int32),
        "lengths": (starts - window_starts + width).to(torch.int32),
        "rows": rows,
        "width": width,
    }


def ring_step(q, k, v, key_ring, value_ring, cos_sin, positions, write_rows, page_table, lengths, rows, width):
    apply_rope_inplace_with_kvcache(q, k, v, key_ring, value_ring, cos_sin, positions, write_rows, is_neox=True)
    return flash_attn_with_kvcache(
        q.view(rows, width, HEADS, HEAD_DIM),
        key_ring.view(-1, 1, HEADS, HEAD_DIM),
        value_ring.view(-1, 1, HEADS, HEAD_DIM),
        page_table=page_table,
        cache_seqlens=lengths,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        window_size=(RETAINED, 0),
    )


def ring_step_sdpa(q, k, v, key_ring, value_ring, cos_sin, positions, write_rows, page_table, lengths, rows, width):
    apply_rope_inplace_with_kvcache(q, k, v, key_ring, value_ring, cos_sin, positions, write_rows, is_neox=True)
    window_rows = page_table.long()
    keys = key_ring[window_rows].view(rows, RING, HEADS, HEAD_DIM).transpose(1, 2)
    values = value_ring[window_rows].view(rows, RING, HEADS, HEAD_DIM).transpose(1, 2)
    query_offsets = (lengths.long() - width)[:, None] + torch.arange(width, device=q.device)
    distances = query_offsets[:, :, None] - torch.arange(RING, device=q.device)
    window_mask = ((distances >= 0) & (distances <= RETAINED))[:, None]
    return torch.nn.functional.scaled_dot_product_attention(
        q.view(rows, width, HEADS, HEAD_DIM).transpose(1, 2), keys, values, attn_mask=window_mask,
        scale=HEAD_DIM**-0.5,
    ).transpose(1, 2)


def fresh(inputs: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", args.card)
    torch.cuda.set_device(device)
    inputs = ring_inputs(device, rows=8, width=8)
    with torch.inference_mode():
        eager_inputs = fresh(inputs)
        expected = ring_step(**eager_inputs).clone()
        print(f"eager FA3: ok, output {tuple(expected.shape)}", flush=True)
        gathered = ring_step_sdpa(**fresh(inputs))
        print(
            f"eager gathered window and SDPA: max abs difference from FA3 "
            f"{(gathered.float() - expected.float()).abs().max().item():.3e}",
            flush=True,
        )
        compiled = None
        for name, step in (("FA3", ring_step), ("gathered window and SDPA", ring_step_sdpa)):
            candidate = torch.compile(step, fullgraph=True, dynamic=False)
            try:
                compiled_inputs = fresh(inputs)
                actual = candidate(**compiled_inputs)
                print(
                    f"compiled fullgraph {name}: ok, max abs difference from eager FA3 "
                    f"{(actual.float() - expected.float()).abs().max().item():.3e}, rings equal to eager's "
                    f"{torch.equal(compiled_inputs['key_ring'], eager_inputs['key_ring'])} "
                    f"{torch.equal(compiled_inputs['value_ring'], eager_inputs['value_ring'])}",
                    flush=True,
                )
                compiled = candidate if compiled is None else compiled
            except Exception as exc:
                print(f"compiled fullgraph {name}: FAILED, {type(exc).__name__}: {str(exc).splitlines()[0]}", flush=True)
        if compiled is None:
            return
        else:
            pass
        graph_inputs = fresh(inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                compiled(**fresh(inputs))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = compiled(**graph_inputs)
            for key in ("q", "k", "v", "key_ring", "value_ring"):
                graph_inputs[key].copy_(inputs[key])
            graph.replay()
            torch.cuda.synchronize()
            print(
                f"compiled in a CUDA graph: ok, max abs difference {(captured.float() - expected.float()).abs().max().item():.3e}",
                flush=True,
            )
        except Exception:
            print("compiled in a CUDA graph: FAILED", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
