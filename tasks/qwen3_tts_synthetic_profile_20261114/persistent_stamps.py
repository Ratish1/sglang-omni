"""Where a persistent predictor layer spends its time, from in-kernel stamps.

Builds the kernel with QWEN3_TTS_PREDICTOR_STAMPS and launches the 5-layer stack at 1 and 16 rows,
with the previous launch's weights in L2 and after a 256 MB write, each launch after a full
synchronize. The kernel stamps its middle layer (layer 2), whose weights layer 1 prefetched.
Phase points carry the global timer and the SM clock; item and chunk points carry the SM clock
only. Per phase: when its consumers were released, the operand staging, the items, the counter
signal, the handoff from the last producer's signal to the first consumer's release, and each item
split into its parts. Medians over launches. Run with PYTHONPATH on the tree that holds
predictor_persistent.py.

Usage: python3 persistent_stamps.py --out <json>
"""

import argparse
import json
import statistics

import torch
from predictor_bench import CACHED, EPS, HEAD_DIM, HIDDEN, KV_HEADS, LAYERS, SLOTS, load_predictor, rope_cache, time_call
from sglang.kernels.jit.utils import load_jit, make_cpp_args

from sglang_omni.models.qwen3_tts import predictor_persistent as persistent

STAMP_CTAS = 160
ENTRY, EXIT = 0, 2
PHASE_BASE, PHASE_POINTS = 4, 4
REACHED, STAGED = 0, 3
LIST_LENGTH = 16
ITEM_END = 24
ITEM_MULTIPLIED = ITEM_END + LIST_LENGTH
ITEM_REDUCED = ITEM_MULTIPLIED + LIST_LENGTH
CHUNK_WAIT = ITEM_REDUCED + LIST_LENGTH
CHUNK_READY = CHUNK_WAIT + LIST_LENGTH
CHUNK_ISSUE = CHUNK_READY + LIST_LENGTH
ATTENTION_NORMED = CHUNK_ISSUE + LIST_LENGTH
ATTENTION_SCORED = ATTENTION_NORMED + 1
SLOTS_PER_CTA = ATTENTION_SCORED + 1
PHASES = ("qkv", "attention", "o_proj", "gate_up", "down")
ATTENTION = 1
CHUNKS_PER_TILE = (1, 0, 2, 1, 3)
WARMUP_LAUNCHES = 5
LAUNCHES = 30
FLUSH_BYTES = 256 * 2**20


def load_stamped_module():
    template_arguments = make_cpp_args(True)
    return load_jit(
        "qwen3_tts_predictor_layer_stamps",
        *template_arguments,
        cuda_files=[str(persistent.PREDICTOR_LAYER_SOURCE)],
        cuda_wrappers=[
            ("predictor_layer", f"qwen3_tts_predictor::PredictorLayer<{template_arguments}>::run"),
            ("read_stamps", "qwen3_tts_predictor::PredictorStamps::read"),
        ],
        extra_cuda_cflags=["-DQWEN3_TTS_PREDICTOR_STAMPS"],
    )


def phase_slot(phase, point):
    return PHASE_BASE + phase * PHASE_POINTS + point


def cta_items(schedule, cta_count):
    items = schedule.items.cpu().tolist()
    lengths = schedule.lengths.cpu().tolist()
    return [[item >> 12 for item in items[cta][: lengths[cta]]] for cta in range(cta_count)]


def item_parts(phase, index, first_chunk, item_start, between):
    """An item's (attention) or a batch's (MMA phases) parts in us, by the SM clock of its CTA."""
    if phase == ATTENTION:
        return {
            "load": between(item_start, ITEM_MULTIPLIED + index),
            "compute": between(ITEM_MULTIPLIED + index, ITEM_REDUCED + index),
            "end barrier": between(ITEM_REDUCED + index, ITEM_END + index),
        }
    return {
        "to first chunk": between(item_start, CHUNK_READY + first_chunk),
        "warp 0 mma": between(CHUNK_READY + first_chunk, ITEM_MULTIPLIED + index),
        "other warps": between(ITEM_MULTIPLIED + index, ITEM_REDUCED + index),
        "epilogue": between(ITEM_REDUCED + index, ITEM_END + index),
    }


def stamped_groups(phase, indices, batch_tiles):
    """The item groups that carry stamps: one attention item, or one batch of an MMA phase's tiles."""
    size = 1 if phase == ATTENTION else batch_tiles
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def analyse_launch(stamps, item_phases, batch_tiles):
    """Named intervals of one launch in microseconds; times are from the earliest CTA entry."""
    timer = stamps[..., 0].double()
    clock = stamps[..., 1].double()
    ctas = len(item_phases)
    t0 = timer[:ctas, ENTRY].min().item()
    rate = statistics.median(
        ((clock[cta, EXIT] - clock[cta, ENTRY]) / (timer[cta, EXIT] - timer[cta, ENTRY])).item() for cta in range(ctas)
    )

    def at(cta, slot):
        return (timer[cta, slot].item() - t0) / 1000

    def clock_between(cta):
        return lambda first, second: (clock[cta, second] - clock[cta, first]).item() / rate / 1000

    row = {
        "sm clock GHz": rate,
        "span": max(at(cta, EXIT) for cta in range(ctas)),
    }
    for phase, name in enumerate(PHASES):
        members = [cta for cta in range(ctas) if phase in item_phases[cta]]
        reached = [at(cta, phase_slot(phase, REACHED)) for cta in members]
        staged = [at(cta, phase_slot(phase, STAGED)) for cta in members]
        staging, work, chunk_waits = [], [], []
        parts = {}
        for cta in members:
            between = clock_between(cta)
            indices = [index for index, item_phase in enumerate(item_phases[cta]) if item_phase == phase]
            chunk = sum(CHUNKS_PER_TILE[item_phase] for item_phase in item_phases[cta][: indices[0]])
            groups = stamped_groups(phase, indices, batch_tiles)
            staging.append(between(phase_slot(phase, REACHED), phase_slot(phase, STAGED)))
            work.append(between(phase_slot(phase, STAGED), ITEM_END + groups[-1][0]))
            item_start = phase_slot(phase, STAGED)
            for group in groups:
                for part, value in item_parts(phase, group[0], chunk, item_start, between).items():
                    parts.setdefault(part, []).append(value)
                for offset in range(CHUNKS_PER_TILE[phase] * len(group)):
                    chunk_waits.append(between(CHUNK_WAIT + chunk + offset, CHUNK_READY + chunk + offset))
                chunk += CHUNKS_PER_TILE[phase] * len(group)
                item_start = ITEM_END + group[0]
            if phase == ATTENTION:
                last = indices[-1]
                for part, (first, second) in {
                    "norm and rope": (ITEM_MULTIPLIED + last, ATTENTION_NORMED),
                    "scores and softmax": (ATTENTION_NORMED, ATTENTION_SCORED),
                    "values and store": (ATTENTION_SCORED, ITEM_REDUCED + last),
                }.items():
                    parts.setdefault(part, []).append(between(first, second))
        prefix = f"{name} "
        # Staging reads the previous phase's words until they are current, so it includes the
        # wait for that phase's last producer.
        row[prefix + "reached first"] = min(reached)
        row[prefix + "reached last"] = max(reached)
        row[prefix + "staged first"] = min(staged)
        row[prefix + "staged last"] = max(staged)
        row[prefix + "staging median"] = statistics.median(staging)
        row[prefix + "items median"] = statistics.median(work)
        row[prefix + "items max"] = max(work)
        for part, values in parts.items():
            row[prefix + f"item {part} median"] = statistics.median(values)
        if chunk_waits:
            row[prefix + "chunk wait median"] = statistics.median(chunk_waits)
    row["last exit"] = max(at(cta, EXIT) for cta in range(ctas))
    return row


def median_rows(rows):
    return {key: statistics.median(row[key] for row in rows) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-tiles", type=int, default=3, help="1 for heads that stamp every tile")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    cos_sin = rope_cache(device)
    weights = persistent.stack_predictor_layers(
        [
            persistent.pack_predictor_layer(
                qkv_weight=each.qkv, output_projection_weight=each.o, gate_up_weight=each.gate_up,
                down_weight=each.down, input_norm_weight=each.input_norm, post_norm_weight=each.post_norm,
                query_norm_weight=each.q_norm, key_norm_weight=each.k_norm,
            )
            for each in model.layers
        ]
    )
    layer = persistent.PersistentPredictorLayers(device, cos_sin, EPS)
    production_module = layer.module
    stamped_module = load_stamped_module()
    key_cache = torch.randn(LAYERS, 64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    flush = torch.empty(FLUSH_BYTES // 4, device=device, dtype=torch.float32)
    host_stamps = torch.zeros(STAMP_CTAS, SLOTS_PER_CTA, 2, dtype=torch.int64)
    results = {}
    for rows in (1, 16):
        x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(x)

        def launch():
            layer.forward(weights, x, out, CACHED, key_cache, value_cache)

        cell = {}
        layer.module = production_module
        cell["production 5 layers us"] = time_call(launch)
        layer.module = stamped_module
        cell["stamped 5 layers us"] = time_call(launch)
        item_phases = cta_items(layer.schedule_for(rows), layer.cta_count)
        for condition in ("l2_warm", "hbm"):
            launch_rows = []
            for index in range(WARMUP_LAUNCHES + LAUNCHES):
                if condition == "hbm":
                    flush.zero_()
                torch.cuda.synchronize()
                launch()
                torch.cuda.synchronize()
                stamped_module.read_stamps(host_stamps)
                if index >= WARMUP_LAUNCHES:
                    launch_rows.append(analyse_launch(host_stamps, item_phases, args.batch_tiles))
            cell[condition] = median_rows(launch_rows)
        results[rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
