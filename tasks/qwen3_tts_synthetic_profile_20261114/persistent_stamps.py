"""Where a persistent predictor layer launch spends its time, from in-kernel stamps.

Builds the layer with QWEN3_TTS_PREDICTOR_STAMPS and launches layer 0 alone at 1 and 16 rows,
L2-warm (the same layer again) and HBM-cold (after a 256 MB write), each launch after a full
synchronize. Every CTA's global timer and SM clock at the stamp points give, per phase: when its
consumers were released, the operand staging, the items, the counter signal, the weight-chunk
waits, and the handoff from the last producer's signal to the first consumer's release. Medians
over launches. Run with PYTHONPATH on the tree that holds predictor_persistent.py.

Usage: python3 persistent_stamps.py --out <json>
"""

import argparse
import json
import statistics

import torch
from predictor_bench import CACHED, EPS, HEAD_DIM, HIDDEN, KV_HEADS, SLOTS, load_predictor, rope_cache, time_call
from sglang.kernels.jit.utils import load_jit, make_cpp_args

from sglang_omni.models.qwen3_tts import predictor_persistent as persistent

STAMP_CTAS = 160
ENTRY, PRIMARY_WAITED, EXIT = 0, 1, 2
PHASE_BASE, PHASE_POINTS = 4, 4
REACHED, SIGNALED, WAITED, STAGED = 0, 1, 2, 3
LIST_LENGTH = 16
ITEM_END = 24
CHUNK_WAIT = ITEM_END + LIST_LENGTH
CHUNK_READY = CHUNK_WAIT + LIST_LENGTH
CHUNK_ISSUE = CHUNK_READY + LIST_LENGTH
SLOTS_PER_CTA = CHUNK_ISSUE + LIST_LENGTH
PHASES = ("qkv", "attention", "o_proj", "gate_up", "down")
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


def analyse_launch(stamps, item_phases):
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

    def between(cta, first, second):
        return (clock[cta, second] - clock[cta, first]).item() / rate / 1000

    distinct = sorted(set(timer[:ctas][timer[:ctas] > 0].tolist()))
    steps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
    row = {
        "timer resolution ns": min(steps) if steps else 0.0,
        "sm clock GHz": rate,
        "span": max(at(cta, EXIT) for cta in range(ctas)),
        "entry skew": max(at(cta, ENTRY) for cta in range(ctas)),
        "primary wait median": statistics.median(between(cta, ENTRY, PRIMARY_WAITED) for cta in range(ctas)),
    }
    previous_last_signal = None
    for phase, name in enumerate(PHASES):
        members = [cta for cta in range(ctas) if phase in item_phases[cta]]
        start_point = WAITED if phase > 0 else REACHED
        released = [at(cta, phase_slot(phase, start_point)) for cta in members]
        staging = [between(cta, phase_slot(phase, start_point), phase_slot(phase, STAGED)) for cta in members]
        work, chunk_waits, signal_costs = [], [], []
        for cta in members:
            indices = [index for index, item_phase in enumerate(item_phases[cta]) if item_phase == phase]
            last = indices[-1]
            work.append(between(cta, phase_slot(phase, STAGED), ITEM_END + last))
            chunk_base = sum(CHUNKS_PER_TILE[item_phase] for item_phase in item_phases[cta][: indices[0]])
            for chunk in range(chunk_base, chunk_base + CHUNKS_PER_TILE[phase] * len(indices)):
                chunk_waits.append(between(cta, CHUNK_WAIT + chunk, CHUNK_READY + chunk))
            if phase < len(PHASES) - 1:
                signal_costs.append(between(cta, ITEM_END + last, phase_slot(phase, SIGNALED)))
        prefix = f"{name} "
        row[prefix + "released first"] = min(released)
        row[prefix + "released last"] = max(released)
        if previous_last_signal is not None:
            row[prefix + "handoff"] = min(released) - previous_last_signal
        row[prefix + "staging median"] = statistics.median(staging)
        row[prefix + "staging max"] = max(staging)
        row[prefix + "items median"] = statistics.median(work)
        row[prefix + "items max"] = max(work)
        if chunk_waits:
            row[prefix + "chunk wait median"] = statistics.median(chunk_waits)
            row[prefix + "chunk wait max"] = max(chunk_waits)
        if signal_costs:
            row[prefix + "signal median"] = statistics.median(signal_costs)
            signals = [at(cta, phase_slot(phase, SIGNALED)) for cta in members]
            row[prefix + "last signal"] = max(signals)
            previous_last_signal = max(signals)
        else:
            row[prefix + "last exit"] = max(at(cta, EXIT) for cta in members)
    first_issue = [between(cta, ENTRY, CHUNK_ISSUE) for cta in range(ctas)]
    first_ready = [
        (clock[cta, CHUNK_READY] - clock[cta, CHUNK_ISSUE]).item() / rate / 1000 for cta in range(ctas)
    ]
    row["first chunk issue median"] = statistics.median(first_issue)
    row["first chunk issue to ready median"] = statistics.median(first_ready)
    row["first chunk issue to ready max"] = max(first_ready)
    return row


def median_rows(rows):
    return {key: statistics.median(row[key] for row in rows) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    cos_sin = rope_cache(device)
    first = model.layers[0]
    weights = persistent.pack_predictor_layer(
        qkv_weight=first.qkv, output_projection_weight=first.o, gate_up_weight=first.gate_up,
        down_weight=first.down, input_norm_weight=first.input_norm, post_norm_weight=first.post_norm,
        query_norm_weight=first.q_norm, key_norm_weight=first.k_norm,
    )
    layer = persistent.PersistentPredictorLayer(device, cos_sin, EPS)
    production_module = layer.module
    stamped_module = load_stamped_module()
    key_cache = torch.randn(64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
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
        cell["production one layer us"] = time_call(launch)
        layer.module = stamped_module
        cell["stamped one layer us"] = time_call(launch)
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
                    launch_rows.append(analyse_launch(host_stamps, item_phases))
            cell[condition] = median_rows(launch_rows)
        results[rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
