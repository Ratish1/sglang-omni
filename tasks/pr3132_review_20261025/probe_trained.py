"""Compare the old and new prefix graph ladders on one trained DiT."""

import bisect
import json
import statistics
import time
from pathlib import Path

import torch
from base_prefix_cuda_graph import PrefixCudaGraphRunner as BasePrefixRunner

from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    PrefixCacheRow, PrefixKVPool, compile_forward_prefix, grow_rows, release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
from sglang_omni.platforms import current_platform


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(3132)
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", device="cuda", gpu_id=0,
        dtype="bfloat16", max_batch_size=16, flow_prefix_cache_gb=0,
        enable_dit_torch_compile=False, enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False, enable_flow_whole_history_cuda_graph=False,
        enable_hift_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    estimator = vocoder.flow.packed_estimator
    assert estimator.is_ragged
    device = torch.device("cuda:0")
    attention = estimator.dit.transformer_blocks[0].attn
    dtype = torch.bfloat16
    pools = [PrefixKVPool(
        layer_num=len(estimator.dit.transformer_blocks), euler_steps=10,
        head_num=attention.heads, head_dim=attention.inner_dim // attention.heads,
        capacity_frames=16384, device=device, dtype=dtype,
    ) for _ in range(2)]
    compiled = compile_forward_prefix()
    for pool in pools:
        pool.forward = compiled
    mel_channels = vocoder.flow.output_size
    speaker_channels = vocoder.flow.spk_embed_affine_layer.out_features
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    records = []
    timing = []
    with vocoder.stream_context, torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        runners = []
        for runner_type, pool in zip((BasePrefixRunner, PrefixCudaGraphRunner), pools):
            for lengths in ((50,), (50, 70)):
                pairs = [(PrefixCacheRow(), PrefixCacheRow()) for _ in lengths]
                for pair, length in zip(pairs, lengths):
                    assert grow_rows(pool, list(pair), [length, length])
                solve_flow_euler_prefix(
                    estimator, pool,
                    torch.randn(1, sum(lengths), mel_channels, device=device),
                    time_span,
                    torch.randn(1, sum(lengths), mel_channels, device=device),
                    torch.randn(len(lengths), speaker_channels, device=device, dtype=dtype),
                    torch.randn(1, sum(lengths), mel_channels, device=device),
                    list(lengths), pairs, cfg_rate=0.7,
                )
                for pair in pairs:
                    release_rows(pool, list(pair))
            runner = runner_type(
                estimator, pool,
                graphs=SolveGraphCapture(current_platform.get_device_graph_backend(device),
                                         device=device, autocast_dtype=dtype),
                frame_dtype=torch.float32, speaker_dtype=dtype, cfg_rate=0.7,
                mel_channels=mel_channels, speaker_channels=speaker_channels,
                max_rows=16, hop_frames=200, min_hop_frames=50, max_frames=1024,
            )
            started = time.perf_counter()
            runner.capture()
            torch.cuda.synchronize()
            print(json.dumps({"capture": runner_type.__name__, "tiers": runner.tier_frames,
                              "seconds": time.perf_counter()-started}), flush=True)
            runners.append(runner)
        streams = [{
            "noise": torch.randn(1024, mel_channels, device=device),
            "mu": torch.randn(1024, mel_channels, device=device),
            "mel_conditioning": torch.randn(1024, mel_channels, device=device) * 0.1,
            "speaker_embeddings": torch.randn(speaker_channels, device=device, dtype=dtype),
        } for _ in range(16)]
        caches = [[(PrefixCacheRow(), PrefixCacheRow()) for _ in streams] for _ in pools]

        def inputs_for(names, starts, totals):
            return {
                **{field: torch.cat([streams[name][field][start:total]
                                    for name, start, total in zip(names, starts, totals)]).unsqueeze(0)
                   for field in ("noise", "mu", "mel_conditioning")},
                "speaker_embeddings": torch.stack([streams[name]["speaker_embeddings"] for name in names]),
                "time_span": time_span,
            }

        patterns = [
            [(0, 50)], [(1, 70), (0, 150)], [(2, 130), (1, 220), (0, 350)],
            [(index, caches[0][index][0].committed_frames + 200) for index in range(16)],
        ]
        # Build later totals at execution time, after earlier commitments exist.
        patterns[-1] = [(index, 600) for index in range(16)]
        for step, pattern in enumerate(patterns):
            names = [name for name, _ in pattern]
            if step == 3:
                totals = [caches[0][name][0].committed_frames + 200 for name in names]
            else:
                totals = [total for _, total in pattern]
            starts = [caches[0][name][0].committed_frames for name in names]
            lengths = [total-start for start,total in zip(starts,totals)]
            inputs = inputs_for(names, starts, totals)
            outputs = []
            for runner, pool, rows in zip(runners, pools, caches):
                pairs = [rows[name] for name in names]
                for pair, total in zip(pairs, totals):
                    assert grow_rows(pool, list(pair), [total, total])
                output = runner.run(**inputs, new_frames=lengths, caches=pairs)
                if output is None:
                    output = solve_flow_euler_prefix(estimator, pool, **inputs,
                        new_frames=lengths, caches=pairs, cfg_rate=0.7)
                outputs.append(output)
            assert torch.equal(*outputs), (step, (outputs[0]-outputs[1]).abs().max().item())
            for name in names:
                for first, second in zip(caches[0][name], caches[1][name]):
                    assert first.committed_frames == second.committed_frames
                    assert torch.equal(first.conv_context, second.conv_context)
                    pages = [row.pages(device)[:row.committed_frames] for row in (first,second)]
                    for pool_field in ("keys", "values"):
                        for step_a, step_b in zip(getattr(pools[0],pool_field),getattr(pools[1],pool_field)):
                            for layer_a,layer_b in zip(step_a,step_b):
                                assert torch.equal(layer_a[pages[0]],layer_b[pages[1]])
            row = {"case": f"persistent_{step}", "new_frames": lengths,
                   "frames":sum(lengths), "output_and_live_state_equal":True}
            records.append(row); print(json.dumps(row),flush=True)
        for pool, rows in zip(pools,caches):
            for pair in rows:
                release_rows(pool,list(pair))

        for row_count, frames_per_row in ((1,100),(2,200),(4,200),(8,200),(12,200),(16,200)):
            lengths = [frames_per_row]*row_count
            inputs = inputs_for(list(range(row_count)),[0]*row_count,lengths)
            outputs=[]
            pairs_by_arm=[]
            for runner,pool in zip(runners,pools):
                pairs=[(PrefixCacheRow(),PrefixCacheRow()) for _ in lengths]
                for pair in pairs:
                    assert grow_rows(pool,list(pair),[frames_per_row,frames_per_row])
                pairs_by_arm.append(pairs)
                outputs.append(runner.run(**inputs,new_frames=lengths,caches=pairs))
            assert torch.equal(*outputs)
            measurements=[[],[]]
            tiers=[runner.tier_frames[bisect.bisect_left(runner.tier_frames,sum(lengths))] for runner in runners]
            graphs=[runner.captured[bisect.bisect_left(runner.tier_frames,sum(lengths))].graph for runner in runners]
            for repeat in range(9):
                order=(0,1) if repeat%2==0 else (1,0)
                for arm in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(3):
                        graphs[arm].replay()
                    end.record(); end.synchronize()
                    measurements[arm].append(start.elapsed_time(end)/3)
            row={"rows":row_count,"frames_per_row":frames_per_row,"tiers":tiers,
                 "milliseconds":measurements,"median_ms":[statistics.median(xs) for xs in measurements]}
            timing.append(row); print(json.dumps(row),flush=True)
            for pool,pairs in zip(pools,pairs_by_arm):
                for pair in pairs:
                    release_rows(pool,list(pair))
        assert all(pool.free_frames==16384 for pool in pools)
    result={"equivalence":records,"timing":timing,"layers":len(estimator.dit.transformer_blocks),
            "tiers":[runner.tier_frames for runner in runners],"all_blocks_released":True}
    Path('/data/review3132oct25/trained.json').write_text(json.dumps(result,indent=2))
    scheduler.stop()


if __name__ == '__main__':
    main()
