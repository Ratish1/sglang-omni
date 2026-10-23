"""Check PR 3096 against its parent kernels and predictor at production dimensions."""

from __future__ import annotations

import ast
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import torch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch import nn

import sglang_omni.models.qwen3_tts.sampling_kernels as sampling
import sglang_omni.models.qwen3_tts.sglang_model as model
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSCodecTables

BASE_COMMIT = "3a4e73436d1206a675156a6d028d3a0fb2e1e753"
REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = Path("/data/review3096")


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def parent_source(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_parent_kernel(name: str) -> ModuleType:
    path = OUTPUT_DIRECTORY / f"parent_{name}.py"
    path.write_text(parent_source(f"sglang_omni/models/qwen3_tts/{name}.py"))
    return load_module(f"pr3096_parent_{name}", path)


def check_leaf(parent_sampling: ModuleType) -> int:
    checked_replays = 0
    generator = torch.Generator(device="cuda").manual_seed(3096)
    for batch_size in (1, 3, 16, 64):
        for max_top_k in (50, 64, 128):
            for projected_size in (0, 1024):
                logits = torch.empty(
                    batch_size, 2048, device="cuda", dtype=torch.bfloat16
                )
                temperatures = torch.full((batch_size,), 0.9, device="cuda")
                top_ks = torch.full(
                    (batch_size,), max_top_k, device="cuda", dtype=torch.long
                )
                top_ps = torch.full((batch_size,), 0.8, device="cuda")
                seeds = torch.arange(batch_size, device="cuda", dtype=torch.long)
                positions = seeds.clone()
                table = torch.randn(
                    2048, 2048, generator=generator, device="cuda"
                ).bfloat16()
                initial_sum = torch.randn(
                    batch_size, 2048, generator=generator, device="cuda"
                ).bfloat16()
                codes = torch.full(
                    (batch_size, 16), -71, device="cuda", dtype=torch.long
                )
                gathered = torch.empty_like(initial_sum)
                accumulated = torch.empty_like(initial_sum)
                if projected_size:
                    projected_table = torch.randn(
                        2048, projected_size, generator=generator, device="cuda"
                    ).bfloat16()
                    projected = torch.empty(
                        batch_size, projected_size, device="cuda", dtype=torch.bfloat16
                    )
                else:
                    projected_table = None
                    projected = None

                def fused_step() -> None:
                    assert sampling.sample_codes_and_gather_codec_embedding(
                        logits,
                        temperatures,
                        top_ks,
                        top_ps,
                        seeds,
                        positions,
                        max_top_k=max_top_k,
                        has_top_p=True,
                        codes=codes[:, 15],
                        embedding_weight=table,
                        gathered=gathered,
                        accumulated=accumulated,
                        projected_weight=projected_table,
                        projected=projected,
                    )

                logits.normal_(generator=generator)
                accumulated.copy_(initial_sum)
                fused_step()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused_step()
                for replay_index in range(3):
                    logits.normal_(generator=generator)
                    logits[0].fill_(-100)
                    logits[0, 0 if replay_index % 2 == 0 else 2047] = 100
                    if batch_size > 1:
                        logits[1].zero_()
                        top_ks[1] = 1
                    else:
                        pass
                    seeds.add_(-(2**40) + 17 + replay_index)
                    positions.add_(2**32 + 31)
                    expected_codes = (
                        parent_sampling.sample_from_logits_with_seed_top_k_top_p(
                            logits,
                            temperatures,
                            top_ks,
                            top_ps,
                            seeds,
                            positions,
                            max_top_k=max_top_k,
                            has_top_p=True,
                        )
                    )
                    expected_gathered = torch.nn.functional.embedding(
                        expected_codes, table
                    )
                    expected_sum = initial_sum + expected_gathered
                    accumulated.copy_(initial_sum)
                    gathered.fill_(float("nan"))
                    codes.fill_(-71)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(codes[:, 15], expected_codes)
                    assert torch.equal(
                        codes[:, :15], torch.full_like(codes[:, :15], -71)
                    )
                    assert torch.equal(gathered, expected_gathered)
                    assert torch.equal(accumulated, expected_sum)
                    if projected is not None:
                        assert torch.equal(
                            projected,
                            torch.nn.functional.embedding(
                                expected_codes, projected_table
                            ),
                        )
                    else:
                        pass
                    checked_replays += 1
                graph.reset()
    return checked_replays


def build_talker(fixtures: ModuleType, *, has_projection: bool) -> model.Qwen3TTSTalker:
    fixtures.MAX_BS = 64
    fixtures.NUM_CODE_GROUPS = 16
    fixtures.PRED_VOCAB = 2048
    fixtures.BUCKETS = (1, 4, 16, 64)
    device = torch.device("cuda")
    talker = fixtures.build_talker(device)
    torch.manual_seed(3096)
    layers = [fixtures.real_shape_layer(device) for _ in range(5)]
    final_norm = fixtures.real_shape_norm(1024, device)
    real_talker = fixtures.real_shape_talker(device, layers, final_norm, fused=True)
    talker.code_predictor = real_talker.code_predictor
    talker.predictor_fused_layers = real_talker.predictor_fused_layers
    talker.predictor_k_cache = real_talker.predictor_k_cache
    talker.predictor_v_cache = real_talker.predictor_v_cache
    talker.predictor_position_rows = real_talker.predictor_position_rows
    talker.predictor_cache_slots = real_talker.predictor_cache_slots
    talker.predictor_pair_positions = real_talker.predictor_pair_positions
    talker.predictor_pair_cache_slots = real_talker.predictor_pair_cache_slots
    talker.predictor_device = talker.predictor_k_cache.device
    talker.config.code_predictor_config = SimpleNamespace(
        vocab_size=2048, hidden_size=1024
    )
    talker.output_embeds = torch.empty(64, 2048, device=device, dtype=torch.bfloat16)
    talker.predictor_embedding_buffer = torch.empty_like(talker.output_embeds)
    talker.code_predictor.model.codec_embedding = Qwen3TTSCodecTables(
        15, 2048, 2048
    ).to(device, torch.bfloat16)
    talker.code_predictor.lm_head = nn.ModuleList(
        [fixtures.TupleLinear(1024, 2048).to(device, torch.bfloat16) for _ in range(15)]
    )
    projection = nn.Linear(2048, 1024, bias=True).to(device, torch.bfloat16)
    talker.code_predictor.project_input = projection
    first_embedding = nn.Embedding(2048, 2048).to(device, torch.bfloat16)
    talker.get_input_embeddings = lambda: first_embedding
    if has_projection:
        tables = talker.code_predictor.model.codec_embedding.weight
        talker.predictor_projected_embeddings = torch.stack(
            [projection(table) for table in tables[:-1]]
        )
        talker.predictor_projected_buffer = torch.empty(
            64, 1024, device=device, dtype=torch.bfloat16
        )
    else:
        pass
    return talker


def kernel_counts(graph: torch.cuda.CUDAGraph) -> dict[str, int]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        graph.replay()
        torch.cuda.synchronize()
    names = [
        event.name for event in profile.events() if event.device_type.name == "CUDA"
    ]
    return {
        "total": len(names),
        "codec_gather": sum(
            "gather_codec_embedding_and_add_kernel" in name for name in names
        ),
        "merge_sampler": sum(
            "seeded_top_k_merge_sample_kernel" in name for name in names
        ),
    }


def check_predictor(
    fixtures: ModuleType, parent_sampling: ModuleType, parent_gather: ModuleType
) -> list[dict]:
    parent_tree = ast.parse(
        parent_source("sglang_omni/models/qwen3_tts/sglang_model.py")
    )
    parent_class = next(
        node
        for node in parent_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3TTSTalker"
    )
    parent_methods = [
        node
        for node in parent_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("code_predictor_forward_incremental", "sample_subtalker_token")
    ]
    namespace = dict(vars(model))
    namespace["sample_from_logits_with_seed_top_k_top_p"] = (
        parent_sampling.sample_from_logits_with_seed_top_k_top_p
    )
    namespace["gather_codec_embedding_and_add"] = (
        parent_gather.gather_codec_embedding_and_add
    )
    exec(
        compile(
            ast.Module(body=parent_methods, type_ignores=[]),
            "parent_predictor_methods",
            "exec",
        ),
        namespace,
    )
    measurements = []
    cases = [
        (1, 50, True, "sampled"),
        (3, 50, True, "sampled"),
        (16, 50, True, "sampled"),
        (64, 50, True, "sampled"),
        (16, 128, True, "sampled"),
        (16, 50, False, "sampled"),
        (16, 50, True, "mixed"),
        (16, 50, True, "greedy"),
    ]
    for batch_size, max_top_k, has_projection, mode in cases:
        talkers = [
            build_talker(fixtures, has_projection=has_projection) for _ in range(2)
        ]
        parent_talker, head_talker = talkers
        for name in ("code_predictor_forward_incremental", "sample_subtalker_token"):
            setattr(parent_talker, name, MethodType(namespace[name], parent_talker))
        requests = [
            fixtures.make_request(
                top_k=max_top_k,
                top_p=0.9,
                sub_seed=1000 + row,
                dosample=mode == "sampled" or (mode == "mixed" and row % 2 == 0),
            )
            for row in range(batch_size)
        ]
        for talker in talkers:
            talker.prepare_decode_buffers(requests)
        layer_zero = torch.randint(0, 2048, (batch_size, 1), device="cuda")
        hidden = torch.randn(batch_size, 1, 2048, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(batch_size, device="cuda", dtype=torch.long)
        graphs = []
        outputs = []
        for talker in talkers:
            for _ in range(2):
                talker.code_predictor_forward_incremental(layer_zero, hidden, positions)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = talker.code_predictor_forward_incremental(
                    layer_zero, hidden, positions
                )
            graphs.append(graph)
            outputs.append(output)
        for replay_index in range(3):
            layer_zero.random_(0, 2048)
            hidden.normal_()
            positions.add_(17)
            for talker, graph in zip(talkers, graphs):
                talker.output_codes.fill_(-31)
                talker.output_embeds.fill_(float("nan"))
                talker.predictor_embedding_buffer.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            for parent_output, head_output in zip(*outputs):
                assert torch.equal(parent_output, head_output), (
                    batch_size,
                    max_top_k,
                    has_projection,
                    mode,
                    replay_index,
                )
            assert bool((outputs[1][0] >= 0).all())
            assert bool(torch.isfinite(outputs[1][1]).all())
        counts = [kernel_counts(graph) for graph in graphs]
        for _ in range(10):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        timings = [[], []]
        for round_index in range(6):
            order = (0, 1) if round_index % 2 == 0 else (1, 0)
            for graph_index in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[graph_index].replay()
                end.record()
                end.synchronize()
                timings[graph_index].append(start.elapsed_time(end) / 50)
        measurement = {
            "batch_size": batch_size,
            "top_k": max_top_k,
            "projected": has_projection,
            "mode": mode,
            "changed_input_replays": 3,
            "parent_kernel_counts": counts[0],
            "head_kernel_counts": counts[1],
            "parent_ms": timings[0],
            "head_ms": timings[1],
            "parent_median_ms": statistics.median(timings[0]),
            "head_median_ms": statistics.median(timings[1]),
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
        for graph in graphs:
            graph.reset()
    return measurements


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(3096)
    parent_sampling = load_parent_kernel("sampling_kernels")
    parent_gather = load_parent_kernel("predictor_kernels")
    fixtures = load_module(
        "pr3096_predictor_fixtures",
        REPOSITORY / "tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py",
    )
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.inference_mode(),
    ):
        leaf_replays = check_leaf(parent_sampling)
        print(f"leaf_replays={leaf_replays}", flush=True)
        predictor_results = check_predictor(fixtures, parent_sampling, parent_gather)
    results = {
        "base_commit": BASE_COMMIT,
        "torch": torch.__version__,
        "triton": sampling.triton.__version__,
        "device": torch.cuda.get_device_name(),
        "leaf_replays": leaf_replays,
        "predictor": predictor_results,
    }
    (OUTPUT_DIRECTORY / "mechanics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("all independent parent comparisons passed", flush=True)


if __name__ == "__main__":
    main()
