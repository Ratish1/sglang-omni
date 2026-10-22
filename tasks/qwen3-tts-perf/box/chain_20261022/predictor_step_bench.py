"""The Qwen3-TTS predictor step as served, through the talker's own predictor graph, at the 1.7B checkpoint's shapes
with seeded random weights, on the tree on PYTHONPATH (one arm per process).

usage: PYTHONPATH=<tree> python3 predictor_step_bench.py [--batch 1 16 64] [--replays 200] [--card 3]

One step is the opening pair pass, fifteen codebook heads each sampled with the subtalker defaults (top_k 50, top_p
1.0, temperature 0.9, one seed per row) and gathered into the summed embedding, and fourteen single-token passes over
the fused layers. Per batch: the replay's device time from CUDA events between back-to-back replays (median, p10,
p90), the kernels per step and their device time by name from the profiler, and a hash of the codes and summed
embeddings, which two arms of a class A change must print equal.
"""

import argparse
import hashlib
import statistics
from collections import defaultdict
from types import SimpleNamespace

import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context
from torch import nn
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSCodecTables, Qwen3TTSTalker
from sglang_omni.utils.predictor_layers import resolve_fused_predictor_layers
from sglang_omni.vendor.sglang.layers import RMSNorm

TALKER_HIDDEN = 2048
TALKER_VOCAB = 3072
HIDDEN = 1024
INTERMEDIATE = 3072
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
LAYERS = 5
GROUPS = 16
VOCAB = 2048
PREDICTOR_LEN = GROUPS + 1
ROPE_THETA = 1000000
DTYPE = torch.bfloat16


class TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.quant_method = UnquantizedLinearMethod()
        self.tp_group = None

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.proj.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.proj(hidden_states), None


class GatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = TupleLinear(HIDDEN, 2 * INTERMEDIATE)
        self.down_proj = TupleLinear(INTERMEDIATE, HIDDEN)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)[0]))[0]


def norm(size: int, device: torch.device) -> RMSNorm:
    layer_norm = RMSNorm(size, eps=1e-6).to(device, DTYPE)
    with torch.no_grad():
        layer_norm.weight.normal_(1.0, 0.1)
    return layer_norm


def layer(device: torch.device) -> SimpleNamespace:
    attention = SimpleNamespace(
        hidden_size=HIDDEN,
        q_size=Q_HEADS * HEAD_DIM,
        kv_size=KV_HEADS * HEAD_DIM,
        num_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        q_norm=norm(HEAD_DIM, device),
        k_norm=norm(HEAD_DIM, device),
        alt_stream=None,
        qkv_proj=TupleLinear(HIDDEN, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM).to(device, DTYPE),
        o_proj=TupleLinear(Q_HEADS * HEAD_DIM, HIDDEN).to(device, DTYPE),
        rotary_emb=RotaryEmbedding(HEAD_DIM, HEAD_DIM, 64, ROPE_THETA, True, DTYPE).to(device),
    )
    return SimpleNamespace(
        input_layernorm=norm(HIDDEN, device),
        post_attention_layernorm=norm(HIDDEN, device),
        self_attn=attention,
        mlp=GatedMLP().to(device, DTYPE),
    )


def build_talker(device: torch.device, batch_sizes: list[int]) -> Qwen3TTSTalker:
    torch.manual_seed(0)
    max_batch_size = max(batch_sizes)
    talker = object.__new__(Qwen3TTSTalker)
    talker.training = False
    talker.model = SimpleNamespace(codec_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    talker.config = SimpleNamespace(
        num_code_groups=GROUPS,
        code_predictor_config=SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN),
    )
    positions = torch.arange(PREDICTOR_LEN, device=device, dtype=torch.long)
    talker.predictor_positions = positions
    talker.predictor_position_rows = positions[:, None].expand(PREDICTOR_LEN, max_batch_size).contiguous()
    talker.predictor_cache_slots = (
        torch.arange(max_batch_size, device=device, dtype=torch.long)[None, :] * PREDICTOR_LEN + positions[:, None]
    ).contiguous()
    talker.predictor_pair_positions = positions[:2].repeat(max_batch_size)
    talker.predictor_pair_cache_slots = talker.predictor_cache_slots[:2].t().reshape(-1)
    talker.predictor_k_cache = torch.zeros(
        LAYERS, max_batch_size, PREDICTOR_LEN, KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    talker.predictor_v_cache = torch.zeros_like(talker.predictor_k_cache)
    talker.predictor_device = device
    talker.predictor_device_module = torch.get_device_module(device)
    talker.predictor_rope_stores_kv = False
    talker.output_codes = torch.zeros(max_batch_size, GROUPS, dtype=torch.long, device=device)
    talker.output_embeds = torch.zeros(max_batch_size, TALKER_HIDDEN, device=device, dtype=DTYPE)
    talker.predictor_embedding_buffer = torch.empty(max_batch_size, TALKER_HIDDEN, device=device, dtype=DTYPE)
    talker.predictor_projected_embeddings = torch.empty(GROUPS - 2, VOCAB, HIDDEN, device=device, dtype=DTYPE)
    talker.predictor_projected_buffer = torch.empty(max_batch_size, HIDDEN, device=device, dtype=DTYPE)
    talker.sampled_token_ids = torch.zeros(max_batch_size, dtype=torch.long, device=device)
    talker.sub_batch_size = 0
    talker.sub_temperature_tensor = torch.full((max_batch_size,), 0.9, device=device, dtype=torch.float32)
    talker.sub_top_p_tensor = torch.ones(max_batch_size, device=device, dtype=torch.float32)
    talker.sub_top_k_tensor = torch.full((max_batch_size,), 50, device=device, dtype=torch.long)
    talker.semantic_sampling_seed_tensor = torch.zeros(max_batch_size, device=device, dtype=torch.long)
    talker.sub_sampling_seed_tensor = torch.zeros(max_batch_size, device=device, dtype=torch.long)
    talker.sub_do_sample_tensor = torch.zeros(max_batch_size, device=device, dtype=torch.bool)
    talker.decode_prep_rids = None
    talker.decode_prep_epoch = 0
    talker.sub_seed_offsets = torch.arange(1, GROUPS, device=device, dtype=torch.long)
    talker.sub_has_sampled_rows = False
    talker.sub_has_argmax_rows = False
    talker.sub_sampled_has_top_p = False
    talker.sub_sampled_max_top_k = 0
    talker.sub_sampled_has_unbounded_top_k = False

    projection = TupleLinear(TALKER_HIDDEN, HIDDEN, bias=True).to(device, DTYPE)
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[layer(device) for _ in range(LAYERS)],
            norm=norm(HIDDEN, device),
            codec_embedding=Qwen3TTSCodecTables(GROUPS - 1, VOCAB, TALKER_HIDDEN).to(device, DTYPE),
        ),
        lm_head=nn.ModuleList([TupleLinear(HIDDEN, VOCAB).to(device, DTYPE) for _ in range(GROUPS - 1)]),
        project_input=lambda hidden: projection(hidden)[0],
    )
    with torch.no_grad():
        talker.code_predictor.model.codec_embedding.weight.normal_(0.0, 0.02)
    layer0_embedding = nn.Embedding(TALKER_VOCAB, TALKER_HIDDEN).to(device, DTYPE)
    talker.get_input_embeddings = lambda: layer0_embedding
    talker.predictor_fused_layers = resolve_fused_predictor_layers(
        talker.code_predictor, PREDICTOR_LEN, max_batch_size, device, DTYPE
    )
    assert talker.predictor_fused_layers is not None
    with torch.no_grad():
        talker.post_load_weights()

    talker.predictor_graphs = {}
    talker.predictor_graph_disabled = set()
    talker.predictor_graph_batch_sizes = tuple(sorted(batch_sizes))
    talker.predictor_graph_enabled = True
    talker.predictor_graph_failure_count = 0
    talker.predictor_graph_capacity_fallback_count = 0
    talker.predictor_graph_capacity_warned = False
    talker.predictor_graph_capture_count = 0
    talker.predictor_graph_startup_count = 0
    talker.predictor_graph_replay_count = 0
    talker.predictor_eager_count = 0
    talker.predictor_graph_pool = None
    talker.predictor_capture_stream = None
    return talker


def requests(batch_size: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                semantic_sampling_seed=2000 + row,
                subtalker_dosample=True,
                subtalker_temperature=0.9,
                subtalker_top_p=1.0,
                subtalker_top_k=50,
                subtalker_sampling_seed=1000 + row,
            )
        )
        for row in range(batch_size)
    ]


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def bench(talker: Qwen3TTSTalker, batch_size: int, replays: int, device: torch.device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(batch_size)
    layer0 = torch.randint(0, TALKER_VOCAB, (batch_size, 1), generator=generator).to(device)
    hidden = torch.randn(batch_size, 1, TALKER_HIDDEN, generator=generator).to(device, DTYPE)
    positions = torch.arange(40, 40 + batch_size, device=device, dtype=torch.long)
    talker.prepare_decode_buffers(requests(batch_size))

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        return talker.code_predictor_forward(layer0, hidden, semantic_positions=positions)

    for _ in range(10):
        codes, embeds = step()
    torch.cuda.synchronize(device)
    assert talker.predictor_graph_replay_count > 0, "the predictor step did not replay a graph"
    digest = f"codes {tensor_hash(codes)} embeds {tensor_hash(embeds)}"

    events = [torch.cuda.Event(enable_timing=True) for _ in range(replays + 1)]
    events[0].record()
    for index in range(replays):
        step()
        events[index + 1].record()
    torch.cuda.synchronize(device)
    times_us = sorted(1000 * events[i].elapsed_time(events[i + 1]) for i in range(replays))
    quantiles = statistics.quantiles(times_us, n=10)

    profiled_replays = 20
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for _ in range(profiled_replays):
            step()
        torch.cuda.synchronize(device)
    by_name: dict[str, list[float]] = defaultdict(list)
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            by_name[event.name].append(event.time_range.elapsed_us())
        else:
            pass
    launches = sum(len(durations) for durations in by_name.values()) / profiled_replays
    print(
        f"bs {batch_size}: replay median {statistics.median(times_us):.1f} us p10 {quantiles[0]:.1f} "
        f"p90 {quantiles[-1]:.1f}, {launches:.0f} kernels a step, {digest}"
    )
    rows = sorted(by_name.items(), key=lambda item: -sum(item[1]))
    for name, durations in rows[:14]:
        print(
            f"    {len(durations) / profiled_replays:5.0f} a step  {sum(durations) / profiled_replays:8.1f} us a step  "
            f"{statistics.mean(durations):6.2f} us each  {name[:90]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend=Backend.DISABLED)),
        ),
        torch.no_grad(),
    ):
        talker = build_talker(device, args.batch)
        for batch_size in args.batch:
            bench(talker, batch_size, args.replays, device)


if __name__ == "__main__":
    main()
