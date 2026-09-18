"""P1-e1 and P1-e2 for slice P1 (slices/P1_PREDICTOR_PAIR_PASS.md).

Inputs: the served talker's predictor inputs, recorded from a live server by
record_predictor_inputs/sitecustomize.py.

run      the shipped Qwen3TTSTalker predictor chain (whichever sglang_omni is on
         PYTHONPATH) at the checkpoint's dims with its weights, on the recorded inputs,
         greedy and seeded-sampled, at bs 1, 2, 4, 8, 16; saves logits and codes
         (--fp32 runs the same chain in fp32: the truth)
compare  the numerics table from three run files (base, pair, truth)
time     CUDA graph replay ms and kernel count of the predictor chain per bucket

The talker is assembled the way tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py
assembles it; rope and qk-norm are plain torch (row-local ops, identical in both arms)
and the KV cache takes the copy path.
"""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

MAX_BS = 16
BUCKETS = (1, 2, 4, 8, 12, 16)
EVAL_BATCHES = (1, 2, 4, 8, 16)
PREFIX = "talker.code_predictor."


class TupleLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, unquantized) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = unquantized
        self.tp_size = 1
        self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight), None


class SwiGLU(nn.Module):
    def __init__(
        self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
    ) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.cat((gate, up)), requires_grad=False)
        self.down = nn.Parameter(down, requires_grad=False)

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down)


class TorchRotary:
    """Neox rope, computed in fp32 and rounded to the input dtype."""

    def __init__(self, head_dim: int, theta: float, device: torch.device) -> None:
        inv_freq = 1.0 / theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        )
        freqs = torch.outer(
            torch.arange(64, device=device, dtype=torch.float32), inv_freq
        )
        self.cos, self.sin, self.head_dim = freqs.cos(), freqs.sin(), head_dim

    def rotate(self, x, positions):
        rows = x.view(x.shape[0], -1, self.head_dim).float()
        first, second = rows.chunk(2, dim=-1)
        cos, sin = self.cos[positions][:, None], self.sin[positions][:, None]
        out = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        return out.to(x.dtype).reshape(x.shape)

    def __call__(self, positions, q, k, fused_set_kv_buffer_arg=None):
        assert fused_set_kv_buffer_arg is None
        return self.rotate(q, positions), self.rotate(k, positions)


class TorchRMSNorm(nn.Module):
    """RMSNorm with the vendor layer's call contract, for the fp32 truth."""

    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size), requires_grad=False)
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        normed = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return normed if residual is None else (normed, x)


def per_head_qk_norm(*, q, k, q_norm, k_norm, head_dim, alt_stream):
    return (
        q_norm(q.reshape(-1, head_dim)).reshape(q.shape),
        k_norm(k.reshape(-1, head_dim)).reshape(k.shape),
    )


def build_talker(model: str, device: torch.device, dtype: torch.dtype):
    import json

    from sglang_omni.models.qwen3_tts import sglang_model
    from sglang_omni.vendor.sglang.layers import RMSNorm

    sglang_model.apply_qk_norm = per_head_qk_norm
    with open(f"{model}/config.json") as handle:
        config = json.load(handle)["talker_config"]
    cp = config["code_predictor_config"]
    groups, hidden, head_dim = (
        config["num_code_groups"],
        cp["hidden_size"],
        cp["head_dim"],
    )
    heads, kv_heads, eps = (
        cp["num_attention_heads"],
        cp["num_key_value_heads"],
        cp["rms_norm_eps"],
    )
    weights = {}
    with safe_open(f"{model}/model.safetensors", "pt", device=str(device)) as handle:
        for key in handle.keys():
            if key.startswith(PREFIX) or key == "talker.model.codec_embedding.weight":
                weights[key] = handle.get_tensor(key).to(dtype)
    unquantized = sglang_model.UnquantizedLinearMethod()

    norm_class = TorchRMSNorm if dtype == torch.float32 else RMSNorm

    def norm(key, size):
        module = norm_class(size, eps=eps).to(device, dtype)
        module.weight.data.copy_(weights[key])
        return module

    rotary = TorchRotary(head_dim, float(cp["rope_theta"]), device)
    layers = []
    for index in range(cp["num_hidden_layers"]):
        p = f"{PREFIX}model.layers.{index}."
        attention = SimpleNamespace(
            q_size=heads * head_dim,
            kv_size=kv_heads * head_dim,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            alt_stream=None,
            q_norm=norm(p + "self_attn.q_norm.weight", head_dim),
            k_norm=norm(p + "self_attn.k_norm.weight", head_dim),
            qkv_proj=TupleLinear(
                torch.cat([weights[p + f"self_attn.{n}_proj.weight"] for n in "qkv"]),
                unquantized,
            ),
            o_proj=TupleLinear(weights[p + "self_attn.o_proj.weight"], unquantized),
            rotary_emb=rotary,
        )
        layers.append(
            SimpleNamespace(
                self_attn=attention,
                input_layernorm=norm(p + "input_layernorm.weight", hidden),
                post_attention_layernorm=norm(
                    p + "post_attention_layernorm.weight", hidden
                ),
                mlp=SwiGLU(
                    *(
                        weights[p + f"mlp.{n}_proj.weight"]
                        for n in ("gate", "up", "down")
                    )
                ),
            )
        )
    projection_weight = weights[PREFIX + "small_to_mtp_projection.weight"]
    projection_bias = weights[PREFIX + "small_to_mtp_projection.bias"]
    codec_embeddings = [
        nn.Embedding.from_pretrained(
            weights[f"{PREFIX}model.codec_embedding.{i}.weight"]
        )
        for i in range(groups - 1)
    ]
    layer0_embedding = nn.Embedding.from_pretrained(
        weights["talker.model.codec_embedding.weight"]
    )

    talker = object.__new__(sglang_model.Qwen3TTSTalker)
    talker.training = False
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(weight=SimpleNamespace(device=device))
    )
    talker.config = SimpleNamespace(
        num_code_groups=groups,
        code_predictor_config=SimpleNamespace(
            vocab_size=cp["vocab_size"], hidden_size=hidden
        ),
    )
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=layers,
            norm=norm(PREFIX + "model.norm.weight", hidden),
            codec_embedding=codec_embeddings,
        ),
        lm_head=[
            TupleLinear(weights[f"{PREFIX}lm_head.{i}.weight"], unquantized)
            for i in range(groups - 1)
        ],
        project_input=lambda h: F.linear(h, projection_weight, projection_bias),
    )
    talker.get_input_embeddings = lambda: layer0_embedding

    predictor_len = groups + 1
    positions = torch.arange(predictor_len, device=device, dtype=torch.long)
    talker._predictor_positions = positions
    talker._predictor_position_rows = (
        positions[:, None].expand(predictor_len, MAX_BS).contiguous()
    )
    talker._predictor_cache_slots = (
        torch.arange(MAX_BS, device=device, dtype=torch.long)[None, :] * predictor_len
        + positions[:, None]
    ).contiguous()
    talker._predictor_pair_positions = positions[:2].repeat(MAX_BS)
    talker._predictor_pair_cache_slots = (
        talker._predictor_cache_slots[:2].t().reshape(-1)
    )
    talker._predictor_k_cache = torch.zeros(
        len(layers),
        MAX_BS,
        predictor_len,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    talker._predictor_device = device
    talker._predictor_device_module = torch.get_device_module(device)
    talker._predictor_rope_stores_kv = False
    talker_hidden = config["hidden_size"]
    talker._output_codes = torch.zeros(MAX_BS, groups, dtype=torch.long, device=device)
    talker._output_embeds = torch.zeros(
        MAX_BS, talker_hidden, device=device, dtype=dtype
    )
    talker._predictor_embedding_buffer = torch.empty(
        MAX_BS, talker_hidden, device=device, dtype=dtype
    )
    talker._sampled_token_ids = torch.zeros(MAX_BS, dtype=torch.long, device=device)
    talker._sub_batch_size = 0
    talker._sub_temperature_tensor = torch.full(
        (MAX_BS,), 0.9, device=device, dtype=torch.float32
    )
    talker._sub_top_p_tensor = torch.ones(MAX_BS, device=device, dtype=torch.float32)
    talker._sub_top_k_tensor = torch.full(
        (MAX_BS,), 50, device=device, dtype=torch.long
    )
    talker._semantic_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_do_sample_tensor = torch.zeros(MAX_BS, device=device, dtype=torch.bool)
    talker._sub_seed_offsets = torch.arange(1, groups, device=device, dtype=torch.long)
    talker._sub_has_sampled_rows = False
    talker._sub_has_argmax_rows = False
    talker._sub_sampled_has_top_p = False
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = False
    talker._predictor_graphs = {}
    talker._predictor_graph_disabled = set()
    talker._predictor_graph_batch_sizes = BUCKETS
    talker._predictor_graph_enabled = True
    talker._predictor_graph_failure_count = 0
    talker._predictor_graph_capacity_fallback_count = 0
    talker._predictor_graph_capacity_warned = False
    talker._predictor_graph_capture_count = 0
    talker._predictor_graph_startup_count = 0
    talker._predictor_graph_pool = None
    talker._predictor_capture_stream = None
    return talker


def requests(batch: int, *, sampled: bool, offset: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                semantic_sampling_seed=0,
                subtalker_dosample=sampled,
                subtalker_temperature=0.9,
                subtalker_top_p=1.0,
                subtalker_top_k=50,
                subtalker_sampling_seed=1000 + offset + row,
            )
        )
        for row in range(batch)
    ]


def run(args) -> None:
    device = torch.device("cuda", 0)
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    talker = build_talker(args.model, device, dtype)
    inputs = torch.load(args.inputs)
    steps = min(args.steps, inputs["layer0_codes"].shape[0])
    captured: list[torch.Tensor] = []
    for head in talker.code_predictor.lm_head:
        head.register_forward_hook(
            lambda module, x, out: captured.append(out[0][:, -1, :].float().cpu())
        )
    results = {}
    for batch in EVAL_BATCHES:
        for sampled in (False, True):
            logits, codes = [], []
            for start in range(0, steps - batch + 1, batch):
                talker._decode_prep_rids = None
                talker.prepare_decode_buffers(
                    requests(batch, sampled=sampled, offset=start)
                )
                layer0 = (
                    inputs["layer0_codes"][start : start + batch]
                    .to(device)
                    .view(batch, 1)
                )
                hidden = (
                    inputs["talker_hidden"][start : start + batch]
                    .to(device, dtype)
                    .view(batch, 1, -1)
                )
                positions = torch.arange(start, start + batch, device=device)
                captured.clear()
                with torch.no_grad():
                    result_codes, _ = talker._code_predictor_forward_incremental(
                        layer0, hidden, positions
                    )
                logits.append(torch.stack(captured, dim=1))
                codes.append(result_codes[:, :, 0].cpu())
            results[(batch, sampled)] = {
                "logits": torch.cat(logits),
                "codes": torch.cat(codes),
            }
            print(
                f"bs {batch} {'sampled' if sampled else 'greedy'}: {results[(batch, sampled)]['codes'].shape[0]} rows",
                flush=True,
            )
    torch.save(results, args.out)


def compare(args) -> None:
    base, pair, truth = (
        torch.load(path) for path in (args.base, args.pair, args.truth)
    )
    print(
        f"{'bs':>3} {'mode':>8} {'arm':>5} {'rows':>5} {'codes==truth':>13} {'codes==base':>12} "
        f"{'logit maxabs (agreeing prefix)':>32} {'logit meanabs':>14}"
    )
    for (batch, sampled), reference in truth.items():
        for arm, run_result in (("base", base), ("pair", pair)):
            candidate = run_result[(batch, sampled)]
            rows = candidate["codes"].shape[0]
            same_truth = (
                (candidate["codes"] == reference["codes"])
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            same_base = (
                (candidate["codes"] == base[(batch, sampled)]["codes"])
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            # note(ratish): sub-step j's logits share inputs with the truth only while codes 1..j agree.
            agree = (
                (candidate["codes"][:, 1:] == reference["codes"][:, 1:])
                .int()
                .cumprod(dim=1)
                .bool()
            )
            agree = torch.cat(
                (torch.ones(rows, 1, dtype=torch.bool), agree[:, :-1]), dim=1
            )
            error = (candidate["logits"] - reference["logits"]).abs()[agree]
            print(
                f"{batch:>3} {'sampled' if sampled else 'greedy':>8} {arm:>5} {rows:>5} {same_truth:>13.4f} "
                f"{same_base:>12.4f} {error.max().item():>32.4e} {error.mean().item():>14.4e}"
            )


def time_chain(args) -> None:
    device = torch.device("cuda", 0)
    talker = build_talker(args.model, device, torch.bfloat16)
    inputs = torch.load(args.inputs)
    print(f"{'bucket':>6} {'ms':>8} {'kernels':>8}")
    for bucket in BUCKETS:
        talker._predictor_graphs.clear()
        talker._decode_prep_rids = None
        talker.prepare_decode_buffers(requests(bucket, sampled=True, offset=0))
        layer0 = inputs["layer0_codes"][:bucket].to(device).view(bucket, 1)
        hidden = (
            inputs["talker_hidden"][:bucket]
            .to(device, torch.bfloat16)
            .view(bucket, 1, -1)
        )
        positions = torch.arange(bucket, device=device)
        with torch.no_grad():
            talker.code_predictor_forward(layer0, hidden, positions)
        (graph,) = talker._predictor_graphs.values()
        times = []
        for _ in range(args.reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            graph.graph.replay()
            torch.cuda.synchronize()
        kernels = sum(
            e.device_type == torch.autograd.DeviceType.CUDA for e in prof.events()
        )
        print(f"{bucket:>6} {statistics.median(times):>8.3f} {kernels:>8}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "compare", "time"))
    parser.add_argument("--model")
    parser.add_argument("--inputs")
    parser.add_argument("--out")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--base")
    parser.add_argument("--pair")
    parser.add_argument("--truth")
    args = parser.parse_args()
    torch.manual_seed(0)
    {"run": run, "compare": compare, "time": time_chain}[args.mode](args)


if __name__ == "__main__":
    main()
