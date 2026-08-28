# SPDX-License-Identifier: Apache-2.0
"""Kernel-level A/B on the Qwen3-Omni thinker path.

Usage (one GPU):
    python kernel_ab.py run --seed 0 --out $OUT/ab/run1.pt [--model-path P] [--video V]
    python kernel_ab.py run --seed 0 --out $OUT/ab/run2.pt [--model-path P] [--video V]
    python kernel_ab.py compare $OUT/ab/run1.pt $OUT/ab/run2.pt
    python kernel_ab.py pairs $OUT/ab/run1.pt

run generates inputs on the CPU with numpy's legacy RandomState (stable
across versions), hashes them, executes every kernel of the inventory in
00_plan.md section 3 and saves the outputs. compare takes two dumps with the
same inputs (it refuses to run when the input hashes differ) and prints, per
case and tensor, whether the outputs are bitwise equal, the mismatch
fraction, the max absolute and max relative difference, and the kernel time.
Two runs of the same stack show which kernels are nondeterministic. pairs
compares, inside one dump, the backends that compute the same function
(PAIRS below and the vision SDPA backends), which sizes the numerical
difference each server-level arm of 00_plan.md section 4.3 introduces.

Cases that need a published sglang runtime context (MRoPE) or model files
(HF processor) run only when --model-path (and --video) are given. Every
case is isolated, a failing case records its exception and the run goes on.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback

import numpy as np
import torch

HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
EXPERTS = 128
TOPK = 8
INTER = 768
VOCAB = 152064
BLOCK = 128
M_LIST = (1, 16, 256, 4096)
SEQ_PREFILL = 4096
DECODE_BS = 16
DECODE_LEN = 14000
AUDIO_HEADS = 20
AUDIO_HEAD_DIM = 64
VISION_HEADS = 16
VISION_HEAD_DIM = 72


class Inputs:
    """Deterministic CPU inputs, cast to bf16 once, hashed."""

    def __init__(self, seed: int):
        self.rs = np.random.RandomState(seed)
        self.cache: dict[str, torch.Tensor] = {}
        self.hashes: dict[str, str] = {}

    def get(
        self,
        name: str,
        shape,
        scale: float = 1.0,
        dtype=torch.bfloat16,
        kind: str = "normal",
    ) -> torch.Tensor:
        if name in self.cache:
            return self.cache[name]
        if kind == "normal":
            arr = self.rs.standard_normal(size=shape).astype(np.float32) * scale
            t = torch.from_numpy(arr).to(dtype)
        elif kind == "uniform_int":
            t = torch.from_numpy(
                self.rs.randint(0, int(scale), size=shape).astype(np.int64)
            )
        else:
            raise ValueError(kind)
        self.hashes[name] = hashlib.sha256(
            t.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()[:16]
        self.cache[name] = t
        return t


def _cuda(t: torch.Tensor) -> torch.Tensor:
    return t.to("cuda", non_blocking=False)


def _timed(fn, iters: int = 5):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return out, float(np.median(times))


def _try_import(*paths: str):
    err = None
    for path in paths:
        mod, _, attr = path.rpartition(".")
        try:
            return getattr(importlib.import_module(mod), attr)
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise ImportError(f"none of {paths} importable: {err}")


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            t = t.view(torch.uint8)
        return t.cpu().contiguous()
    if isinstance(obj, (tuple, list)):
        return [_to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def _fp8_block_quantize(w: torch.Tensor, block: int = BLOCK):
    """Block-wise FP8 weight quantization on the GPU (weights only, deterministic)."""
    *lead, n, k = w.shape
    wf = w.float().view(*lead, n // block, block, k // block, block)
    amax = wf.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-10)
    scale = amax / 448.0
    q = (wf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(*lead, n, k)
    return q, scale.squeeze(-1).squeeze(-2).float().contiguous()


def _topk_reference(logits: torch.Tensor):
    probs = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(probs, TOPK, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.contiguous(), ids.to(torch.int32).contiguous()


def _load_moe_config(m: int, dtype_tag: str | None):
    """Mirror get_moe_configs for E=128, N=768 on H100 without a runtime context."""
    import glob
    import os

    import sglang

    root = os.path.join(
        os.path.dirname(sglang.__file__),
        "srt",
        "layers",
        "moe",
        "moe_runner",
        "triton_utils",
        "configs",
    )
    suffix = "" if dtype_tag is None else f",dtype={dtype_tag},block_shape=[128, 128]"
    name = f"E={EXPERTS},N={INTER},device_name=NVIDIA_H100_80GB_HBM3{suffix}.json"
    files = sorted(glob.glob(os.path.join(root, "triton_*", name)))
    if files:
        cfg = json.load(open(files[-1]))
        key = min(cfg.keys(), key=lambda x: abs(int(x) - m))
        return dict(cfg[key]), files[-1]
    if dtype_tag == "fp8_w8a8":
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": BLOCK,
            "BLOCK_SIZE_K": BLOCK,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }, "default"
    if m <= EXPERTS:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }, "default"
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }, "default"


def _moe_triton_direct(x, w13, w2, weights, ids, *, fp8: bool, w13_s=None, w2_s=None):
    """The kernel sequence of fused_moe._fused_moe_kernel_sequence, launched directly."""
    import triton.language as tl

    invoke = _try_import(
        "sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel"
    )
    align = _try_import(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.moe_align_block_size"
    )
    silu = _try_import(
        "sglang.kernels.ops.activation.activation.silu_and_mul",
        "sglang.jit_kernel.activation.silu_and_mul",
    )
    moe_sum_reduce = _try_import("sgl_kernel.moe_sum_reduce")
    m = x.shape[0]
    tag = "fp8_w8a8" if fp8 else None
    cfg, src = _load_moe_config(m, tag)
    sorted_ids, expert_ids, num_post = align(ids, cfg["BLOCK_SIZE_M"], EXPERTS)
    c1 = torch.empty((m * TOPK, 2 * INTER), device=x.device, dtype=x.dtype)
    c2 = torch.empty((m * TOPK, INTER), device=x.device, dtype=x.dtype)
    c3 = torch.empty((m, TOPK, HIDDEN), device=x.device, dtype=x.dtype)
    out = torch.empty((m, HIDDEN), device=x.device, dtype=x.dtype)
    block_shape = [BLOCK, BLOCK] if fp8 else None
    invoke(
        x,
        w13,
        None,
        c1,
        None,
        w13_s,
        None,
        weights,
        ids,
        sorted_ids,
        expert_ids,
        num_post,
        False,
        TOPK,
        cfg,
        tl.bfloat16,
        fp8,
        False,
        False,
        False,
        False,
        block_shape,
    )
    silu(c1, c2)
    invoke(
        c2,
        w2,
        None,
        c3,
        None,
        w2_s,
        None,
        weights,
        ids,
        sorted_ids,
        expert_ids,
        num_post,
        True,
        1,
        cfg,
        tl.bfloat16,
        fp8,
        False,
        False,
        False,
        False,
        block_shape,
    )
    moe_sum_reduce(c3, out)
    return out, src


def _cutlass_moe(x, w13_q, w2_q, w13_s, w2_s, weights, ids):
    fn = _try_import("sglang.srt.layers.moe.cutlass_moe.cutlass_fused_experts_fp8")
    dev = x.device
    e = EXPERTS
    buf = dict(
        ab_strides1=torch.full((e,), HIDDEN, device=dev, dtype=torch.int64),
        c_strides1=torch.full((e,), 2 * INTER, device=dev, dtype=torch.int64),
        ab_strides2=torch.full((e,), INTER, device=dev, dtype=torch.int64),
        c_strides2=torch.full((e,), HIDDEN, device=dev, dtype=torch.int64),
        workspace=torch.empty(90000, device=dev, dtype=torch.uint8),
        a_ptr=torch.empty(e, device=dev, dtype=torch.int64),
        b_ptr=torch.empty(e, device=dev, dtype=torch.int64),
        out_ptr=torch.empty(e, device=dev, dtype=torch.int64),
        a_scales_ptr=torch.empty(e, device=dev, dtype=torch.int64),
        b_scales_ptr=torch.empty(e, device=dev, dtype=torch.int64),
        expert_offsets=torch.empty(e + 1, device=dev, dtype=torch.int32),
        problem_sizes1=torch.empty(e, 3, device=dev, dtype=torch.int32),
        problem_sizes2=torch.empty(e, 3, device=dev, dtype=torch.int32),
    )
    return fn(
        x,
        w13_q.transpose(1, 2),
        w2_q.transpose(1, 2),
        w13_s.transpose(1, 2),
        w2_s.transpose(1, 2),
        weights,
        ids,
        buf["ab_strides1"],
        buf["c_strides1"],
        buf["ab_strides2"],
        buf["c_strides2"],
        buf["workspace"],
        buf["a_ptr"],
        buf["b_ptr"],
        buf["out_ptr"],
        buf["a_scales_ptr"],
        buf["b_scales_ptr"],
        buf["expert_offsets"],
        buf["problem_sizes1"],
        buf["problem_sizes2"],
        use_fp8_blockscale=True,
    )


def env_info() -> dict:
    info = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for mod in (
        "triton",
        "sglang",
        "sgl_kernel",
        "flashinfer",
        "deep_gemm",
        "transformers",
        "torchcodec",
        "torchvision",
    ):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"missing ({type(exc).__name__})"
    return info


def publish_context(model_path: str) -> str | None:
    try:
        from sglang.srt.runtime_context import get_context
        from sglang.srt.server_args import ServerArgs

        get_context().set_server_args(
            ServerArgs(model_path=model_path, trust_remote_code=True)
        )
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def run(args) -> None:
    inp = Inputs(args.seed)
    results: dict[str, dict] = {}
    context_error = (
        publish_context(args.model_path) if args.model_path else "no --model-path"
    )

    def case(name, fn):
        rec: dict = {"error": None, "out": None, "time_ms": None, "note": None}
        try:
            out, ms = _timed(fn)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str):
                out, rec["note"] = out
            rec["out"] = _to_cpu(out)
            rec["time_ms"] = ms
        except Exception:  # noqa: BLE001
            rec["error"] = traceback.format_exc(limit=3)
        results[name] = rec
        status = "ok" if rec["error"] is None else "ERROR"
        print(
            f"{name:40s} {status} {rec['time_ms'] if rec['time_ms'] is not None else ''}",
            flush=True,
        )
        torch.cuda.empty_cache()

    scale = HEAD_DIM**-0.5
    w_gate = _cuda(inp.get("w_gate", (EXPERTS, HIDDEN), 0.02))
    w_norm = _cuda(inp.get("w_norm", (HIDDEN,), 0.1)) + 1.0
    w_qknorm = _cuda(inp.get("w_qknorm", (HEAD_DIM,), 0.1)) + 1.0
    w_qkv = _cuda(inp.get("w_qkv", ((Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, HIDDEN), 0.02))
    w_vocab = _cuda(inp.get("w_vocab", (VOCAB, HIDDEN), 0.02))
    w13 = _cuda(inp.get("w13", (EXPERTS, 2 * INTER, HIDDEN), 0.02))
    w2 = _cuda(inp.get("w2", (EXPERTS, HIDDEN, INTER), 0.02))
    w_q = w_qkv[: Q_HEADS * HEAD_DIM].contiguous()
    w13_q, w13_s = _fp8_block_quantize(w13)
    w2_q, w2_s = _fp8_block_quantize(w2)
    wq_q, wq_s = _fp8_block_quantize(w_q)

    rmsnorm = _try_import("sgl_kernel.rmsnorm")
    fused_add_rmsnorm = _try_import("sgl_kernel.fused_add_rmsnorm")
    silu_aot = _try_import("sgl_kernel.silu_and_mul")
    silu_jit = _try_import(
        "sglang.kernels.ops.activation.activation.silu_and_mul",
        "sglang.jit_kernel.activation.silu_and_mul",
    )
    moe_sum_reduce = _try_import("sgl_kernel.moe_sum_reduce")
    moe_fused_gate = _try_import(
        "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
        "sglang.jit_kernel.moe_fused_gate.moe_fused_gate",
    )
    group_quant = _try_import(
        "sglang.kernels.ops.quantization.fp8_kernel.sglang_per_token_group_quant_fp8"
    )
    dense_deepgemm = _try_import(
        "sglang.srt.layers.quantization.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback"
    )
    dense_triton = _try_import(
        "sglang.srt.layers.quantization.fp8_utils.triton_w8a8_block_fp8_linear"
    )
    fa_varlen = _try_import("sgl_kernel.flash_attn.flash_attn_varlen_func")
    fa_kvcache = _try_import("sgl_kernel.flash_attn.flash_attn_with_kvcache")

    for m in M_LIST:
        x = _cuda(inp.get(f"x_{m}", (m, HIDDEN), 1.0))
        res = _cuda(inp.get(f"res_{m}", (m, HIDDEN), 1.0))
        logits = _cuda(inp.get(f"router_logits_{m}", (m, EXPERTS), 2.0))
        x_act = _cuda(inp.get(f"x_act_{m}", (m, 2 * INTER), 1.0))
        x_sum = _cuda(inp.get(f"x_sum_{m}", (m, TOPK, HIDDEN), 1.0))
        w_ref, ids_ref = _topk_reference(logits)

        case(f"gate_gemm/M={m}", lambda: torch.matmul(x, w_gate.t()))
        case(f"qkv_gemm_bf16/M={m}", lambda: torch.matmul(x, w_qkv.t()))
        if m <= 256:

            def lm_head():
                lg = torch.matmul(x, w_vocab.t())
                return {
                    "argmax": lg.argmax(dim=-1),
                    "logits_slice": lg[:, :8192].clone(),
                    "logits_max": lg.max(dim=-1).values,
                }

            case(f"lm_head/M={m}", lm_head)
        case(f"rmsnorm/M={m}", lambda: rmsnorm(x, w_norm, 1e-6))

        def far():
            a, r = x.clone(), res.clone()
            fused_add_rmsnorm(a, r, w_norm, 1e-6)
            return {"x": a, "residual": r}

        case(f"fused_add_rmsnorm/M={m}", far)
        case(
            f"qk_norm/M={m}",
            lambda: rmsnorm(
                x[:, : Q_HEADS * HEAD_DIM].reshape(-1, HEAD_DIM).contiguous(),
                w_qknorm,
                1e-6,
            ),
        )

        def router():
            w, i = moe_fused_gate(
                logits,
                torch.zeros(EXPERTS, device="cuda", dtype=torch.float32),
                TOPK,
                scoring_func="softmax",
                renormalize=True,
            )
            return {"weights": w, "ids": i}

        case(f"router/M={m}", router)
        case(
            f"silu_and_mul_aot/M={m}",
            lambda: silu_aot(
                x_act, torch.empty((m, INTER), device="cuda", dtype=torch.bfloat16)
            ),
        )
        case(
            f"silu_and_mul_jit/M={m}",
            lambda: silu_jit(
                x_act, torch.empty((m, INTER), device="cuda", dtype=torch.bfloat16)
            ),
        )

        def msr():
            o = torch.empty((m, HIDDEN), device="cuda", dtype=torch.bfloat16)
            moe_sum_reduce(x_sum, o)
            return o

        case(f"moe_sum_reduce/M={m}", msr)
        case(
            f"moe_bf16_triton/M={m}",
            lambda: _moe_triton_direct(x, w13, w2, w_ref, ids_ref, fp8=False),
        )
        case(
            f"moe_fp8_triton/M={m}",
            lambda: _moe_triton_direct(
                x, w13_q, w2_q, w_ref, ids_ref, fp8=True, w13_s=w13_s, w2_s=w2_s
            ),
        )
        case(
            f"moe_fp8_cutlass/M={m}",
            lambda: _cutlass_moe(x, w13_q, w2_q, w13_s, w2_s, w_ref, ids_ref),
        )
        case(
            f"fp8_group_quant/M={m}",
            lambda: dict(zip(("q", "s"), group_quant(x, BLOCK))),
        )
        case(
            f"fp8_group_quant_colmajor/M={m}",
            lambda: dict(
                zip(
                    ("q", "s"),
                    group_quant(
                        x, BLOCK, column_major_scales=True, scale_tma_aligned=True
                    ),
                )
            ),
        )
        case(
            f"fp8_dense_deepgemm/M={m}",
            lambda: dense_deepgemm(x, wq_q, [BLOCK, BLOCK], wq_s),
        )
        case(
            f"fp8_dense_triton/M={m}",
            lambda: dense_triton(x, wq_q, [BLOCK, BLOCK], wq_s),
        )

        if context_error is None:

            def mrope():
                from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

                rope = MRotaryEmbedding(
                    HEAD_DIM,
                    HEAD_DIM,
                    65536,
                    1000000,
                    True,
                    torch.bfloat16,
                    mrope_section=[24, 20, 20],
                ).to("cuda")
                pos = _cuda(
                    inp.get(
                        f"pos_{m}", (3, m), 8192, dtype=torch.int64, kind="uniform_int"
                    )
                )
                q = _cuda(inp.get(f"q_rope_{m}", (m, Q_HEADS * HEAD_DIM), 1.0)).clone()
                k = _cuda(inp.get(f"k_rope_{m}", (m, KV_HEADS * HEAD_DIM), 1.0)).clone()
                q, k = rope.forward_cuda(pos, q, k)
                return {"q": q, "k": k}

            case(f"mrope/M={m}", mrope)
        else:
            results[f"mrope/M={m}"] = {
                "error": f"skipped: {context_error}",
                "out": None,
                "time_ms": None,
                "note": None,
            }

    q_p = _cuda(inp.get("q_prefill", (SEQ_PREFILL, Q_HEADS, HEAD_DIM), 1.0))
    k_p = _cuda(inp.get("k_prefill", (SEQ_PREFILL, KV_HEADS, HEAD_DIM), 1.0))
    v_p = _cuda(inp.get("v_prefill", (SEQ_PREFILL, KV_HEADS, HEAD_DIM), 1.0))
    for seqs in (1, 4):
        L = SEQ_PREFILL // seqs
        cu = torch.arange(0, SEQ_PREFILL + 1, L, device="cuda", dtype=torch.int32)
        case(
            f"fa3_prefill/seqs={seqs}",
            lambda: fa_varlen(
                q_p, k_p, v_p, cu, cu, L, L, causal=True, softmax_scale=scale
            ),
        )

    def fa3_decode():
        q = _cuda(inp.get("q_decode", (DECODE_BS, 1, Q_HEADS, HEAD_DIM), 1.0))
        kc = _cuda(
            inp.get("k_cache", (DECODE_BS * DECODE_LEN, 1, KV_HEADS, HEAD_DIM), 1.0)
        )
        vc = _cuda(
            inp.get("v_cache", (DECODE_BS * DECODE_LEN, 1, KV_HEADS, HEAD_DIM), 1.0)
        )
        page_table = torch.arange(
            DECODE_BS * DECODE_LEN, device="cuda", dtype=torch.int32
        ).view(DECODE_BS, DECODE_LEN)
        lens = torch.full((DECODE_BS,), DECODE_LEN, device="cuda", dtype=torch.int32)
        return fa_kvcache(
            q=q,
            k_cache=kc,
            v_cache=vc,
            page_table=page_table,
            cache_seqlens=lens,
            causal=True,
            softmax_scale=scale,
            num_splits=0,
        )

    case("fa3_decode/bs=16,len=14000", fa3_decode)

    import torch.nn.functional as F

    for L in (1024, 4096):
        qv = _cuda(inp.get(f"vq_{L}", (1, VISION_HEADS, L, VISION_HEAD_DIM), 1.0))
        kv = _cuda(inp.get(f"vk_{L}", (1, VISION_HEADS, L, VISION_HEAD_DIM), 1.0))
        vv = _cuda(inp.get(f"vv_{L}", (1, VISION_HEADS, L, VISION_HEAD_DIM), 1.0))
        case(
            f"vision_sdpa_default/L={L}",
            lambda: F.scaled_dot_product_attention(qv, kv, vv),
        )
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            for name, backend in (
                ("math", SDPBackend.MATH),
                ("flash", SDPBackend.FLASH_ATTENTION),
                ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                ("cudnn", SDPBackend.CUDNN_ATTENTION),
            ):

                def forced(backend=backend):
                    with sdpa_kernel([backend]):
                        return F.scaled_dot_product_attention(qv, kv, vv)

                case(f"vision_sdpa_{name}/L={L}", forced)
        except Exception:  # noqa: BLE001
            results[f"vision_sdpa_forced/L={L}"] = {
                "error": traceback.format_exc(limit=2),
                "out": None,
                "time_ms": None,
                "note": None,
            }
        aq = _cuda(inp.get(f"aq_{L}", (L, AUDIO_HEADS, AUDIO_HEAD_DIM), 1.0))
        ak = _cuda(inp.get(f"ak_{L}", (L, AUDIO_HEADS, AUDIO_HEAD_DIM), 1.0))
        av = _cuda(inp.get(f"av_{L}", (L, AUDIO_HEADS, AUDIO_HEAD_DIM), 1.0))
        cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)
        case(
            f"audio_sdpa/L={L}",
            lambda: F.scaled_dot_product_attention(
                aq.transpose(0, 1)[None],
                ak.transpose(0, 1)[None],
                av.transpose(0, 1)[None],
            )[0]
            .transpose(0, 1)
            .contiguous(),
        )
        case(
            f"audio_fa3_varlen/L={L}",
            lambda: fa_varlen(
                aq,
                ak,
                av,
                cu,
                cu,
                L,
                L,
                causal=False,
                softmax_scale=AUDIO_HEAD_DIM**-0.5,
            ),
        )

    if args.model_path and args.video:

        def hf_processor():
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(args.model_path)
            out = proc(
                text="<|vision_start|><|video_pad|><|vision_end|>describe",
                videos=[args.video],
                return_tensors="pt",
                videos_kwargs={
                    "fps": 2,
                    "max_frames": 128,
                    "max_pixels": 401408,
                    "device": "cpu",
                    "use_audio_in_video": True,
                },
            )
            rec = {}
            for key in (
                "pixel_values_videos",
                "input_features",
                "input_ids",
                "video_grid_thw",
                "feature_attention_mask",
            ):
                t = out.get(key)
                if isinstance(t, torch.Tensor):
                    rec[f"{key}_sha256"] = hashlib.sha256(
                        t.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    rec[f"{key}_shape"] = list(t.shape)
                    rec[f"{key}_head"] = t.flatten()[:64].float().clone()
            return rec

        case("hf_processor", hf_processor)

    torch.save(
        {
            "env": env_info(),
            "seed": args.seed,
            "input_hashes": inp.hashes,
            "context_error": context_error,
            "results": results,
        },
        args.out,
    )
    print(json.dumps(env_info(), indent=1))
    print(f"saved {args.out}")


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "shape_or_dtype_mismatch": f"{tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
        }
    equal = bool(torch.equal(a, b))
    if not a.is_floating_point():
        mism = (a != b).sum().item()
        return {
            "equal": equal,
            "mismatch_frac": mism / max(a.numel(), 1),
            "max_abs": float((a.long() - b.long()).abs().max()) if a.numel() else 0.0,
            "max_rel": None,
        }
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    return {
        "equal": equal,
        "mismatch_frac": (diff > 0).float().mean().item(),
        "max_abs": diff.max().item(),
        "max_rel": (diff / bf.abs().clamp(min=1e-3)).max().item(),
    }


def compare(path_a: str, path_b: str) -> int:
    A, B = torch.load(path_a, weights_only=False), torch.load(
        path_b, weights_only=False
    )
    if A["input_hashes"] != B["input_hashes"]:
        bad = [
            k
            for k in A["input_hashes"]
            if A["input_hashes"].get(k) != B["input_hashes"].get(k)
        ]
        print(f"input hashes differ for {bad}, refusing to compare")
        return 2
    print("A env:", json.dumps(A["env"]))
    print("B env:", json.dumps(B["env"]))
    rows = []
    for name in A["results"]:
        ra, rb = A["results"][name], B["results"].get(name)
        if rb is None:
            rows.append((name, "-", "missing in B", None, None, None, None, None))
            continue
        if ra["error"] or rb["error"]:
            rows.append(
                (
                    name,
                    "-",
                    f"error A={bool(ra['error'])} B={bool(rb['error'])}",
                    None,
                    None,
                    None,
                    ra["time_ms"],
                    rb["time_ms"],
                )
            )
            continue
        oa, ob = ra["out"], rb["out"]
        if not isinstance(oa, dict):
            oa, ob = {"out": oa}, {"out": ob}
        for key in oa:
            va, vb = oa[key], ob.get(key)
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                mt = _metrics(va, vb)
                rows.append(
                    (
                        name,
                        key,
                        "bitwise" if mt.get("equal") else "differs",
                        mt.get("mismatch_frac"),
                        mt.get("max_abs"),
                        mt.get("max_rel"),
                        ra["time_ms"],
                        rb["time_ms"],
                    )
                )
            elif va != vb:
                rows.append(
                    (
                        name,
                        key,
                        f"differs: {str(va)[:40]} vs {str(vb)[:40]}",
                        None,
                        None,
                        None,
                        ra["time_ms"],
                        rb["time_ms"],
                    )
                )
            else:
                rows.append(
                    (name, key, "equal", 0.0, 0.0, 0.0, ra["time_ms"], rb["time_ms"])
                )
    rows.sort(key=lambda r: -(r[5] or 0.0))
    print(
        f"{'case':44s} {'tensor':14s} {'verdict':34s} {'mismatch':>9s} {'max_abs':>10s} {'max_rel':>10s} {'ms_A':>8s} {'ms_B':>8s}"
    )
    for r in rows:
        f = lambda v, w: ("" if v is None else f"{v:.3g}").rjust(w)
        print(
            f"{r[0]:44s} {r[1]:14s} {r[2]:34s} {f(r[3], 9)} {f(r[4], 10)} {f(r[5], 10)} {f(r[6], 8)} {f(r[7], 8)}"
        )
    return 0


# Backend pairs that compute the same function inside one dump: the kernel
# family the server switches between with the section 4.3 arms, plus the
# encoder attention implementations. Each pair is compared at every M or L
# the run produced.
PAIRS = (
    ("fp8_dense_deepgemm", "fp8_dense_triton"),
    ("moe_fp8_cutlass", "moe_fp8_triton"),
    ("silu_and_mul_aot", "silu_and_mul_jit"),
    ("audio_sdpa", "audio_fa3_varlen"),
)


def _pair_rows(results: dict, name_a: str, name_b: str):
    rows = []
    for name in results:
        if not name.startswith(name_a + "/"):
            continue
        other = name_b + name[len(name_a) :]
        ra, rb = results[name], results.get(other)
        if rb is None:
            rows.append((name, other, "-", "missing", None, None, None))
            continue
        if ra["error"] or rb["error"]:
            rows.append(
                (
                    name,
                    other,
                    "-",
                    f"error A={bool(ra['error'])} B={bool(rb['error'])}",
                    None,
                    None,
                    None,
                )
            )
            continue
        oa, ob = ra["out"], rb["out"]
        if not isinstance(oa, dict):
            oa, ob = {"out": oa}, {"out": ob}
        for key, va in oa.items():
            vb = ob.get(key)
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                mt = _metrics(va, vb)
                rows.append(
                    (
                        name,
                        other,
                        key,
                        "bitwise" if mt.get("equal") else "differs",
                        mt.get("mismatch_frac"),
                        mt.get("max_abs"),
                        mt.get("max_rel"),
                    )
                )
    return rows


def pairs(path: str) -> int:
    """Compare the backend pairs of PAIRS inside one dump. The vision SDPA
    backends are compared against vision_sdpa_default."""
    dump = torch.load(path, weights_only=False)
    results = dump["results"]
    print("env:", json.dumps(dump["env"]))
    rows = []
    for name_a, name_b in PAIRS:
        rows += _pair_rows(results, name_a, name_b)
    vision_backends = sorted(
        {
            n.split("/")[0]
            for n in results
            if n.startswith("vision_sdpa_") and not n.startswith("vision_sdpa_default")
        }
    )
    for backend in vision_backends:
        rows += _pair_rows(results, "vision_sdpa_default", backend)
    rows.sort(key=lambda r: -(r[6] or 0.0))
    print(
        f"{'case A':38s} {'case B':38s} {'tensor':10s} {'verdict':22s} {'mismatch':>9s} {'max_abs':>10s} {'max_rel':>10s}"
    )
    for r in rows:
        f = lambda v, w: ("" if v is None else f"{v:.3g}").rjust(w)
        print(
            f"{r[0]:38s} {r[1]:38s} {r[2]:10s} {r[3]:22s} {f(r[4], 9)} {f(r[5], 10)} {f(r[6], 10)}"
        )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--model-path", default=None)
    r.add_argument("--video", default=None)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    pr = sub.add_parser("pairs")
    pr.add_argument("dump")
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a)
        return 0
    if a.cmd == "pairs":
        return pairs(a.dump)
    return compare(a.a, a.b)


if __name__ == "__main__":
    sys.exit(main())
