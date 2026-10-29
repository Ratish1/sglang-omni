"""Gate for the general reference-encoder speedups (STATE 10-29 item 3), on the tree first on the path: main's bf16
encoder or D's float32 one, loaded as the server loads it, encoded through the reference batcher with its graphs.

Variants (each with a fresh batcher, its graphs captured after the change; dynamo reset between variants):
  base          the tree as it is
  pads          each MimiConv1d's padding from Python integers (Mimi keeps stride, kernel and padding as tensors and
                computes the extra padding with tensor ops, modeling_mimi.py:241-275), and no pad call at zero widths
  qkv           q, k and v as one GEMM per transformer layer (SGLang's QKVParallelLinear), weights viewed from it
  compile       pads, then torch.compile(fullgraph=True, dynamic=False) over the encode frame per graph bucket,
                compiled before capture under a raised recompile limit (the incremental codec's mechanism); the eager
                arm runs pads alone, since compiled code belongs to the bucket graphs only
  compile_qkv   compile with qkv
  compile_dynamic  pads, then one torch.compile(fullgraph=True, dynamic=True) shared by every bucket (startup: one trace
                instead of one per bucket)
  compile_dynamic_frames  compile_dynamic with the input length checked to be whole 1,920-sample frames, a fact
                inductor can use for alignment
  compile_regional  pads, then torch.compile(dynamic=True, fullgraph=True) on the repeated regions only, SGLang's
                default-path form (small helpers, one dynamic program for every shape): each MimiResnetBlock and
                MimiTransformerLayer (Module.compile), and one residual-quantizer step shared by the 16 codebooks;
                the downsampling ELUs and convs, the transformer's mask and rope stay eager
  compile_cosy  Fun-CosyVoice3's compile form on the whole encode frame: its process-wide settings
                (set_dit_compile_config, fun_cosyvoice3/stages.py:1447-1455: the FX graph cache, a cache size limit of
                1,024, Triton's libdevice set once so a warm boot's lookups hit), its options (packed_dit.py:35-39:
                triton.autotune_pointwise off, emulate_precision_casts on), fullgraph, not dynamic=True, the input's
                length dimension hinted dynamic with maybe_mark_dynamic on every call (stages.py:1533-1535), so one
                program serves every bucket
  compile_static_cosy  compile (static, per bucket) with compile_cosy's settings and options, no length hint
  compile_regional_cosy  compile_regional's regions with compile_cosy's settings and options, not dynamic=True:
                automatic dynamic, so a region's second bucket makes its length symbolic (CosyVoice's two-size warmup)
  served      the tree exactly as loaded, nothing installed or reset (a branch that carries its own mechanisms)
Per variant: capture wall time and Dynamo's per-phase compile times; codes through graphs and eager against
official's float32 exact-length encode and against the base variant's graph codes, all clips; device time, kernel
count and top kernels per encode over the first 200 clips under the profiler.

usage: PYTHONPATH=TREE python3 encoder_speedups_gate.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-1.7B-Base [--card 0]
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch._inductor.config as inductor_config  # noqa: leading-underscore  # upstream name
import torch.nn.functional as F
from torch._dynamo.utils import compile_times  # noqa: leading-underscore  # upstream name
from torch._inductor.runtime.compile_tasks import _set_triton_libdevice_path  # noqa: leading-underscore  # upstream name
from torch.profiler import ProfilerActivity, profile
from transformers.models.mimi import modeling_mimi
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiResidualVectorQuantizer,
    MimiResnetBlock,
    MimiTransformerLayer,
)

import sglang_omni
from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES,
    Qwen3TTSReferenceEncoderCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.checkpoint import resolve_checkpoint

VARIANTS = {
    "base": (),
    "pads": ("pads",),
    "qkv": ("qkv",),
    "compile": ("pads", "compile"),
    "compile_qkv": ("pads", "qkv", "compile"),
    "compile_dynamic": ("pads", "compile_dynamic"),
    "compile_dynamic_frames": ("pads", "compile_dynamic_frames"),
    "compile_regional": ("pads", "compile_regional"),
    "compile_cosy": ("pads", "compile_cosy"),
    "compile_static_cosy": ("pads", "compile_static_cosy"),
    "compile_regional_cosy": ("pads", "compile_regional_cosy"),
    "served": None,
}
COSYVOICE_INDUCTOR_OPTIONS = {"triton.autotune_pointwise": False, "emulate_precision_casts": True}
DEFAULT_VARIANTS = "base,pads,qkv,compile,compile_qkv"
TIMED_CLIPS = 200
TOP_KERNELS = 10
RECOMPILE_LIMIT = 64
ORIGINAL_ENCODE = Qwen3TTSReferenceEncoderCudaGraphRunner._encode


def integer_padding_forward(conv: MimiConv1d):
    """MimiConv1d.forward (transformers 5.19.0, causal, no padding cache) with its padding as Python integers."""
    stride = int(conv.conv.stride[0])
    kernel = (int(conv.conv.kernel_size[0]) - 1) * int(conv.conv.dilation[0]) + 1
    padding_total = kernel - stride
    assert conv.causal, "the encoder's convs are causal"

    def forward(hidden_states, padding_cache=None):
        length = hidden_states.shape[-1]
        frames = -(-(length - kernel + padding_total) // stride)
        extra_padding = frames * stride + kernel - padding_total - length
        if padding_total or extra_padding:
            hidden_states = F.pad(hidden_states, (padding_total, extra_padding), mode=conv.pad_mode)
        else:
            pass
        return conv.conv(hidden_states)

    return forward


def merged_qkv_forward(attention: torch.nn.Module):
    """MimiAttention.forward (transformers 5.19.0) with one qkv GEMM."""
    qkv_weight = torch.cat([attention.q_proj.weight.data, attention.k_proj.weight.data, attention.v_proj.weight.data])
    sizes = [attention.q_proj.out_features, attention.k_proj.out_features, attention.v_proj.out_features]
    start = 0
    for projection, size in zip((attention.q_proj, attention.k_proj, attention.v_proj), sizes):
        projection.weight.data = qkv_weight[start : start + size]
        start += size

    def forward(hidden_states, attention_mask=None, past_key_values=None, position_embeddings=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query, key, value = F.linear(hidden_states, qkv_weight).split(sizes, dim=-1)
        query = query.reshape(hidden_shape).transpose(1, 2)
        key = key.reshape(hidden_shape).transpose(1, 2)
        value = value.reshape(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = modeling_mimi.apply_rotary_pos_emb(query, key, cos, sin)
        interface = modeling_mimi.ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation, modeling_mimi.eager_attention_forward
        )
        output, weights = interface(
            attention, query, key, value, attention_mask, dropout=0.0, scaling=attention.scaling,
            sliding_window=attention.sliding_window, **kwargs,
        )
        return attention.o_proj(output.reshape(*input_shape, -1).contiguous()), weights

    return forward


def set_cosyvoice_compile_config() -> None:
    """Fun-CosyVoice3's set_dit_compile_config (fun_cosyvoice3/stages.py:1447-1455)."""
    inductor_config.fx_graph_cache = True
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    with inductor_config.patch({"eager_numerics.use_pytorch_libdevice": True}):
        _set_triton_libdevice_path()


def compiled_encode(
    encoder: torch.nn.Module,
    dynamic: bool | None,
    frame_multiple: int = 0,
    options: dict[str, bool] | None = None,
    is_length_hinted: bool = False,
):
    def encode_frame(values: torch.Tensor, num_quantizers: int) -> torch.Tensor:
        if frame_multiple:
            torch._check(values.shape[-1] % frame_multiple == 0)
        else:
            pass
        return encoder._encode_frame(values, num_quantizers, None, use_streaming=False, return_dict=True)[0]

    compiled = torch.compile(encode_frame, dynamic=dynamic, fullgraph=True, options=options)

    def runner_encode(self, values: torch.Tensor) -> torch.Tensor:
        if is_length_hinted:
            torch._dynamo.maybe_mark_dynamic(values, 2)
        else:
            pass
        return compiled(values, self.num_quantizers)

    return runner_encode


def codebook_step_eager(embed: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One iteration of MimiResidualVectorQuantizer.encode (transformers 5.19.0): nearest centroid, new residual."""
    hidden = residual.permute(0, 2, 1)
    flat = hidden.reshape(-1, hidden.shape[-1])
    indices = torch.cdist(flat[None].float(), embed[None].float(), p=2)[0].argmin(dim=-1).view(*hidden.shape[:-1])
    return indices, residual - F.embedding(indices, embed).permute(0, 2, 1)


codebook_step = torch.compile(codebook_step_eager, dynamic=True, fullgraph=True)
codebook_step_cosy = torch.compile(codebook_step_eager, fullgraph=True, options=COSYVOICE_INDUCTOR_OPTIONS)


def regional_quantizer_encode(quantizer: MimiResidualVectorQuantizer, step):
    def encode(embeddings: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        if quantizer.input_proj is not None:
            embeddings = quantizer.input_proj(embeddings)
        else:
            pass
        num_quantizers = num_quantizers if num_quantizers is not None else quantizer.num_quantizers
        residual = embeddings
        all_indices = []
        for layer in quantizer.layers[:num_quantizers]:
            indices, residual = step(layer.codebook.embed, residual)
            all_indices.append(indices)
        return torch.stack(all_indices)

    return encode


def install(encoder: torch.nn.Module, mechanisms: tuple[str, ...]) -> None:
    torch._dynamo.reset()
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.__dict__.pop("forward", None)
            if "pads" in mechanisms:
                module.forward = integer_padding_forward(module)
            else:
                pass
        elif isinstance(module, (MimiResnetBlock, MimiTransformerLayer)):
            module._compiled_call_impl = None
            if "compile_regional" in mechanisms:
                module.compile(dynamic=True, fullgraph=True)
            elif "compile_regional_cosy" in mechanisms:
                module.compile(fullgraph=True, options=COSYVOICE_INDUCTOR_OPTIONS)
            else:
                pass
        elif isinstance(module, MimiResidualVectorQuantizer):
            module.__dict__.pop("encode", None)
            if "compile_regional" in mechanisms:
                module.encode = regional_quantizer_encode(module, codebook_step)
            elif "compile_regional_cosy" in mechanisms:
                module.encode = regional_quantizer_encode(module, codebook_step_cosy)
            else:
                pass
        else:
            pass
    if any(mechanism.endswith("_cosy") for mechanism in mechanisms):
        set_cosyvoice_compile_config()
    else:
        pass
    for layer in encoder.encoder_transformer.layers:
        layer.self_attn.__dict__.pop("forward", None)
        if "qkv" in mechanisms:
            layer.self_attn.forward = merged_qkv_forward(layer.self_attn)
        else:
            pass
    if "compile" in mechanisms:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = compiled_encode(encoder, dynamic=False)
    elif "compile_dynamic" in mechanisms:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = compiled_encode(encoder, dynamic=True)
    elif "compile_dynamic_frames" in mechanisms:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = compiled_encode(encoder, dynamic=True, frame_multiple=1920)
    elif "compile_cosy" in mechanisms:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = compiled_encode(
            encoder, dynamic=None, options=COSYVOICE_INDUCTOR_OPTIONS, is_length_hinted=True
        )
    elif "compile_static_cosy" in mechanisms:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = compiled_encode(
            encoder, dynamic=False, options=COSYVOICE_INDUCTOR_OPTIONS
        )
    else:
        Qwen3TTSReferenceEncoderCudaGraphRunner._encode = ORIGINAL_ENCODE


def compare(first: list[torch.Tensor], second: list[torch.Tensor]) -> str:
    equal = total = fully = 0
    for left, right in zip(first, second):
        frames = min(left.shape[0], right.shape[0])
        same = left[:frames] == right[:frames]
        equal += int(same.sum())
        total += same.numel()
        fully += int(bool(same.all()) and left.shape == right.shape)
    return f"codes {equal / total:.4f}, clips fully equal {fully}/{len(first)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--clips", type=int, default=1088)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer

    clips = [librosa.load(str(path), sr=None, mono=True) for path in sorted(args.wavs.glob("*.wav"))[: args.clips]]
    ours = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    official = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(resolve_checkpoint(args.model), "speech_tokenizer"), device_map=device, dtype=torch.float32
    )
    encoder = ours.model.encoder
    print(f"sglang_omni from {sglang_omni.__file__}; clips {len(clips)}, encoder {next(encoder.parameters()).dtype}",
          flush=True)
    with torch.inference_mode():
        reference = [official.encode([waveform], sr=rate).audio_codes[0].cpu() for waveform, rate in clips]
    del official
    baseline: list[torch.Tensor] | None = None
    for name in args.variants.split(","):
        mechanisms = VARIANTS[name]
        if mechanisms is not None:
            install(encoder, tuple(mechanism for mechanism in mechanisms if not mechanism.startswith("compile")))
        else:
            pass
        eager = Qwen3TTSRefCodeBatcher(ours)
        with torch.inference_mode():
            eager_codes = [eager.encode_waveform(waveform, rate).cpu() for waveform, rate in clips]
        eager.close()
        if mechanisms is not None:
            install(encoder, mechanisms)
        else:
            pass
        began = time.perf_counter()
        try:
            graphs = Qwen3TTSRefCodeBatcher(ours, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
        except Exception as exc:
            print(f"== {name}: batcher failed: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            continue
        capture_s = time.perf_counter() - began
        if any(mechanism.startswith("compile") for mechanism in mechanisms or ()):
            print(f"-- {name} compile phases (s):\n{compile_times(repr='str', aggregate=True)}", flush=True)
        else:
            pass
        stats = graphs.graph_runner.stats() if graphs.graph_runner is not None else {}
        if not stats.get("enabled"):
            print(f"== {name}: graphs disabled: {stats.get('disable_reason')}", flush=True)
            graphs.close()
            continue
        else:
            pass
        with torch.inference_mode():
            graph_codes = [graphs.encode_waveform(waveform, rate).cpu() for waveform, rate in clips]
            torch.cuda.synchronize()
            walls_ms = []
            for waveform, rate in clips[:TIMED_CLIPS]:
                started = time.perf_counter()
                graphs.encode_waveform(waveform, rate)
                torch.cuda.synchronize()
                walls_ms.append(1e3 * (time.perf_counter() - started))
            with profile(activities=[ProfilerActivity.CUDA]) as profiler:
                for waveform, rate in clips[:TIMED_CLIPS]:
                    graphs.encode_waveform(waveform, rate)
                torch.cuda.synchronize()
        graphs.close()
        if baseline is None:
            baseline = graph_codes
        else:
            pass
        totals_us: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for event in profiler.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                totals_us[event.name] += event.device_time
                counts[event.name] += 1
            else:
                pass
        encodes = min(len(clips), TIMED_CLIPS)
        print(
            f"== {name}: device {sum(totals_us.values()) / 1e3 / encodes:.3f} ms and "
            f"{sum(counts.values()) / encodes:.1f} kernels per encode, wall q50 {np.median(walls_ms):.2f} ms, "
            f"capture {capture_s:.1f} s",
            flush=True,
        )
        print(f"   official / graphs: {compare(reference, graph_codes)}", flush=True)
        print(f"   official / eager:  {compare(reference, eager_codes)}", flush=True)
        print(f"   base / graphs:     {compare(baseline, graph_codes)}", flush=True)
        for kernel, total_us in sorted(totals_us.items(), key=lambda item: -item[1])[:TOP_KERNELS]:
            print(f"   {total_us / encodes:8.1f} us, {counts[kernel] / encodes:5.1f} calls: {kernel[:100]}", flush=True)
    install(encoder, ())


if __name__ == "__main__":
    main()
