"""Replay actual residual predictor histories against retained native cache bytes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from sglang.srt.layers.layernorm import RMSNorm
from torch import nn
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_native_weights import (
    NativeBufferRuntime,
    NativeManifest,
    audit_native_talker_parameters,
)
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.utils.predictor_layers import FusedPredictorLayers, codebook_step


class NativePredictorReplay:
    """Reconstruct the observed single-request predictor from audited checkpoint weights."""

    def __init__(
        self,
        reference: Qwen3OmniMoeTalkerForConditionalGeneration,
        capture_directory: Path,
        output_directory: Path,
        model_path: str,
        model_revision: str,
    ) -> None:
        runtime = NativeBufferRuntime.model_validate_json(
            (capture_directory / "native-talker-buffer-runtime.json").read_bytes()
        )
        assert (
            runtime.predictor_fused_layers and runtime.predictor_layer_shape is not None
        )
        checkpoint_directory = Path(
            snapshot_download(model_path, revision=model_revision)
        )
        audit_checkpoint_weights(
            reference,
            checkpoint_directory,
            output_directory / "raw-checkpoint-weights.json",
        )
        manifest_bytes = (
            capture_directory / "native-talker-parameters.json"
        ).read_bytes()
        audit_native_talker_parameters(
            reference,
            NativeManifest.model_validate_json(manifest_bytes),
            output_directory / "native-checkpoint-weights.json",
            model_path=model_path,
            model_revision=model_revision,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
        configuration = reference.config
        predictor_configuration = configuration.code_predictor_config
        shape = runtime.predictor_layer_shape
        assert shape.hidden_size == predictor_configuration.hidden_size
        assert shape.head_dim == predictor_configuration.head_dim
        assert shape.num_q_heads == predictor_configuration.num_attention_heads
        assert shape.num_kv_heads == predictor_configuration.num_key_value_heads
        assert shape.intermediate_size == predictor_configuration.intermediate_size
        captured_buffers = torch.load(
            capture_directory / "native-talker-buffers.pt", weights_only=True
        )
        table_name = next(
            name
            for name in captured_buffers
            if name.startswith("code_predictor.") and name.endswith(".cos_sin_cache")
        )
        rotary_table = captured_buffers[table_name].to(reference.device)
        assert rotary_table.dtype == torch.float32
        self.reference: Qwen3OmniMoeTalkerForConditionalGeneration = reference
        self.layers: nn.ModuleList = reference.code_predictor.model.layers
        for layer in self.layers:
            attention = layer.self_attn
            attention.qkv_proj = nn.Linear(
                shape.hidden_size,
                (shape.num_q_heads + 2 * shape.num_kv_heads) * shape.head_dim,
                bias=False,
                device="meta",
            )
            attention.qkv_proj.weight = nn.Parameter(
                torch.cat(
                    [
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ],
                    dim=0,
                ),
                requires_grad=False,
            )
            attention.rotary_emb = nn.Module()
            attention.rotary_emb.register_buffer(
                "cos_sin_cache", rotary_table, persistent=False
            )
            layer.mlp.gate_up_proj = nn.Linear(
                shape.hidden_size,
                2 * shape.intermediate_size,
                bias=False,
                device="meta",
            )
            layer.mlp.gate_up_proj.weight = nn.Parameter(
                torch.cat(
                    [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0
                ),
                requires_grad=False,
            )
        self.final_norm: RMSNorm = RMSNorm(
            shape.hidden_size, eps=predictor_configuration.rms_norm_eps
        )
        self.final_norm.weight = reference.code_predictor.model.norm.weight
        self.fused: FusedPredictorLayers = FusedPredictorLayers(
            shape, 2, reference.device, torch.bfloat16
        )
        self.group_count: int = configuration.num_code_groups
        self.keys: torch.Tensor = torch.zeros(
            len(self.layers),
            1,
            self.group_count + 1,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=torch.bfloat16,
            device=reference.device,
        )
        self.values: torch.Tensor = torch.zeros_like(self.keys)

    def forward_tokens(
        self, embeddings: torch.Tensor, cache_length: int
    ) -> torch.Tensor:
        positions = torch.arange(
            cache_length,
            cache_length + embeddings.shape[1],
            dtype=torch.long,
            device=embeddings.device,
        )
        return self.fused.forward(
            layers=self.layers,
            final_norm=self.final_norm,
            token_embeds=embeddings,
            batch_size=1,
            cache_len=cache_length,
            positions=positions,
            k_cache=self.keys,
            v_cache=self.values,
        )

    def generate(
        self, main_code: torch.Tensor, talker_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert main_code.shape == (1,) and talker_hidden.shape[:2] == (1, 1)
        codes = torch.empty(
            (1, self.group_count), dtype=torch.long, device=main_code.device
        )
        codes[:, 0].copy_(main_code)
        main_embedding = self.reference.model.codec_embedding(main_code[:, None])
        feedback = main_embedding[:, 0].clone()
        embeddings = torch.cat((talker_hidden, main_embedding), dim=1)
        cache_length = 0
        for group in range(self.group_count - 1):
            output = self.forward_tokens(embeddings, cache_length)
            cache_length += embeddings.shape[1]
            logits = self.reference.code_predictor.lm_head[group](output[:, -1:])
            assert logits.isfinite().all()
            embeddings = codebook_step(
                logits[:, -1],
                self.reference.code_predictor.model.codec_embedding[group].weight,
                codes[:, group + 1],
            )
            feedback.add_(embeddings[:, 0])
        assert cache_length == self.group_count
        return codes, feedback[:, None]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        configuration.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    predictor = NativePredictorReplay(
        reference=reference,
        capture_directory=arguments.capture_directory,
        output_directory=arguments.output_directory,
        model_path=arguments.model_path,
        model_revision=arguments.model_revision,
    )
    group_count = configuration.num_code_groups
    keys, values = predictor.keys, predictor.values
    reports = []
    for directory in sorted(arguments.capture_directory.iterdir()):
        if not directory.is_dir():
            continue
        else:
            pass
        finished = torch.load(directory / "finished.pt", weights_only=True)
        frame_count = len(finished["output_ids"])
        snapshots = sorted(directory.glob("*-predictor.pt"))
        assert [path.name for path in snapshots] == [
            f"{frame:04d}-predictor.pt" for frame in range(frame_count)
        ]
        keys.zero_()
        values.zero_()
        for frame, snapshot_file in enumerate(snapshots):
            observed = torch.load(
                directory / f"{frame:04d}-output.pt", weights_only=True
            )
            snapshot = torch.load(snapshot_file, weights_only=True)
            assert int(observed["codec_ids"][0]) == int(finished["output_ids"][frame])
            codes = observed["codec_ids"].to(device=reference.device, dtype=torch.long)
            assert codes.numel() == group_count
            hidden = observed["hidden_states"].to(reference.device).reshape(1, 1, -1)
            assert hidden.shape[-1] == configuration.code_predictor_config.hidden_size
            main_embedding = reference.model.codec_embedding(codes[:1].reshape(1, 1))
            inputs = torch.cat((hidden, main_embedding), dim=1)
            feedback = main_embedding[:, 0].clone()
            predictions = []
            cache_length = 0
            for group in range(group_count - 1):
                token_count = inputs.shape[1]
                output = predictor.forward_tokens(inputs, cache_length)
                cache_length += token_count
                logits = reference.code_predictor.lm_head[group](output[:, -1:])
                assert logits.isfinite().all()
                predictions.append(int(logits.argmax(-1)))
                inputs = reference.code_predictor.model.codec_embedding[group](
                    codes[group + 1 : group + 2].reshape(1, 1)
                )
                feedback.add_(inputs[:, 0])
            assert cache_length == group_count
            cache_checks = []
            for name, cache in (("keys", keys), ("values", values)):
                expected = snapshot[name][:, :group_count].to(reference.device)
                actual = cache[:, 0, :group_count]
                assert actual.shape == expected.shape
                cache_checks.append(
                    {
                        "name": name,
                        "exact": torch.equal(actual, expected),
                        "different_values": int((actual != expected).sum()),
                        "maximum_absolute_error": float(
                            (actual - expected).abs().max()
                        ),
                    }
                )
            held_counters_zero = not bool(predictor.fused.counters.count_nonzero())
            free_codes, free_feedback = predictor.generate(codes[:1], hidden)
            generated_cache_exact = all(
                torch.equal(
                    cache[:, 0, :group_count],
                    snapshot[name][:, :group_count].to(reference.device),
                )
                for name, cache in (("keys", keys), ("values", values))
            )
            reports.append(
                {
                    "request_id": directory.name,
                    "frame": frame,
                    "is_eos": int(codes[0]) == configuration.codec_eos_token_id,
                    "native_residual_codes": codes[1:].tolist(),
                    "replayed_argmax_codes": predictions,
                    "codes_exact": predictions == codes[1:].tolist(),
                    "generated_codes_exact": torch.equal(free_codes[0], codes),
                    "generated_cache_exact": generated_cache_exact,
                    "generated_feedback_exact": torch.equal(
                        free_feedback[0, 0].cpu(), observed["codec_feedback"]
                    ),
                    "feedback_exact": torch.equal(
                        feedback[0].cpu(), observed["codec_feedback"]
                    ),
                    "native_counters_zero": not bool(
                        snapshot["fused_counters"].count_nonzero()
                    ),
                    "replayed_counters_zero": held_counters_zero,
                    "generated_counters_zero": not bool(
                        predictor.fused.counters.count_nonzero()
                    ),
                    "cache_checks": cache_checks,
                }
            )
    assert reports
    kernel_source = Path(inspect.getfile(FusedPredictorLayers))
    report = {
        "model_path": arguments.model_path,
        "model_revision": arguments.model_revision,
        "kernel_source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
        "frames": reports,
        "scope": "Actual recorded native backbone queries, main codes and residual-code prefixes are held for every frame through EOS. Independently raw-audited checkpoint weights are packed for the unchanged native fused kernels, using captured rotary-table bytes and recorded split geometry. Every consumed predictor KV slot, generated residual argmax, sequential BF16 feedback and scratch-counter reset is checked. The allocated but never consumed last cache slot is excluded. This is native kernel/state reproducibility, not an independent mathematical or free-running speech-quality oracle. No new main-code sampling or waveforms.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    assert all(
        frame["codes_exact"]
        and frame["generated_codes_exact"]
        and frame["generated_cache_exact"]
        and frame["generated_feedback_exact"]
        and frame["feedback_exact"]
        and frame["native_counters_zero"]
        and frame["replayed_counters_zero"]
        and frame["generated_counters_zero"]
        and all(cache["exact"] for cache in frame["cache_checks"])
        for frame in reports
    ), "Retained full predictor comparison differs; preserve the complete report."
    print(json.dumps({"frames": len(reports), "all_checks_exact": True}), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
