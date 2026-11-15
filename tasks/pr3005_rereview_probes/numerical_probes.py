"""Compare checkpoint-backed acoustic components with the original implementation."""

import gc
import json
from pathlib import Path

import torch
from fireredtts3.llm.dit import DiT as OriginalDiT
from fireredtts3.llm.modules import RMSNorm as OriginalRMSNorm
from fireredtts3.llm.patch_encoder import PatchEncoder as OriginalPatchEncoder
from huggingface_hub import snapshot_download

from sglang_omni.models.firered_tts3.flow_execution import euler_flow_loop, flow_t_span
from sglang_omni.models.firered_tts3.flow_graph import FlowGraphRunner
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.weights import (
    build_dit,
    build_patch_encoder,
    read_component_tensors,
)


def emit_comparison(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-5,
) -> None:
    difference = actual.double() - expected.double()
    print(
        json.dumps(
            {
                "probe": label,
                "shape": list(actual.shape),
                "max_abs": difference.abs().max().item(),
                "rmse": difference.square().mean().sqrt().item(),
                "byte_equal": torch.equal(actual, expected),
                "rtol": relative_tolerance,
                "atol": absolute_tolerance,
            }
        ),
        flush=True,
    )
    torch.testing.assert_close(
        actual, expected, rtol=relative_tolerance, atol=absolute_tolerance
    )


def select_native_rmsnorm(model: torch.nn.Module) -> None:
    """Use the original Torch >=2.4 path; its string parser mistakes 2.14 for 2.1."""
    for module in model.modules():
        if isinstance(module, OriginalRMSNorm):
            module.native_rms_norm = True
        else:
            pass


def original_euler(
    model: OriginalDiT,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    timestep_knots: torch.Tensor,
    guidance_scales: torch.Tensor,
) -> torch.Tensor:
    flow_state = initial_state.clone()
    batch_size = flow_state.shape[0]
    for index, timestep in enumerate(timestep_knots[:-1]):
        timestep_delta = timestep_knots[index + 1] - timestep
        conditional_inputs = torch.cat([flow_state, conditioning], dim=2)
        unconditional_inputs = torch.cat([flow_state, conditioning * 0], dim=2)
        inputs = torch.cat([conditional_inputs, unconditional_inputs], dim=0)
        timesteps = timestep.view(-1, 1, 1).expand(2 * batch_size, -1, -1)
        velocity = model(x=inputs, t=timesteps)
        conditional_velocity, unconditional_velocity = velocity.chunk(2, dim=0)
        velocity = (
            1 + guidance_scales
        ) * conditional_velocity - guidance_scales * unconditional_velocity
        flow_state[:, -4:] = (
            flow_state[:, -4:] + timestep_delta.view(-1, 1, 1) * velocity[:, -4:]
        )
    return flow_state[:, -4:]


@torch.inference_mode()
def compare_variant(variant: str) -> None:
    directory = Path(
        snapshot_download(
            f"CKermit/FireRedTTS3-{variant}",
            revision={
                "Base": "53f8b59c3252f15d539a9205fbcab3e334a36b9f",
                "Instruct": "edd602371bd356a974dfa8ed7eed949ec4c398b8",
            }[variant],
            local_files_only=True,
        )
    )
    configuration = FireRedTTS3Config(
        **json.loads((directory / "config.json").read_text())
    )
    original_dit = OriginalDiT(
        in_channels=configuration.dit_in_channels,
        out_channels=configuration.redae_dim,
        mlp_ratio=configuration.dit_mlp_ratio,
        depth=configuration.dit_depth,
        num_heads=configuration.dit_num_heads,
        hidden_size=configuration.dit_hidden_size,
    ).eval()
    adapted_dit = build_dit(configuration).eval()
    weights = read_component_tensors(directory, "dit.")
    original_dit.load_state_dict(weights, strict=True)
    adapted_dit.load_state_dict(weights, strict=True)
    select_native_rmsnorm(original_dit)
    del weights
    original_dit.cuda()
    adapted_dit.cuda()
    timestep_knots = flow_t_span(10, torch.device("cuda:0"))
    for batch_size in (1, 2):
        torch.manual_seed(912 + batch_size)
        initial_state = torch.randn(batch_size, 12, 64, device="cuda")
        conditioning = torch.randn(
            batch_size, 12, configuration.dit_in_channels - 64, device="cuda"
        )
        guidance_scales = torch.tensor([0.0, 2.0][:batch_size], device="cuda").reshape(
            batch_size, 1, 1
        )
        model_inputs = torch.cat([initial_state, conditioning], dim=2)
        timesteps = torch.full((batch_size, 1, 1), 0.4, device="cuda")
        emit_comparison(
            f"{variant}_dit_b{batch_size}",
            adapted_dit(model_inputs, timesteps),
            original_dit(model_inputs, timesteps),
        )
        expected = original_euler(
            original_dit, initial_state, conditioning, timestep_knots, guidance_scales
        )
        actual = euler_flow_loop(
            adapted_dit,
            initial_state.clone(),
            conditioning,
            timestep_knots,
            guidance_scales,
            10,
            4,
        )
        emit_comparison(f"{variant}_euler_b{batch_size}", actual, expected)
        graph_runner = FlowGraphRunner(
            adapted_dit,
            timestep_knots,
            n_timesteps=10,
            patch_size=4,
            history_length=8,
            max_graphs=1,
            capture_sizes=[2],
            mode="fp32",
        )
        replayed = graph_runner.run(
            initial_state[:, :8],
            initial_state[:, 8:],
            conditioning,
            guidance_scales,
            10,
        )
        assert replayed is not None
        padded_state = torch.cat(
            [initial_state, initial_state.new_zeros(2 - batch_size, 12, 64)]
        )
        padded_conditioning = torch.cat(
            [
                conditioning,
                conditioning.new_zeros(2 - batch_size, 12, conditioning.shape[-1]),
            ]
        )
        padded_guidance = torch.cat(
            [guidance_scales, guidance_scales.new_zeros(2 - batch_size, 1, 1)]
        )
        bucket_reference = original_euler(
            original_dit,
            padded_state,
            padded_conditioning,
            timestep_knots,
            padded_guidance,
        )[:batch_size]
        emit_comparison(
            f"{variant}_fp32_graph_bucket_reference_b{batch_size}",
            replayed,
            bucket_reference,
        )
        emit_comparison(
            f"{variant}_bucket_padding_error_b{batch_size}",
            replayed,
            expected,
            relative_tolerance=1e-2,
            absolute_tolerance=1e-3,
        )
        del graph_runner
    del original_dit, adapted_dit
    original_patch = OriginalPatchEncoder(
        in_dim=configuration.redae_dim,
        out_dim=configuration.hidden_size,
        patch_size=configuration.patch_size,
        hidden_size=configuration.patch_encoder_hidden_size,
        mlp_ratio=configuration.patch_encoder_mlp_ratio,
        depth=configuration.patch_encoder_depth,
        num_heads=configuration.patch_encoder_num_heads,
    ).eval()
    adapted_patch = build_patch_encoder(configuration).eval()
    weights = read_component_tensors(directory, "patch_encoder.")
    original_patch.load_state_dict(weights, strict=True)
    adapted_patch.load_state_dict(weights, strict=True)
    select_native_rmsnorm(original_patch)
    del weights
    original_patch.cuda()
    adapted_patch.cuda()
    for frame_count in (4, 12, 40):
        inputs = torch.randn(1, frame_count, configuration.redae_dim, device="cuda")
        emit_comparison(
            f"{variant}_patch_frames_{frame_count}",
            adapted_patch(inputs),
            original_patch(inputs),
        )
    del original_patch, adapted_patch
    gc.collect()


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "original_version_parser": float(torch.__version__[:3]),
                "original_rmsnorm_path": "native, intended for Torch >=2.4",
            }
        ),
        flush=True,
    )
    torch.manual_seed(912)
    compare_variant("Base")
    compare_variant("Instruct")
