"""Apply an audited checkpoint predictor at the actual post-graph feedback boundary."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import GenerationConfig, Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from sglang_omni.scheduling.types import SchedulerRequest


class PredictorControlMode(str, Enum):
    IDENTITY = "identity"
    CHECKPOINT = "checkpoint"
    CHECKPOINT_CODES = "checkpoint-codes"
    CHECKPOINT_FEEDBACK = "checkpoint-feedback"


class CheckpointPredictorControl:
    """Keep native main draws while optionally replacing the residual-code recurrence."""

    def __init__(
        self,
        *,
        mode: PredictorControlMode,
        model_path: str,
        model_revision: str,
        main_embedding: nn.Embedding,
        request_ids: frozenset[str],
        directory: Path,
    ) -> None:
        self.mode: PredictorControlMode = mode
        self.main_embedding: nn.Embedding = main_embedding
        self.request_ids: frozenset[str] = request_ids
        self.directory: Path = directory
        self.directory.mkdir()
        configuration = Qwen3OmniMoeConfig.from_pretrained(
            model_path, revision=model_revision
        ).talker_config.code_predictor_config
        device = main_embedding.weight.device
        self.predictor: Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
        self.predictor, loading = (
            Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration.from_pretrained(
                model_path,
                revision=model_revision,
                config=configuration,
                generation_config=GenerationConfig.from_model_config(configuration),
                dtype=torch.bfloat16,
                device_map={"": device.index},
                key_mapping={r"^talker\.code_predictor\.": ""},
                attn_implementation="eager",
                use_kernels=False,
                output_loading_info=True,
            )
        )
        assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
        self.predictor.eval()
        derived_buffers = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.predictor.named_buffers()
        }
        torch.save(derived_buffers, self.directory / "loaded-buffers.pt")
        checkpoint_directory = Path(
            hf_hub_download(model_path, "config.json", revision=model_revision)
        ).parent
        weight_map = json.loads(
            (checkpoint_directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        expected_names = {
            name.removeprefix("talker.code_predictor.")
            for name in weight_map
            if name.startswith("talker.code_predictor.")
        }
        parameters = self.predictor.state_dict()
        assert set(parameters) == expected_names
        parameters["main_embedding.weight"] = main_embedding.weight
        reports = []
        for name, actual in parameters.items():
            checkpoint_name = (
                "talker.model.codec_embedding.weight"
                if name == "main_embedding.weight"
                else f"talker.code_predictor.{name}"
            )
            with safe_open(
                checkpoint_directory / weight_map[checkpoint_name],
                framework="pt",
                device="cpu",
            ) as checkpoint:
                expected = checkpoint.get_tensor(checkpoint_name)
            assert actual.shape == expected.shape, name
            assert actual.dtype == expected.dtype == torch.bfloat16, name
            assert torch.equal(actual, expected.to(device)), name
            reports.append({"name": checkpoint_name, "exact": True})
        (self.directory / "loaded-weights.json").write_text(
            json.dumps(
                {
                    "model_path": model_path,
                    "model_revision": model_revision,
                    "mode": mode.value,
                    "generation_config": self.predictor.generation_config.to_dict(),
                    "parameters": reports,
                    "derived_buffers": {
                        name: {
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "sha256": hashlib.sha256(
                                tensor.contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest(),
                        }
                        for name, tensor in derived_buffers.items()
                    },
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "scope": "Partial checkpoint predictor and actual native main embedding, every raw learned tensor checked without casting. No global numerical/backend flag is changed.",
                },
                indent=2,
            )
        )

    @torch.inference_mode()
    def apply(
        self,
        requests: list[SchedulerRequest],
        codes: torch.Tensor,
        feedback: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        for row, request in enumerate(requests):
            if request.request_id not in self.request_ids:
                continue
            else:
                pass
            assert len(requests) == 1
            native_codes = codes[row].clone()
            native_feedback = feedback[row].clone()
            query = hidden_states[row : row + 1].unsqueeze(1)
            main_embedding = self.main_embedding(native_codes[:1].view(1, 1))
            assert query.dtype == main_embedding.dtype == torch.bfloat16
            predicted = self.predictor.generate(
                inputs_embeds=torch.cat((query, main_embedding), dim=1),
                max_new_tokens=codes.shape[1] - 1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            reference_codes = torch.cat(
                (native_codes[:1].view(1, 1), predicted.sequences), dim=-1
            )
            reference_rows = [main_embedding]
            reference_rows.extend(hidden[0] for hidden in predicted.hidden_states[1:])
            reference_rows.append(
                self.predictor.model.codec_embedding[-1](predicted.sequences[:, -1:])
            )
            reference_embeddings = torch.cat(reference_rows, dim=1)
            assert reference_embeddings.shape[1] == codes.shape[1]
            for group, embedding in enumerate(self.predictor.model.codec_embedding):
                assert torch.equal(
                    reference_embeddings[:, group + 1 : group + 2],
                    embedding(predicted.sequences[:, group : group + 1]),
                )
            reference_feedback = reference_embeddings.sum(dim=1)
            assert reference_codes.shape == (1, codes.shape[1])
            assert reference_feedback.shape == feedback[row : row + 1].shape
            assert reference_feedback.isfinite().all()
            if self.mode is PredictorControlMode.CHECKPOINT:
                selected_codes = reference_codes
                selected_feedback = reference_feedback
            elif self.mode is PredictorControlMode.CHECKPOINT_CODES:
                selected_codes = reference_codes
                selected_feedback = reference_embeddings[:, 0].clone()
                for group in range(1, reference_embeddings.shape[1]):
                    selected_feedback.add_(reference_embeddings[:, group])
            elif self.mode is PredictorControlMode.CHECKPOINT_FEEDBACK:
                selected_codes = native_codes[None]
                native_rows = [main_embedding]
                native_rows.extend(
                    embedding(native_codes[group + 1 : group + 2].view(1, 1))
                    for group, embedding in enumerate(
                        self.predictor.model.codec_embedding
                    )
                )
                selected_feedback = torch.cat(native_rows, dim=1).sum(dim=1)
            else:
                selected_codes = native_codes[None]
                selected_feedback = native_feedback[None]
            destination = self.directory / request.request_id
            destination.mkdir(exist_ok=True)
            torch.save(
                {
                    "query": query.cpu(),
                    "native_codes": native_codes.cpu(),
                    "native_feedback": native_feedback.cpu(),
                    "reference_codes": reference_codes.cpu(),
                    "reference_feedback": reference_feedback.cpu(),
                    "reference_embedding_rows": reference_embeddings.cpu(),
                    "selected_codes": selected_codes.cpu(),
                    "selected_feedback": selected_feedback.cpu(),
                },
                destination / f"{len(request.data.req.output_ids):04d}.pt",
            )
            if self.mode is not PredictorControlMode.IDENTITY:
                codes[row].copy_(selected_codes[0])
                feedback[row].copy_(selected_feedback[0])
            else:
                pass
