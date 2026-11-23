# SPDX-License-Identifier: Apache-2.0
"""Hold captured future text fixed after the native Thinker has completed."""

import json
from hashlib import sha256
from pathlib import Path

import torch

from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

RESERVED_CODEC_TOKENS = 1024


class NativeConditioningTrial:
    def __init__(self, reference: Path, destination: Path) -> None:
        self.request_id = f"feedback-trace-native-{reference.name}"
        self.destination = destination
        self.conditioning: dict[str, torch.Tensor] = torch.load(
            reference / "conditioning.pt", weights_only=True, map_location="cpu"
        )
        receipt = json.loads((reference / "conditioning.json").read_text())
        self.policy: dict[str, int | float | str] = receipt["sampling_policy"]
        assert receipt["native_frames"] == receipt["verified_native_draws"]
        assert receipt["native_frames"] == len(self.conditioning["codec_ids"])
        assert receipt["prefill_tokens"] == len(self.conditioning["input_ids"])
        assert receipt["complete_consumed_text_rows"] == len(
            self.conditioning["text_embeddings"]
        )
        self.applied = False
        self.opening_verified = False
        self.checks: dict[str, bool] = {}
        self.outcomes: dict[str, int | bool] = {}
        self.provenance: dict[str, str] = {
            "request_id": self.request_id,
            "conditioning_sha256": sha256(
                (reference / "conditioning.pt").read_bytes()
            ).hexdigest(),
            "conditioning_receipt_sha256": sha256(
                (reference / "conditioning.json").read_bytes()
            ).hexdigest(),
            "original_capture_sha256": receipt["capture_sha256"],
            "scope": "Future text replacement only; opening state must match. Score against original captured text, not the live returned answer. No performance qualification.",
        }

    def record_checks(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.destination.write_text(
            json.dumps(
                {**self.provenance, "checks": self.checks, "outcomes": self.outcomes},
                indent=2,
            )
            + "\n"
        )
        if all(self.checks.values()):
            pass
        else:
            torch.save(snapshot, self.destination.with_suffix(".failure.pt"))
            raise AssertionError(
                f"Fixed-conditioning control failed: {self.request_id}"
            )

    def before_prefill(
        self, request: SGLangARRequestData, snapshot: dict[str, torch.Tensor]
    ) -> None:
        assert not self.applied
        assert request.thinker_chunks_done
        assert not request.req.output_ids and not request.pending_feedback_queue
        assert request.tts_pad_embed is not None and request.tts_eos_embed is not None
        sampling = request.req.sampling_params
        actual_policy = {
            "seed": sampling.sampling_seed,
            "temperature": sampling.temperature,
            "repetition_penalty": sampling.repetition_penalty,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
        }
        self.checks["sampling_policy"] = (
            all(value == self.policy[name] for name, value in actual_policy.items())
            and sampling.min_p == 0.0
        )
        expected_suppressed = set(
            range(
                int(self.policy["vocabulary_tokens"]) - RESERVED_CODEC_TOKENS,
                int(self.policy["vocabulary_tokens"]),
            )
        ) - {int(self.policy["eos_token"])}
        self.checks["suppression"] = set(request.suppress_tokens) == expected_suppressed
        self.checks["codec_eos"] = sampling.stop_token_ids == {
            int(self.policy["eos_token"])
        }
        for name in ("prefill_embeddings", "input_ids"):
            actual = snapshot[name].cpu()
            expected = self.conditioning[name]
            self.checks[name] = actual.dtype == expected.dtype and torch.equal(
                actual, expected
            )
        positions = torch.arange(len(self.conditioning["input_ids"]))
        self.checks["positions"] = torch.equal(snapshot["positions"].cpu(), positions)
        self.checks["rotary_positions"] = torch.equal(
            snapshot["rotary_positions"].cpu(), positions.expand(3, -1)
        )
        for name, actual, expected in (
            ("padding", request.tts_pad_embed, self.conditioning["padding_embedding"]),
            (
                "text_eos",
                request.tts_eos_embed,
                self.conditioning["text_embeddings"][-1],
            ),
        ):
            self.checks[name] = actual.dtype == expected.dtype and torch.equal(
                actual.cpu().reshape(-1), expected.reshape(-1)
            )
        self.record_checks(snapshot)
        request.pending_text_queue = PendingTextTensorQueue.from_tensor(
            self.conditioning["text_embeddings"].to(request.tts_pad_embed.device)
        )
        self.applied = True

    def after_prefill(self, snapshot: dict[str, torch.Tensor]) -> None:
        assert self.applied and not self.opening_verified
        for name, expected in (
            ("stored_keys", self.conditioning["prefill_keys"]),
            ("stored_values", self.conditioning["prefill_values"]),
            ("codec_ids", self.conditioning["codec_ids"][0]),
            ("feedback", self.conditioning["first_feedback"]),
        ):
            actual = snapshot[name].cpu()
            self.checks[f"opening_{name}"] = (
                actual.dtype == expected.dtype and torch.equal(actual, expected)
            )
        self.record_checks(snapshot)
        self.opening_verified = True

    def finish(self, snapshots: list[dict[str, torch.Tensor]]) -> None:
        assert self.applied and self.opening_verified
        text_embeddings = self.conditioning["text_embeddings"]
        self.outcomes = {
            "emitted_frames": len(snapshots),
            "ended_with_codec_eos": int(snapshots[-1]["codec_ids"][0])
            == self.policy["eos_token"],
            "text_rows_available": len(text_embeddings),
            "complete_text_consumed": len(snapshots) > len(text_embeddings),
        }
        for frame_index, snapshot in enumerate(snapshots[1:], start=1):
            expected = (
                text_embeddings[frame_index - 1]
                if frame_index <= len(text_embeddings)
                else self.conditioning["padding_embedding"]
            )
            actual = snapshot["text_embedding"].cpu()
            self.checks[f"text_row_{frame_index}"] = (
                actual.dtype == expected.dtype
                and torch.equal(actual.reshape(-1), expected.reshape(-1))
            )
            if self.checks[f"text_row_{frame_index}"]:
                pass
            else:
                self.record_checks(snapshot)
        self.record_checks(snapshots[-1])
