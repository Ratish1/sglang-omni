# SPDX-License-Identifier: Apache-2.0
"""Run matched layer-zero normalizer interventions over distinct SeedTTS contexts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
import torch
import yaml

from benchmarks.benchmarker.utils import (
    get_wav_duration,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    load_qwen3_asr,
    transcribe,
)
from sglang_omni.models.qwen3_omni.boundary_witness import WitnessControl
from sglang_omni.utils import find_available_port

SERVER_START_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 300
EXPECTED_SEEDTTS_SAMPLES = 1088
HISTORICAL_SEVERE_WER = 0.5
MODERATE_CASE_COUNT = 8
CLEAN_CASE_COUNT = 9
HELD_SAMPLE_ID = "common_voice_en_19690002-common_voice_en_19690001"
HELD_AUDIO_SHA256 = "b3180f36dadd42f6381c106b0538b842a1fb38f684e44c8a5674f9f1b87e247c"
NATIVE_AUDIT_CLIENT_CONCURRENCY = 16
PREFILL_PROBE_TOKEN_BY_SAMPLE = {
    "common_voice_en_19756059-common_voice_en_19756062": 117,
    "common_voice_en_17817002-common_voice_en_17817003": 99,
}


def post_seedtts_sample(
    client: httpx.Client,
    endpoint: str,
    sample: SampleInput,
    request_id: str,
    start_barrier: Barrier | None = None,
) -> httpx.Response:
    prompt = (
        f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
        f"Now please read the following text out loud in the same voice and style: "
        f"{sample.target_text}"
    )
    if start_barrier is not None:
        start_barrier.wait(timeout=REQUEST_TIMEOUT_SECONDS)
    else:
        pass
    return client.post(
        endpoint,
        json={
            "model": "qwen3-omni",
            "request_id": request_id,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
            "audios": [sample.ref_audio],
        },
    )


def select_cases(
    h200_history: Path, h100_history: Path
) -> list[dict[str, str | float]]:
    h200_reports = json.loads(h200_history.read_bytes())["reports"]
    h100_records = json.loads(h100_history.read_bytes())["reports"]["h100-main-full"][
        "records"
    ]
    cases = []
    selected = set()
    for name in ("factor-full-main-a", "factor-full-main-b"):
        for row in h200_reports[name]["records"]:
            if row["spoken_wer"] > HISTORICAL_SEVERE_WER and row["id"] not in selected:
                cases.append(
                    {
                        "sample_id": row["id"],
                        "request_id": row["request_id"],
                        "cohort": "historic_severe",
                        "source": name,
                        "historic_spoken_wer": row["spoken_wer"],
                    }
                )
                selected.add(row["id"])
            else:
                pass
    assert len(cases) == 7
    moderate = sorted(
        (
            row
            for row in h100_records
            if row["id"] not in selected
            and 0 < row["spoken_wer"] <= HISTORICAL_SEVERE_WER
            and row["matches_requested_normalized"]
        ),
        key=lambda row: (-row["spoken_wer"], row["id"]),
    )[:MODERATE_CASE_COUNT]
    for row in moderate:
        cases.append(
            {
                "sample_id": row["id"],
                "request_id": row["request_id"],
                "cohort": "historic_moderate",
                "source": "h100-main-full",
                "historic_spoken_wer": row["spoken_wer"],
            }
        )
        selected.add(row["id"])
    clean = sorted(
        (
            row
            for row in h100_records
            if row["id"] not in selected
            and row["spoken_wer"] == 0
            and row["matches_requested_normalized"]
        ),
        key=lambda row: (row["audio_duration_s"], row["id"]),
    )
    for index in range(CLEAN_CASE_COUNT):
        row = clean[len(clean) * (2 * index + 1) // (2 * CLEAN_CASE_COUNT)]
        cases.append(
            {
                "sample_id": row["id"],
                "request_id": row["request_id"],
                "cohort": "historic_clean",
                "source": "h100-main-full",
                "historic_spoken_wer": row["spoken_wer"],
            }
        )
    assert (
        len({row["sample_id"] for row in cases})
        == 7 + MODERATE_CASE_COUNT + CLEAN_CASE_COUNT
    )
    return cases


def controlled_history(
    trace_directory: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor], bool]:
    frames = [
        torch.load(path, weights_only=True)["codec_ids"].reshape(-1)
        for path in sorted(trace_directory.glob("codec-[0-9][0-9][0-9].pt"))
    ]
    text_rows = []
    for path in sorted(trace_directory.glob("backbone-decode-*.pt")):
        row = torch.load(path, weights_only=True)
        if not row["uses_pad"]:
            text_rows.append(row["text_row"].reshape(-1))
        else:
            pass
    finish = torch.load(trace_directory / "backbone-finish-000.pt", weights_only=True)
    text_rows.extend(finish["pending_rows"])
    complete = (
        bool(finish["thinker_done"]) and len(frames) == finish["codec_frames_seen"]
    )
    return frames, text_rows, complete


def compare_pair(
    baseline: Path, candidate: Path, held: bool
) -> dict[str, int | bool | None]:
    if not all(
        (directory / "first-decode-layers.pt").is_file()
        for directory in (baseline, candidate)
    ):
        return {"opening_state_equal": False, "future_text_equal": False}
    else:
        pass
    opening_equal: dict[str, bool] = {}
    for name in (
        "backbone-prefill-000.pt",
        "first-decode-prefix-cache-000.pt",
        "backbone-decode-001.pt",
        "codec-000.pt",
        "decode-sampling-001.pt",
    ):
        before = torch.load(baseline / name, weights_only=True)
        after = torch.load(candidate / name, weights_only=True)
        assert before.keys() == after.keys()
        fields = before.keys() - {"raw_logits", "sampled_code", "predictor_query"}
        opening_equal[name] = all(
            (
                torch.equal(value, after[key])
                if isinstance(value, torch.Tensor)
                else value == after[key]
            )
            for key in fields
            if (value := before[key]) is not None or after[key] is not None
        )
    before_layers = torch.load(baseline / "first-decode-layers.pt", weights_only=True)
    after_layers = torch.load(candidate / "first-decode-layers.pt", weights_only=True)
    assert not before_layers[0]["norm_swap_applied"]
    assert after_layers[0]["norm_swap_applied"]
    assert torch.equal(
        before_layers[0]["attention_normalized"],
        before_layers[0]["attention_normalized_before_override"],
    )
    before_frames, before_text, before_complete = controlled_history(baseline)
    after_frames, after_text, after_complete = controlled_history(candidate)
    first_frame_change = next(
        (
            index
            for index, (left, right) in enumerate(zip(before_frames, after_frames))
            if not torch.equal(left, right)
        ),
        None,
    )
    if first_frame_change is None and len(before_frames) != len(after_frames):
        first_frame_change = min(len(before_frames), len(after_frames))
    else:
        pass
    first_main_change = next(
        (
            index
            for index, (left, right) in enumerate(zip(before_frames, after_frames))
            if int(left[0]) != int(right[0])
        ),
        None,
    )
    prefix_equal = all(
        equal
        for name, equal in opening_equal.items()
        if not held
        or name not in ("first-decode-prefix-cache-000.pt", "backbone-decode-001.pt")
    )
    if held:
        before_held = torch.load(
            baseline / "held-first-decode-000.pt", weights_only=True
        )
        after_held = torch.load(
            candidate / "held-first-decode-000.pt", weights_only=True
        )
        prefix_equal = prefix_equal and torch.equal(
            before_held["embedding"], after_held["embedding"]
        )
    else:
        pass
    original_norm_exact = torch.equal(
        before_layers[0]["attention_normalized"],
        after_layers[0]["attention_normalized_before_override"],
    )
    combined_input_exact = torch.equal(
        before_layers[0]["input_contribution"], after_layers[0]["input_contribution"]
    )
    cast_changed = not torch.equal(
        after_layers[0]["attention_normalized"],
        after_layers[0]["attention_normalized_before_override"],
    )
    return {
        "opening_state_equal": prefix_equal
        and original_norm_exact
        and combined_input_exact,
        "future_text_equal": before_complete
        and after_complete
        and len(before_text) == len(after_text)
        and all(
            torch.equal(left, right) for left, right in zip(before_text, after_text)
        ),
        "first_changed_codec_frame": first_frame_change,
        "first_changed_main_codec_frame": first_main_change,
        "before_second_code": int(before_frames[1][0]),
        "after_second_code": int(after_frames[1][0]),
        "before_frames": len(before_frames),
        "after_frames": len(after_frames),
        "cast_changed": cast_changed,
    }


def audit_first_decode_attention(
    trace_directory: Path,
) -> dict[str, bool | int | float]:
    thinker_publish = torch.load(
        trace_directory / "thinker-publish-000.pt", weights_only=True
    )
    talker_prompt = torch.load(
        trace_directory / "talker-prompt-000.pt", weights_only=True
    )
    addresses = torch.load(
        trace_directory / "first-decode-addresses-000.pt", weights_only=True
    )
    prefix = torch.load(
        trace_directory / "first-decode-prefix-cache-000.pt", weights_only=True
    )
    operands = torch.load(trace_directory / "backbone-decode-001.pt", weights_only=True)
    first_layer = torch.load(
        trace_directory / "first-decode-layers.pt", weights_only=True
    )[0]
    cache_length = int(first_layer["attention_cache_lengths"])
    page_size = int(first_layer["attention_page_sizes"])
    logical_positions = torch.arange(cache_length)
    attention_slots = (
        first_layer["attention_page_table"][logical_positions // page_size] * page_size
        + logical_positions % page_size
    )
    expected_slots = torch.cat(
        (
            addresses["scheduler_read_slots"],
            addresses["scheduler_mapped_write_slot"].view(1),
        )
    )
    prefix_keys = prefix["key_rows"][0]
    prefix_values = prefix["value_rows"][0]
    key_values = torch.cat(
        (prefix_keys, first_layer["key_rope"].reshape_as(prefix_keys[:1]))
    ).double()
    value_values = torch.cat(
        (prefix_values, first_layer["value"].reshape_as(prefix_values[:1]))
    ).double()
    head_dimension = prefix_keys.shape[-1]
    query = first_layer["query_rope"].reshape(-1, head_dimension).double()
    query_heads_per_key_head = query.shape[0] // prefix_keys.shape[1]
    key_values = key_values.repeat_interleave(query_heads_per_key_head, dim=1)
    value_values = value_values.repeat_interleave(query_heads_per_key_head, dim=1)
    attention_weights = (
        torch.einsum("hd,shd->hs", query, key_values)
        .mul(head_dimension**-0.5)
        .softmax(dim=-1)
    )
    expected_context = torch.einsum("hs,shd->hd", attention_weights, value_values)
    permuted_context = torch.einsum(
        "hs,shd->hd", attention_weights, value_values.roll(shifts=1, dims=0)
    )
    actual_context = first_layer["attention_context"].reshape_as(expected_context)
    combined_input = operands["feedback_row"].reshape(-1) + operands[
        "text_row"
    ].reshape(-1)
    return {
        "cache_length": cache_length,
        "page_size": page_size,
        "native_decode_batch_size": addresses.get("native_decode_batch_size", 1),
        "native_decode_row_index": addresses.get("native_decode_row_index", 0),
        "captured_request_matches_scheduler": torch.equal(
            first_layer["attention_request_indices"], addresses["request_pool_index"]
        ),
        "valid_attention_slots_match_scheduler": torch.equal(
            attention_slots, expected_slots
        ),
        "write_slot_matches_scheduler_mapping": torch.equal(
            first_layer["attention_write_slots"],
            addresses["scheduler_write_slot"].to(first_layer["attention_write_slots"]),
        )
        and int(addresses["scheduler_write_slot"])
        == int(addresses["scheduler_mapped_write_slot"]),
        "cache_length_matches_position": cache_length
        == int(addresses["scheduler_cache_length"])
        == int(operands["position"]) + 1,
        "layer_input_matches_projected_text_and_feedback": torch.equal(
            combined_input, first_layer["input_contribution"]
        ),
        "thinker_handoff_matches_talker": (
            thinker_publish["media_positions"].tolist()
            == talker_prompt["media_positions"]
        )
        and torch.equal(
            thinker_publish["media_hidden_states"], talker_prompt["media_hidden_states"]
        ),
        "attention_context_exact_after_rounding": torch.equal(
            actual_context, expected_context.to(actual_context)
        ),
        "attention_context_max_abs_error": float(
            (actual_context - expected_context).abs().max()
        ),
        "attention_context_relative_l2_error": float(
            (actual_context - expected_context).norm() / expected_context.norm()
        ),
        "permuted_values_max_abs_error": float(
            (actual_context - permuted_context).abs().max()
        ),
    }


def compare_native_prefill(
    breakable_trace: Path, eager_trace: Path
) -> dict[str, bool | int | float | list[int] | None]:
    comparison: dict[str, bool | int | float | list[int] | None] = {}
    for boundary in (
        "thinker-publish-000.pt",
        "talker-prompt-000.pt",
        "backbone-prefill-000.pt",
        "prefill-sampling-000.pt",
        "codec-000.pt",
        "backbone-decode-001.pt",
        "first-decode-prefix-cache-000.pt",
    ):
        breakable = torch.load(breakable_trace / boundary, weights_only=True)
        eager = torch.load(eager_trace / boundary, weights_only=True)
        assert breakable.keys() == eager.keys()
        for name, value in breakable.items():
            if name == "request_pool_index":
                continue
            else:
                pass
            equivalent = (
                value.dtype == eager[name].dtype and torch.equal(value, eager[name])
                if isinstance(value, torch.Tensor)
                else value == eager[name]
            )
            comparison[f"{boundary.removesuffix('.pt')}/{name}"] = equivalent
    for trace, name in ((breakable_trace, "breakable"), (eager_trace, "eager")):
        addresses = torch.load(
            trace / "first-decode-addresses-000.pt", weights_only=True
        )
        comparison[f"{name}_prefill_graph_replays"] = addresses.get(
            "prefill_graph_replays"
        )
        comparison[f"{name}_prefill_eager_forwards"] = addresses.get(
            "prefill_eager_forwards"
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asr-path", required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--h200-history", type=Path, required=True)
    parser.add_argument("--h100-history", type=Path, required=True)
    parser.add_argument("--held-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transfer-from", type=Path)
    parser.add_argument(
        "--transfer-state", choices=("codec", "kv", "both"), default="codec"
    )
    parser.add_argument("--exclude-frozen-transfer-control", action="store_true")
    parser.add_argument("--native-kv-audit", action="store_true")
    parser.add_argument("--eager-talker-prefill-reference", type=Path)
    parser.add_argument("--batched-native-reference", type=Path)
    parser.add_argument("--exact-talker-prefill-reference", type=Path)
    parser.add_argument("--prefill-token-probe", action="store_true")
    parser.add_argument("--fixed-shared-gate", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--score-existing", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    reuse_capture = arguments.score_existing or arguments.audit_existing
    assert not arguments.audit_existing or arguments.native_kv_audit
    if reuse_capture:
        assert (output / "manifest.json").is_file()
        assert not (output / "generation.jsonl").exists()
        assert not (output / "scored.jsonl").exists()
    else:
        output.mkdir()
    cases = select_cases(arguments.h200_history, arguments.h100_history)
    donor_population = arguments.transfer_from
    prefill_reference = arguments.eager_talker_prefill_reference
    batched_reference = arguments.batched_native_reference
    exact_prefill_reference = arguments.exact_talker_prefill_reference
    assert (
        sum(
            reference is not None
            for reference in (
                prefill_reference,
                batched_reference,
                exact_prefill_reference,
            )
        )
        <= 1
    )
    native_reference = prefill_reference or batched_reference or exact_prefill_reference
    assert native_reference is None or arguments.native_kv_audit
    donor_rows = {}
    if arguments.native_kv_audit:
        assert donor_population is None and not arguments.score_existing
        scopes = ("plain", "observe") if native_reference is None else ("observe",)
        cases = [case for case in cases if case["sample_id"] != HELD_SAMPLE_ID]
        if batched_reference is not None:
            cases = [
                case
                for case in cases
                if case["cohort"] == "historic_severe"
                or case["sample_id"]
                == "common_voice_en_19401124-common_voice_en_19401123"
            ]
        else:
            pass
        if exact_prefill_reference is not None:
            cases = [
                case
                for case in cases
                if case["sample_id"] in PREFILL_PROBE_TOKEN_BY_SAMPLE
            ]
        else:
            pass
        if arguments.prefill_token_probe:
            assert native_reference is None
            scopes = ("observe", "eager")
            cases = [
                case
                for case in cases
                if case["sample_id"] in PREFILL_PROBE_TOKEN_BY_SAMPLE
            ]
        else:
            pass
    elif donor_population is None:
        scopes = ("none", "first", "decode")
    else:
        scopes = ("transfer", "restore")
        eligible = []
        for line in (donor_population / "scored.jsonl").read_text().splitlines():
            row = json.loads(line)
            before, after, pair = (
                row["arms"]["none"],
                row["arms"]["first"],
                row["pairs"]["first"],
            )
            frame_index = pair.get("first_changed_codec_frame")
            if (
                pair["opening_state_equal"]
                and not (
                    arguments.exclude_frozen_transfer_control
                    and row["used_frozen_h200_state"]
                )
                and pair["future_text_equal"]
                and "spoken_wer" in before
                and "spoken_wer" in after
                and before["returned_text"] == after["returned_text"]
                and frame_index is not None
                and (arguments.transfer_state == "codec" or frame_index == 1)
                and 0
                < frame_index
                < min(pair["before_frames"], pair["after_frames"]) - 1
            ):
                eligible.append(
                    (after["word_errors"] - before["word_errors"], row["sample_id"])
                )
                donor_rows[row["sample_id"]] = row
        worsening = sorted((entry for entry in eligible if entry[0] > 0), reverse=True)[
            :2
        ]
        improving = sorted(entry for entry in eligible if entry[0] < 0)[:2]
        selected = {sample_id for _, sample_id in (*worsening, *improving)}
        cases = [case for case in cases if case["sample_id"] in selected]
    if reuse_capture:
        assert json.loads((output / "manifest.json").read_bytes()) == cases
    else:
        (output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
        samples = load_seedtts_samples(DATASETS["seedtts"])
        assert len(samples) == EXPECTED_SEEDTTS_SAMPLES
        samples_by_id = {sample.sample_id: sample for sample in samples}
        witness_ids = {case["sample_id"] for case in cases}
        peer_samples = [
            sample for sample in samples if sample.sample_id not in witness_ids
        ][: NATIVE_AUDIT_CLIENT_CONCURRENCY - 1]
    control_path = output / "control.json"
    server_config = arguments.server_config
    if exact_prefill_reference is not None and not reuse_capture:
        selected_samples = {case["sample_id"] for case in cases}
        exact_buckets = sorted(
            row["pairs"]["observe"]["cache_length"] - 1
            for row in map(
                json.loads,
                (exact_prefill_reference / "generation.jsonl").read_text().splitlines(),
            )
            if row["sample_id"] in selected_samples
        )
        assert len(exact_buckets) == len(selected_samples)
        resolved_config = yaml.safe_load(server_config.read_text())
        talker_engine = resolved_config["stages"]["talker_ar"].setdefault("engine", {})
        talker_engine["cuda_graph_bs_prefill"] = exact_buckets
        server_config = output / "server-config.yaml"
        server_config.write_text(yaml.safe_dump(resolved_config))
    else:
        pass
    for scope in () if reuse_capture else scopes:
        control_path.write_text('{"request_id": ""}\n')
        talker_port = find_available_port()
        server_environment: dict[str, str] = {}
        if scope != "plain":
            server_environment["OMNI_WITNESS_BATCH_CONTROL"] = str(control_path)
            if arguments.native_kv_audit:
                server_environment["OMNI_WITNESS_NATIVE_KV_AUDIT"] = "1"
            else:
                pass
            if arguments.prefill_token_probe:
                server_environment["OMNI_WITNESS_TRACE_PREFILL_TOKEN"] = "1"
            else:
                pass
        else:
            pass
        if arguments.fixed_shared_gate:
            server_environment["OMNI_WITNESS_FIXED_SHARED_GATE"] = "1"
        else:
            pass
        talker_server = start_server_from_cmd(
            [
                sys.executable,
                "-P",
                "-m",
                "sglang_omni.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(talker_port),
                "--config",
                str(server_config),
                "--model-path",
                arguments.model_path,
                "--model-name",
                "qwen3-omni",
                *(
                    ["--talker_ar.engine.cuda_graph_backend_prefill", "disabled"]
                    if prefill_reference is not None or scope == "eager"
                    else []
                ),
            ],
            output / f"talker-{scope}.log",
            talker_port,
            timeout=SERVER_START_TIMEOUT_SECONDS,
            env=server_environment,
            strip_proxy=True,
        )
        try:
            endpoint = f"http://127.0.0.1:{talker_port}/v1/chat/completions"
            with httpx.Client(
                trust_env=False, timeout=REQUEST_TIMEOUT_SECONDS
            ) as client:
                for index, case in enumerate(cases, 1):
                    sample = samples_by_id[case["sample_id"]]
                    input_sha256 = hashlib.sha256(
                        Path(sample.ref_audio).read_bytes()
                    ).hexdigest()
                    is_held = case["sample_id"] == HELD_SAMPLE_ID
                    if is_held:
                        assert input_sha256 == HELD_AUDIO_SHA256
                    else:
                        pass
                    case_directory = output / f'{case["cohort"]}-{case["sample_id"]}'
                    arm_directory = case_directory / scope
                    arm_directory.mkdir(parents=True)
                    control: WitnessControl = {
                        "request_id": case["request_id"],
                        "output_directory": str(arm_directory),
                        "norm_scope": {
                            "transfer": "none",
                            "restore": "first",
                            "plain": "none",
                            "observe": "none",
                            "eager": "none",
                        }.get(scope, scope),
                        "held_input_directory": (
                            str(arguments.held_input.resolve()) if is_held else None
                        ),
                    }
                    if arguments.prefill_token_probe:
                        control["prefill_probe_token_index"] = (
                            PREFILL_PROBE_TOKEN_BY_SAMPLE[case["sample_id"]]
                        )
                    else:
                        pass
                    if donor_population is not None:
                        donor_row = donor_rows[case["sample_id"]]
                        donor_scope = "first" if scope == "transfer" else "none"
                        frame_index = donor_row["pairs"]["first"][
                            "first_changed_codec_frame"
                        ]
                        control["codec_override_frame"] = frame_index
                        donor_directory = (
                            donor_population / donor_row["directory"] / donor_scope
                        )
                        if arguments.transfer_state != "kv":
                            control["codec_override_file"] = str(
                                donor_directory / f"codec-{frame_index:03d}.pt"
                            )
                        else:
                            pass
                        if arguments.transfer_state != "codec":
                            assert frame_index == 1
                            control["kv_override_file"] = str(
                                donor_directory / "first-decode-layers.pt"
                            )
                        else:
                            pass
                    else:
                        pass
                    next_control = control_path.with_suffix(".next")
                    next_control.write_text(json.dumps(control))
                    next_control.replace(control_path)
                    if batched_reference is None:
                        response = post_seedtts_sample(
                            client, endpoint, sample, case["request_id"]
                        )
                    else:
                        start_barrier = Barrier(NATIVE_AUDIT_CLIENT_CONCURRENCY)
                        assert len(peer_samples) == NATIVE_AUDIT_CLIENT_CONCURRENCY - 1
                        with ThreadPoolExecutor(
                            NATIVE_AUDIT_CLIENT_CONCURRENCY
                        ) as workers:
                            peer_responses = [
                                workers.submit(
                                    post_seedtts_sample,
                                    client,
                                    endpoint,
                                    peer_sample,
                                    f"audit-peer-{index}-{peer_index}",
                                    start_barrier,
                                )
                                for peer_index, peer_sample in enumerate(peer_samples)
                            ]
                            response = post_seedtts_sample(
                                client,
                                endpoint,
                                sample,
                                case["request_id"],
                                start_barrier,
                            )
                            for peer_response in peer_responses:
                                peer_response.result().raise_for_status()
                    arm = {
                        "http_status": response.status_code,
                        "input_sha256": input_sha256,
                    }
                    if response.is_success:
                        completion = response.json()
                        assert completion["id"] == f'chatcmpl-{case["request_id"]}'
                        (choice,) = completion["choices"]
                        waveform = base64.b64decode(
                            choice["message"]["audio"]["data"], validate=True
                        )
                        if not arguments.native_kv_audit:
                            (arm_directory / "audio.wav").write_bytes(waveform)
                        else:
                            pass
                        arm.update(
                            returned_text=choice["message"]["content"],
                            finish_reason=choice["finish_reason"],
                            waveform_sha256=hashlib.sha256(waveform).hexdigest(),
                            duration_seconds=get_wav_duration(waveform),
                        )
                    else:
                        arm["error"] = response.text
                    (arm_directory / "response.json").write_text(json.dumps(arm))
                    print(
                        f'{scope} {index}/{len(cases)}: {case["sample_id"]} HTTP {response.status_code}',
                        flush=True,
                    )
        finally:
            stop_server(talker_server)

    results = []
    for case in cases:
        case_directory = output / f'{case["cohort"]}-{case["sample_id"]}'
        is_held = case["sample_id"] == HELD_SAMPLE_ID
        arms = {
            scope: json.loads((case_directory / scope / "response.json").read_bytes())
            for scope in scopes
        }
        assert len({arm["input_sha256"] for arm in arms.values()}) == 1
        if arguments.prefill_token_probe:
            assert all(arm["http_status"] == 200 for arm in arms.values())
            graph_trace = case_directory / "observe"
            eager_trace = case_directory / "eager"
            graph_layers = torch.load(
                graph_trace / "prefill-layers.pt", weights_only=True
            )
            eager_layers = torch.load(
                eager_trace / "prefill-layers.pt", weights_only=True
            )
            differing_layers = {}
            for layer_index, (graph_layer, eager_layer) in enumerate(
                zip(graph_layers, eager_layers, strict=True)
            ):
                assert graph_layer.keys() == eager_layer.keys()
                assert torch.equal(graph_layer["positions"], eager_layer["positions"])
                changed_operands = [
                    field
                    for field, operand in graph_layer.items()
                    if not torch.equal(operand, eager_layer[field])
                ]
                if changed_operands:
                    differing_layers[layer_index] = changed_operands
                else:
                    pass
            pairs = {
                "prefill_probe": {
                    **compare_native_prefill(graph_trace, eager_trace),
                    "logical_token_index": PREFILL_PROBE_TOKEN_BY_SAMPLE[
                        case["sample_id"]
                    ],
                    "different_operands_by_layer": differing_layers,
                },
            }
        elif arguments.native_kv_audit:
            assert all(arm["http_status"] == 200 for arm in arms.values())
            if native_reference is None:
                reference_arm = arms["plain"]
                prefill_comparison = {}
            else:
                reference_trace = native_reference / case_directory.name / "observe"
                reference_arm = json.loads(
                    (reference_trace / "response.json").read_bytes()
                )
                assert reference_arm["input_sha256"] == arms["observe"]["input_sha256"]
                prefill_comparison = compare_native_prefill(
                    reference_trace, case_directory / "observe"
                )
            pairs = {
                "observe": {
                    **audit_first_decode_attention(case_directory / "observe"),
                    **prefill_comparison,
                    "waveform_matches_reference": (
                        reference_arm["waveform_sha256"]
                        == arms["observe"]["waveform_sha256"]
                    ),
                    "returned_text_matches_reference": (
                        reference_arm["returned_text"]
                        == arms["observe"]["returned_text"]
                    ),
                },
            }
        elif donor_population is None:
            pairs = {
                scope: compare_pair(
                    case_directory / "none", case_directory / scope, is_held
                )
                for scope in ("first", "decode")
            }
        else:
            pairs = {}
            donor_row = donor_rows[case["sample_id"]]
            frame_index = donor_row["pairs"]["first"]["first_changed_codec_frame"]
            for scope, recipient_scope, donor_scope in (
                ("transfer", "none", "first"),
                ("restore", "first", "none"),
            ):
                reference_directory = donor_population / donor_row["directory"]
                recipient_frames, recipient_text, recipient_complete = (
                    controlled_history(reference_directory / recipient_scope)
                )
                donor_frames, _, _ = controlled_history(
                    reference_directory / donor_scope
                )
                replay_frames, replay_text, replay_complete = controlled_history(
                    case_directory / scope
                )
                if arguments.transfer_state != "kv":
                    assert (
                        case_directory / scope / f"codec-override-{frame_index:03d}.pt"
                    ).is_file()
                    expected_frame = donor_frames[frame_index]
                else:
                    expected_frame = recipient_frames[frame_index]
                if arguments.transfer_state != "codec":
                    assert (
                        case_directory / scope / f"kv-override-{frame_index:03d}.pt"
                    ).is_file()
                else:
                    pass
                assert torch.equal(replay_frames[frame_index], expected_frame)
                sampling_name = f"decode-sampling-{frame_index:03d}.pt"
                recipient_sampling = torch.load(
                    reference_directory / recipient_scope / sampling_name,
                    weights_only=True,
                )
                replay_sampling = torch.load(
                    case_directory / scope / sampling_name, weights_only=True
                )
                assert recipient_sampling.keys() == replay_sampling.keys()
                pairs[scope] = {
                    "overridden_frame": frame_index,
                    "transferred_state": arguments.transfer_state,
                    "complete_donor_codec_history_equal": len(replay_frames)
                    == len(donor_frames)
                    and all(
                        torch.equal(left, right)
                        for left, right in zip(replay_frames, donor_frames)
                    ),
                    "waveform_matches_donor": arms[scope].get("waveform_sha256")
                    == donor_row["arms"][donor_scope]["waveform_sha256"],
                    "preintervention_sampling_exact": all(
                        torch.equal(value, replay_sampling[name])
                        for name, value in recipient_sampling.items()
                    ),
                    "preceding_codec_history_exact": all(
                        torch.equal(left, right)
                        for left, right in zip(
                            recipient_frames[:frame_index],
                            replay_frames[:frame_index],
                            strict=True,
                        )
                    ),
                    "future_text_equal": recipient_complete
                    and replay_complete
                    and len(recipient_text) == len(replay_text)
                    and all(
                        torch.equal(left, right)
                        for left, right in zip(recipient_text, replay_text)
                    ),
                    "returned_text_equal": arms[scope].get("returned_text")
                    == donor_row["arms"][recipient_scope]["returned_text"],
                    "recipient_wer": donor_row["arms"][recipient_scope]["spoken_wer"],
                    "donor_wer": donor_row["arms"][donor_scope]["spoken_wer"],
                }
        result = {
            **case,
            "used_frozen_h200_state": is_held,
            "directory": case_directory.name,
            "arms": arms,
            "pairs": pairs,
        }
        results.append(result)
        with (output / "generation.jsonl").open("a") as generated:
            generated.write(json.dumps(result) + "\n")

    if arguments.native_kv_audit:
        return
    else:
        pass
    asr_port = find_available_port()
    asr_server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(asr_port),
            "--model-path",
            arguments.asr_path,
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        output / "asr.log",
        asr_port,
        timeout=SERVER_START_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        asr = load_qwen3_asr(asr_port)
        transcripts = {}
        for index, result in enumerate(results, 1):
            for scope, arm in result["arms"].items():
                if "waveform_sha256" not in arm:
                    continue
                else:
                    pass
                path = output / result["directory"] / scope / "audio.wav"
                digest = arm["waveform_sha256"]
                if digest not in transcripts:
                    transcripts[digest] = transcribe(asr, str(path), "en", "cuda:0")
                else:
                    pass
                spoken = apply_wer(
                    SampleOutput(
                        sample_id=result["sample_id"], target_text=arm["returned_text"]
                    ),
                    transcripts[digest],
                    "en",
                )
                assert spoken.is_success
                arm.update(
                    transcript=spoken.whisper_text,
                    spoken_wer=spoken.wer,
                    word_errors=spoken.substitutions
                    + spoken.deletions
                    + spoken.insertions,
                    reference_words=spoken.substitutions
                    + spoken.deletions
                    + spoken.hits,
                )
            with (output / "scored.jsonl").open("a") as scored:
                scored.write(json.dumps(result) + "\n")
            print(f'{index}/{len(results)} scored: {result["sample_id"]}', flush=True)
    finally:
        stop_server(asr_server)


if __name__ == "__main__":
    main()
else:
    pass
