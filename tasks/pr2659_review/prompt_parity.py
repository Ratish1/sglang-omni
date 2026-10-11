from __future__ import annotations

import ast
import json
import subprocess
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_tts import sglang_model
from sglang_omni.models.qwen3_tts.prompt_frontend import Qwen3TTSPromptFrontend
from sglang_omni.preprocessing.cache_key import PromptRowCacheKeys

BASE = "284381f3a6f6207eeb1f89310173caf410f123df"


def run() -> None:
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:sglang_omni/models/qwen3_tts/sglang_model.py"],
        text=True,
    )
    original_class = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3TTSPromptBuilderMixin"
    )
    original_class.name = "ReferencePromptBuilder"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            original_class,
        ],
        type_ignores=[],
    )
    namespace = vars(sglang_model).copy()
    exec(
        compile(ast.fix_missing_locations(module), "pre-pr-prompt-builder", "exec"),
        namespace,
    )

    class ReferenceFrontend(
        namespace["ReferencePromptBuilder"], Qwen3TTSPromptFrontend
    ):
        pass

    observations: list[dict[str, str | int | bool]] = []
    for device in (torch.device("cpu"), torch.device("cuda")):
        for dtype in (torch.float32, torch.bfloat16):
            for hidden_size, text_hidden_size in ((16, 12), (1024, 2048)):
                torch.manual_seed(311)
                talker = SimpleNamespace(
                    vocab_size=64,
                    hidden_size=hidden_size,
                    text_vocab_size=64,
                    text_hidden_size=text_hidden_size,
                    num_code_groups=3,
                    code_predictor_config=SimpleNamespace(vocab_size=32),
                    spk_id={"vivian": 3, "eric": 4},
                    spk_is_dialect={"eric": "sichuan"},
                    codec_language_id={"en": 1, "zh": 2, "sichuan": 5},
                    codec_pad_id=0,
                    codec_bos_id=1,
                    codec_nothink_id=2,
                    codec_think_id=3,
                    codec_think_bos_id=4,
                    codec_think_eos_id=5,
                )
                root = SimpleNamespace(
                    talker_config=talker,
                    tts_model_type="custom_voice",
                    tts_bos_token_id=0,
                    tts_eos_token_id=1,
                    tts_pad_token_id=2,
                )
                current = Qwen3TTSPromptFrontend(root, device=device, dtype=dtype)
                reference = ReferenceFrontend(root, device=device, dtype=dtype)
                reference.load_state_dict(current.state_dict())
                input_id = torch.arange(20, device=device).unsqueeze(0) % 64
                ref_id = torch.arange(11, device=device).unsqueeze(0) % 64
                instruct_id = torch.arange(7, device=device).unsqueeze(0)
                with torch.no_grad():
                    constants = current.build_prompt_constants()
                    before = {
                        name: tensor.clone()
                        for name, tensor in (
                            ("bos", constants.tts_bos_embed),
                            ("pad", constants.tts_pad_embed),
                            ("feedback_pad", constants.feedback_pad_embed),
                        )
                    }
                    keys = PromptRowCacheKeys.for_rows(
                        row_bytes=hidden_size * constants.tts_pad_embed.element_size(),
                        device=device,
                    )
                    for non_streaming_mode in (False, True):
                        for language in ("auto", "en", "zh"):
                            for variant in (
                                "custom_voice",
                                "voice_design",
                                "xvector",
                                "icl_short",
                                "icl_long",
                            ):
                                arguments = dict(
                                    input_id=input_id,
                                    language=language,
                                    non_streaming_mode=non_streaming_mode,
                                    instruct_id=instruct_id,
                                )
                                if variant == "custom_voice":
                                    arguments["voice"] = "eric"
                                    current_builder = current.build_custom_voice_inputs
                                    reference_builder = (
                                        reference.build_custom_voice_inputs
                                    )
                                elif variant == "voice_design":
                                    current_builder = current.build_voice_design_inputs
                                    reference_builder = (
                                        reference.build_voice_design_inputs
                                    )
                                else:
                                    code_length = 4 if variant == "icl_short" else 40
                                    voice_prompt = {
                                        "ref_spk_embedding": [
                                            torch.randn(
                                                hidden_size, device=device, dtype=dtype
                                            )
                                        ],
                                        "icl_mode": [variant != "xvector"],
                                        "ref_code": [
                                            (
                                                None
                                                if variant == "xvector"
                                                else torch.randint(
                                                    0,
                                                    32,
                                                    (code_length, 3),
                                                    device=device,
                                                )
                                            )
                                        ],
                                    }
                                    arguments.update(
                                        ref_id=ref_id, voice_clone_prompt=voice_prompt
                                    )
                                    current_builder = current.build_voice_clone_inputs
                                    reference_builder = (
                                        reference.build_voice_clone_inputs
                                    )
                                expected = reference_builder(**arguments)
                                actual = current_builder(
                                    **arguments, prompt_constants=constants
                                )
                                for index, (observed, wanted) in enumerate(
                                    zip(actual, expected, strict=True)
                                ):
                                    if wanted is None:
                                        assert observed is None
                                    else:
                                        assert torch.equal(observed, wanted), (
                                            device,
                                            dtype,
                                            hidden_size,
                                            variant,
                                            language,
                                            non_streaming_mode,
                                            index,
                                            (observed - wanted).abs().max(),
                                        )
                                assert keys.key_ids(
                                    actual[0].squeeze(0)
                                ) == keys.key_ids(expected[0].squeeze(0))
                                observations.append(
                                    dict(
                                        device=str(device),
                                        dtype=str(dtype),
                                        hidden_size=hidden_size,
                                        variant=variant,
                                        language=language,
                                        non_streaming_mode=non_streaming_mode,
                                        exact=True,
                                    )
                                )
                    for name, tensor in (
                        ("bos", constants.tts_bos_embed),
                        ("pad", constants.tts_pad_embed),
                        ("feedback_pad", constants.feedback_pad_embed),
                    ):
                        assert torch.equal(before[name], tensor)
                    expected_feedback_pad = (
                        reference.text_projection(
                            reference.get_text_embeddings()(
                                torch.tensor([[2]], device=device)
                            )
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    assert torch.equal(
                        constants.feedback_pad_embed, expected_feedback_pad
                    )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {"comparisons": len(observations), "results": observations}, indent=2
        )
    )


if __name__ == "__main__":
    run()
