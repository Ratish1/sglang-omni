"""Exercise actual FireRed admission, Client lowering and prompt construction."""

import ast
import base64
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang_omni.client.client import Client
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.preprocessing import (
    preprocess_firered_tts3_payload,
)
from sglang_omni.models.firered_tts3.speech_validation import (
    FireRedTTS3InstructSpeechRequestValidator,
    FireRedTTS3SpeakerSpeechRequestValidator,
    FireRedTTS3SpeechRequestValidator,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.serve.speech_errors import SpeechAPIError, speech_generation_error

fixture_path = Path("tests/unit_test/fixtures/firered_tts3.py")
tokenizer_definition = next(
    node
    for node in ast.parse(fixture_path.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "make_speaker_tokenizer"
)
namespace = {
    "Regex": Regex,
    "Tokenizer": Tokenizer,
    "decoders": decoders,
    "models": models,
    "pre_tokenizers": pre_tokenizers,
    "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[tokenizer_definition], type_ignores=[])
        ),
        str(fixture_path),
        "exec",
    ),
    namespace,
)
tokenizer = namespace["make_speaker_tokenizer"]()
reference = "data:audio/wav;base64," + base64.b64encode(
    Path("docs/_static/audio/ref_voice.wav").read_bytes()
).decode("ascii")
instruct = FireRedTTS3InstructSpeechRequestValidator(
    default_model="tts",
    max_speech_input_chars=FireRedTTS3PipelineConfig.max_speech_input_chars,
)
speaker = FireRedTTS3SpeakerSpeechRequestValidator(
    default_model="tts",
    max_speech_input_chars=FireRedTTS3PipelineConfig.max_speech_input_chars,
)
base = FireRedTTS3SpeechRequestValidator(
    default_model="tts",
    max_speech_input_chars=FireRedTTS3PipelineConfig.max_speech_input_chars,
    required_speech_reference_count=1,
    speech_reference_text_required=True,
)
instruct_config = FireRedTTS3Config(
    **json.loads((Path(__file__).parent / "instruct_config.json").read_text())
)
speaker_config = FireRedTTS3Config(
    text_latent_interleave_ratio=[2, 1], use_lm_head=True
)
base_config = FireRedTTS3Config(
    **json.loads((Path(__file__).parent / "base_config.json").read_text())
)
cases = [
    (
        "base_cloning",
        base,
        base_config,
        {
            "input": "Hello",
            "ref_audio": reference,
            "ref_text": "reference",
            "language": "English",
        },
    ),
    (
        "instruct_cloning",
        instruct,
        instruct_config,
        {"input": "Hello", "ref_audio": reference, "ref_text": "reference"},
    ),
    (
        "voice_design",
        instruct,
        instruct_config,
        {"input": "Hello", "task_type": "VoiceDesign", "instructions": "A calm voice"},
    ),
    (
        "semantic_edit",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "semantic_edit",
            "instructions": "Replace 'a' with 'b'.",
            "ref_audio": reference,
        },
    ),
    (
        "acoustic_edit",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "AcousticEdit",
            "speed": 0.8,
            "ref_audio": reference,
        },
    ),
    (
        "speaker_dual_stream",
        speaker,
        speaker_config,
        {
            "input": "Hello",
            "stage_params": {
                "preprocessing": {"dual_stream": True, "style": "calm"},
                "talker": {"n_timesteps": "3", "min_patches": 0},
            },
            "seed": 7,
            "temperature": 0.5,
            "extra_params": {
                "temperature": "invalid",
                "stage_params": {"talker": {"n_timesteps": -1}},
            },
        },
    ),
    (
        "unsupported_customvoice",
        instruct,
        instruct_config,
        {"input": "Hello", "task_type": "CustomVoice"},
    ),
    (
        "missing_design_instruction",
        instruct,
        instruct_config,
        {"input": "Hello", "task_type": "VoiceDesign"},
    ),
    (
        "missing_edit_source",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "SemanticEdit",
            "instructions": "Replace 'a' with 'b'.",
        },
    ),
    (
        "long_design_text",
        instruct,
        instruct_config,
        {
            "input": "a" * 401,
            "task_type": "VoiceDesign",
            "instructions": "A calm voice",
        },
    ),
    (
        "invalid_acoustic_value",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "AcousticEdit",
            "ref_audio": reference,
            "stage_params": {
                "preprocessing": {
                    "acoustic_attribute": "Speed",
                    "acoustic_value": "not-a-number",
                }
            },
        },
    ),
    (
        "invalid_acoustic_attribute",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "AcousticEdit",
            "ref_audio": reference,
            "stage_params": {
                "preprocessing": {"acoustic_attribute": "Tempo", "acoustic_value": 0.8}
            },
        },
    ),
    (
        "infinite_acoustic_value",
        instruct,
        instruct_config,
        {
            "input": "",
            "task_type": "AcousticEdit",
            "ref_audio": reference,
            "stage_params": {
                "preprocessing": {
                    "acoustic_attribute": "Speed",
                    "acoustic_value": "inf",
                }
            },
        },
    ),
]
cases.extend(
    [
        (
            "exact_text_boundary",
            instruct,
            instruct_config,
            {"input": "a" * 400, "task_type": "VoiceDesign", "instructions": "calm"},
        ),
        (
            "base_auto",
            base,
            base_config,
            {
                "input": "Hello",
                "language": "Auto",
                "ref_audio": reference,
                "ref_text": "reference",
            },
        ),
        (
            "instruct_auto",
            instruct,
            instruct_config,
            {
                "input": "Hello",
                "language": "Auto",
                "task_type": "VoiceDesign",
                "instructions": "calm",
            },
        ),
        (
            "speaker_auto",
            speaker,
            speaker_config,
            {"input": "Hello", "language": "Auto"},
        ),
        (
            "explicit_instruction_precedence",
            instruct,
            instruct_config,
            {
                "input": "",
                "task_type": "AcousticEdit",
                "instructions": "adjust the volume to 0.5",
                "ref_audio": reference,
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": "unused",
                        "acoustic_value": "nan",
                    }
                },
            },
        ),
        (
            "normalized_pitch",
            instruct,
            instruct_config,
            {
                "input": "",
                "task_type": "AcousticEdit",
                "ref_audio": reference,
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": " pItCh ",
                        "acoustic_value": "-1",
                    }
                },
            },
        ),
        (
            "null_composition_fallback",
            instruct,
            instruct_config,
            {
                "input": "",
                "task_type": "AcousticEdit",
                "ref_audio": reference,
                "speed": 0.8,
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": None,
                        "acoustic_value": None,
                    }
                },
            },
        ),
    ]
)
observations = []
for case_name, validator, config, body in cases:
    observation = {"case": case_name}
    try:
        prepared = validator.parse_generation_request(body)
    except SpeechAPIError as error:
        observation.update(admitted=False, status=error.status_code, param=error.param)
    else:
        observation["admitted"] = True
        generated = validator.build_generate_request(
            prepared.request,
            validate=False,
            reference_descriptors=prepared.reference_descriptors,
        )
        request = Client.build_omni_request(generated)
        try:
            payload = preprocess_firered_tts3_payload(
                StagePayload(request_id=case_name, request=request, data={}),
                tokenizer=tokenizer,
                config=config,
                n_timesteps=10,
                max_patches=400,
            )
        except (ValueError, TypeError) as error:
            observation.update(
                stage_error=str(error),
                mapped_status=speech_generation_error(error).status_code,
            )
        else:
            state = FireRedTTS3State.from_dict(payload.data)
            observation.update(
                task=state.task_type,
                dual_stream=state.dual_stream,
                n_timesteps=state.n_timesteps,
                min_patches=state.min_patches,
                seed=state.seed,
                prompt=tokenizer.decode(state.input_ids),
                temperature=request.params.get("temperature"),
            )
    observations.append(observation)
print(
    json.dumps(
        {
            "source": __import__(
                "sglang_omni.models.firered_tts3.speech_validation",
                fromlist=["__file__"],
            ).__file__,
            "cases": observations,
        },
        indent=2,
    )
)

negative = {
    "unsupported_customvoice",
    "missing_design_instruction",
    "missing_edit_source",
    "long_design_text",
    "invalid_acoustic_value",
    "invalid_acoustic_attribute",
    "infinite_acoustic_value",
    "base_auto",
}
for observation in observations:
    if observation["case"] in negative:
        assert (
            observation["admitted"] is False and observation["status"] == 400
        ), observation
    else:
        assert (
            observation["admitted"] is True and "stage_error" not in observation
        ), observation
assert "shift the pitch by -1 step" in next(
    row["prompt"] for row in observations if row["case"] == "normalized_pitch"
)
assert "adjust the volume to 0.5" in next(
    row["prompt"]
    for row in observations
    if row["case"] == "explicit_instruction_precedence"
)
