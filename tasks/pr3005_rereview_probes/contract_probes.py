"""Review probes using real request lowering, Flow steps, and HTTP/WS output."""

import base64
import io
import json
import wave

import numpy as np
import torch
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.firered_tts3.embeddings import materialize_decode_embeddings
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.preprocessing import (
    preprocess_firered_tts3_payload,
)
from sglang_omni.models.firered_tts3.speech_validation import (
    FireRedTTS3SpeechRequestValidator,
)
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.firered_tts3 import (
    SmallFlowModel,
    make_flow_request,
    make_flow_result,
    make_flow_runner,
    make_speaker_tokenizer,
)
from tests.unit_test.fixtures.firered_tts3_preprocessing import (
    make_preprocessing_payload,
    preprocess_payload,
)


def emit(label, value):
    print(json.dumps({"probe": label, "observed": value}), flush=True)


def generation_probes():
    torch.manual_seed(27)
    model = SmallFlowModel().eval()
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-20)
    for batching in (False, True):
        runner = make_flow_runner(model, batching=batching)
        request = make_flow_request()
        request.data.state.max_patches = 1
        request.data.state.stop_threshold = 0.5
        for _ in range(3):
            runner.post_decode(
                make_flow_result(torch.zeros(1, 8)), None, None, [request]
            )
            request.data.req.output_ids.append(0)
            if request.data.req.finished():
                break
        emit(
            "max_patches",
            {
                "batching": batching,
                "requested": 1,
                "generated": len(request.data.latent_patches),
                "finished": request.data.req.finished(),
            },
        )
    payload = make_preprocessing_payload(params={"min_patches": 0})
    restored = preprocess_payload(payload)
    emit(
        "min_patches_stage_handoff",
        {
            "wire_value": payload.data["min_patches"],
            "restored": restored.min_patches,
            "direct_roundtrip": FireRedTTS3State.from_dict(
                FireRedTTS3State(min_patches=0).to_dict()
            ).min_patches,
        },
    )
    schedule_payload = make_preprocessing_payload()
    preprocess_firered_tts3_payload(
        schedule_payload,
        tokenizer=make_speaker_tokenizer(),
        config=FireRedTTS3Config(
            spk_in_dim=512,
            dit_in_channels=1600,
            generation_defaults={"n_timesteps": 10},
        ),
        n_timesteps=8,
        max_patches=400,
    )
    emit(
        "deployment_timestep_override",
        {
            "stage_n_timesteps": 8,
            "request_n_timesteps": schedule_payload.data["n_timesteps"],
        },
    )


class AudioBoundaryClient(Client):
    """Replace model inference only; retain the real Client audio encoder."""

    def __init__(self):
        self.speeds = []

    async def generate(self, request, request_id=None):
        self.speeds.append(request.metadata.get("tts_params", {}).get("speed"))
        yield GenerateChunk(
            request_id=request_id or "probe",
            modality="audio",
            audio_data=np.sin(np.arange(2400, dtype=np.float32) * 0.02),
            sample_rate=24000,
            finish_reason="stop",
        )

    async def abort(self, request_id, *args, **kwargs):
        return None


def speed_probes():
    reference = io.BytesIO()
    with wave.open(reference, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(np.zeros(2400, dtype=np.int16).tobytes())
    reference_uri = (
        "data:audio/wav;base64," + base64.b64encode(reference.getvalue()).decode()
    )
    backend = AudioBoundaryClient()
    app = create_app(
        backend, speech_request_validator_cls=FireRedTTS3SpeechRequestValidator
    )
    payload = {
        "input": "Hello.",
        "task_type": "AcousticEdit",
        "ref_audio": reference_uri,
        "speed": 0.8,
        "response_format": "pcm",
    }
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=payload)
        emit(
            "http_native_speed",
            {"status": response.status_code, "bytes": len(response.content)},
        )
        for stream_audio in (False, True):
            with client.websocket_connect("/v1/audio/speech/stream") as websocket:
                config = {
                    name: value for name, value in payload.items() if name != "input"
                }
                config["stream_audio"] = stream_audio
                websocket.send_json({"type": "session.config", "session": config})
                configured = websocket.receive_json()
                if configured["type"] != "session.configured":
                    emit("ws_config_error", configured)
                    continue
                websocket.send_json({"type": "input.text", "text": "Hello."})
                websocket.send_json({"type": "input.done"})
                byte_count = 0
                events = []
                while True:
                    message = websocket.receive()
                    if message.get("bytes") is not None:
                        byte_count += len(message["bytes"])
                    elif message.get("text") is not None:
                        event = json.loads(message["text"])
                        events.append(event["type"])
                        if event["type"] == "session.done":
                            break
                    else:
                        break
                emit(
                    "ws_native_speed",
                    {
                        "stream_audio": stream_audio,
                        "bytes": byte_count,
                        "events": events,
                    },
                )
    emit("native_speed_reaches_generation", backend.speeds)


def variant_probe():
    from sglang_omni.config.manager import ConfigManager

    configuration = ConfigManager.from_model_path("CKermit/FireRedTTS3-Instruct").config
    emit(
        "remote_instruct_config",
        {
            "variant": configuration.variant,
            "required_references": configuration.required_speech_reference_count,
            "required_transcript": configuration.speech_reference_text_required,
        },
    )
    validator = FireRedTTS3SpeechRequestValidator(
        default_model="tts",
        required_speech_reference_count=configuration.required_speech_reference_count,
        speech_reference_text_required=configuration.speech_reference_text_required,
    )
    try:
        validator.parse_generation_request(
            {"input": "Hello.", "task_type": "VoiceDesign", "instructions": "calm"}
        )
    except Exception as error:
        emit(
            "remote_instruct_voice_design",
            {"type": type(error).__name__, "error": str(error)},
        )


def embedding_probe():
    from types import SimpleNamespace

    from sglang_omni.models.firered_tts3.request_builders import TEXT_MODE_TEXT

    model = SmallFlowModel().eval().cuda()
    requests = [
        make_flow_request(index=index, variant="instruct") for index in range(3)
    ]
    for request in requests:
        request.data.text_ar_mode = TEXT_MODE_TEXT
    batch = SimpleNamespace(input_ids=torch.tensor([100, 101, 102], device="cuda"))
    with torch.inference_mode():
        materialize_decode_embeddings(model, batch, requests)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            materialize_decode_embeddings(model, batch, requests)
            torch.cuda.synchronize()
        expected = model.get_input_embeddings()(batch.input_ids)
        torch.testing.assert_close(batch.input_embeds, expected)
    emit(
        "text_embedding_roundtrip",
        {
            event.key: event.count
            for event in profile.key_averages()
            if any(
                name in event.key
                for name in (
                    "item",
                    "local_scalar",
                    "Memcpy",
                    "cudaStreamSynchronize",
                    "embedding",
                )
            )
        },
    )


if __name__ == "__main__":
    generation_probes()
    speed_probes()
    variant_probe()
    embedding_probe()
