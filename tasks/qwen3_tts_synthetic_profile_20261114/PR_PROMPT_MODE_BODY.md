## Problem

Official `qwen_tts` takes `non_streaming_mode` on all three generation calls. `generate_custom_voice` and `generate_voice_design` default it to `True`, which puts the whole target text in the talker's prompt. `generate_voice_clone` defaults it to `False`, which puts only the first text token in the prompt and feeds the rest to the talker one token per codec frame (official calls this a simulation of streaming text input).

The speech API dropped the field. `CreateSpeechRequest`, the batch models and the WebSocket session config did not declare it, so it never reached the model, and the CustomVoice and VoiceDesign builders set `True` whatever the request said. A client that asked for the streaming-text prompt received the whole-text prompt with no error.

Declaring the field alone would have broken streaming. `resolve_stream_codec_output` also read `non_streaming_mode` from the speech API's parameters as "do not stream codec output", so a streamed CustomVoice request sending the official default `true` would have received all its audio at the end.

## Change

- `CreateSpeechRequest`, `CreateSpeechBatchRequest`, `SpeechBatchItem` and `SpeechStreamSessionConfig` declare `non_streaming_mode`. The speech service refuses a value that is not a boolean and forwards the field in `tts_params`.
- The CustomVoice and VoiceDesign builders keep the requested value. When a request does not set it, each task keeps official's default: `true` for CustomVoice and VoiceDesign, `false` for Base.
- `resolve_stream_codec_output` reads `non_streaming_mode` only from the raw request parameters, the per-request switch #1852 kept. The speech API's field sets the prompt mode alone, so a streamed request streams at either value.
- The SeedTTS benchmark takes `--non-streaming-mode` and `--no-non-streaming-mode`.
- The speech parameter table and the Qwen3-TTS reference document the field. The native Apple Silicon server ignores it and always puts the whole text in the prompt, and its platform page says so.

## Results

Every value on every task: 1.7B Base, CustomVoice and VoiceDesign servers on one H100, main against this change, three inputs each, seeded, one request at a time, streamed and whole.

- A request without the field, or with its task's default value, gives the same audio bytes as main's on all three models, streamed and whole.
- The other value gives different bytes, so the prompt mode reaches the model. Main gives `true` and `false` the same bytes, since it drops the field.
- Every streamed request streams at every value: 15 to 21 chunks, first byte 16 to 79 ms. That includes `true`, which the old reading would have turned into one response at the end.

SEEDTTS_PLACEHOLDER

What the streaming-text prompt does to CustomVoice output: SeedTTS EN, all 1,088 prompts, Ryan, streaming at 16 concurrent requests, main with the default prompt against this change with `non_streaming_mode: false`, each order on one H100.

| | Main, default prompt | This change, `false` |
|---|---|---|
| WER corpus | 1.74 / 1.45 % | 2.38 / 1.99 % |
| WER below 50 % | 1.42 / 1.42 % | 1.68 / 1.62 % |
| Audio mean | 4.66 / 4.69 s | 4.87 / 4.93 s |
| First audio mean at 1 request | 13.4 / 13.2 ms | 13.3 / 13.2 ms |

- That is the cost of the mode official names a simulation of streaming text, paid only by a client that asks for it. A request that does not set the field gets the same prompt as on main.

## Tests

- `test_qwen3_tts_speech_api_prompt_mode_keeps_codec_streaming` builds CustomVoice, VoiceDesign and Base requests with each value in `tts_params`: the state keeps the value and streams codec output. On main, CustomVoice and VoiceDesign keep `true` for a `false` request, and every `true` request stops streaming.
- `test_speech_service_normalizes_tts_extension_fields_into_tts_params` now carries the field through the speech service. On main it is dropped.
- `test_prompt_mode_flag_reaches_every_request_and_the_record` checks the benchmark flag reaches every request and the results record.
