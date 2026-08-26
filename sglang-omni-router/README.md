# SGLang-Omni Rust router

`sgl-omni-router` is a standalone Rust router for static SGLang-Omni chat,
speech, batch speech, transcription, and translation workers. It selects
compatible healthy workers from correlated startup profiles and preserves one
direct upstream attempt with bounded admission and joined shutdown.

## Development setup

Install [Rustup](https://rustup.rs/), then enter this directory. The checked-in
`rust-toolchain.toml` selects Rust 1.97.1 with rustfmt and Clippy. Rust 1.90.0
is the minimum supported Rust version and is used only for the separate MSRV
check.

```console
rustup toolchain install 1.90.0 --profile minimal
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo build --workspace --all-features --locked
cargo test --workspace --all-features --locked
cargo +1.90.0 check --workspace --all-targets --all-features --locked
```

Install `cargo-deny` 0.20.2 to run the dependency-policy check locally:

```console
cargo deny --locked check
```

The complete formatting, lint, build, test, documentation, dependency, and
MSRV commands are recorded in `.github/workflows/rust-router.yml`.

## Configuration

The manifest is strict and limited to 64 KiB. Unknown and duplicate fields,
uncorrelated profiles, invalid defaults, duplicate workers/targets, and
unbounded values fail startup.

```toml
schema_version = 1

[server]
listen = "127.0.0.1:30000"
max_connections = 1024

[shutdown]
drain_timeout_ms = 30000

[logging]
format = "json"
filter = "info"

[router]
strategy = "round_robin" # or "least_requests"
max_concurrent_classifications = 4

[admission]
global = 128
# Required only when [http_generation] is configured.
generation_http = 64
# Configure only the media classes used by enabled routes.
speech_http = 32
speech_batch = 64
transcription_http = 32

[health]
interval_ms = 5000
timeout_ms = 1000
success_threshold = 2
failure_threshold = 3
max_concurrent_probes = 16

[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 8388608
buffered_request_total_bytes = 268435456
streamed_request_max_bytes = 536870912
connect_timeout_ms = 5000
request_timeout_ms = 1800000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8 # accepted range: 1 through 1024

[[workers]]
worker_id = "omni-a"
base_url = "http://127.0.0.1:8000/"
trust_domain = "local"
default_model_id = "omni"
health_path = "/health"

[workers.capacity]
generation_http = 8

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "image", "audio", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
```

Enable any subset of the media HTTP routes with one shared transport policy:

```toml
[http_media]
routes = ["speech", "speech_batch", "transcription", "translation"]
trust_domain = "local"
buffered_request_max_bytes = 8388608
buffered_request_total_bytes = 268435456
streamed_request_max_bytes = 536870912
connect_timeout_ms = 5000
request_timeout_ms = 1800000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8 # accepted range: 1 through 1024
```

`[http_generation]` and `[http_media]` are independently optional, but at least
one must be configured. Each handler owns its transport settings, pooled
client, request timeout, and aggregate byte budget. They share the bounded
classifier semaphore, worker health, routing policy, and response-body lease.

Add only the capacity and profile rows a worker actually serves. A
media-only worker may omit `generation_http` capacity and generation profiles,
and may omit `default_model_id`. Requests for such a service must then provide
an explicit body/form model; a route-model assertion only claims a configured
worker default and cannot create one. Speech-to-text rows carry exactly one
`task`, so transcription and translation capabilities cannot combine
accidentally:

```toml
[workers.capacity]
speech_http = 8
speech_batch = 32
transcription_http = 8

[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts"]
response_formats = ["mp3", "opus", "aac", "flac", "wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false

[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts"]
response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
managed_voice = false
max_batch_size = 32
effective_features = ["model", "format", "task", "reference", "voice"]

[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["asr"]
task = "transcribe" # use a separate row with task = "translate"
response_formats = ["json", "text", "verbose_json", "srt", "vtt", "sse"]
media_profiles = ["audio", "audio_video"]
stream_modes = ["non_streaming", "streaming"]
```

`speech_batch` capacity is measured in items, not HTTP envelopes. One batch is
never split: the router atomically reserves its complete item count from the
class and selected worker, and one response lease returns every credit. Every
`max_batch_size` must fit both the worker's `speech_batch` capacity and the
configured `admission.speech_batch` limit.

Repeat the complete `[[workers]]`, `[workers.capacity]`, and correlated
`[[workers.service_profiles]]` group for each static replica or heterogeneous
worker. The stable worker fields are `worker_id`, `base_url`, and optional
`resolved_ip`; a DNS authority requires a pinned `resolved_ip`, while Host and
TLS SNI remain the configured URL authority. If `default_model_id` is set, that
model must appear in a profile row for every service class advertised by the
worker, and separately for each advertised transcription or translation task.
Trust domains never cross.

A profile row is a correlated capability claim. The router never combines a
model from one row with modalities, forms, outputs, or stream support from
another. A missing request model uses a worker default only when that default
is unambiguous for the configured route trust domain.

Validate configuration without binding the listener:

```console
cargo run --locked -- --config router.toml --check-config
```

Run the service:

```console
cargo run --locked -- --config router.toml
```

Send a fixed-length request:

```console
curl --http1.1 --request POST http://127.0.0.1:30000/v1/chat/completions \
  --header 'content-type: application/json' \
  --header 'x-request-id: example-1' \
  --data-binary '{"model":"omni","messages":[{"role":"user","content":"hello"}]}'
```

The media routes are:

- `POST /v1/audio/speech` for encoded audio or streaming PCM;
- `POST /v1/audio/speech/batch` for one ordered, unsplit batch;
- `POST /v1/audio/transcriptions` for multipart speech recognition;
- `POST /v1/audio/translations` for multipart speech translation.

For transcription and translation, non-streaming `response_format` supports
`json`, `text`, `verbose_json`, `srt`, and `vtt`. With `stream=true`, the form
format must be `json` or `text`, and the worker response is relayed as SSE.
Multipart bodies are forwarded byte-for-byte and are never reconstructed.

```console
curl --http1.1 http://127.0.0.1:30000/v1/audio/transcriptions \
  --form 'file=@sample.wav' \
  --form 'model=asr' \
  --form 'response_format=srt'

curl --http1.1 http://127.0.0.1:30000/v1/audio/speech \
  --header 'content-type: application/json' \
  --data-binary '{"model":"tts","input":"hello","response_format":"opus"}' \
  --output speech.opus
```

On media routes, `x-sglang-omni-route-model` and
`x-sglang-omni-route-stream` are bounded, router-local metadata assertions and
are never sent to workers. An explicit body or form value must match its
assertion. An absent model can be asserted only against the selected worker's
configured default. An absent stream value means `false`, so a header-only
`true` assertion is rejected; `true` is valid only when the body or form also
explicitly requests streaming.

## Routing and resource ownership

At startup the router proves content-blind cohorts separately for generation
and for each media service and speech-to-text task. Generation authorities in a
trust domain must have identical default-model semantics and correlated rows;
media workers must match for the specific service and task. Equal replicas and
a sole route authority can satisfy a proof; worker count by itself cannot.
`speech_batch` is always classified because its item credits come from the
body.

Only fixed-length non-batch requests are eligible for the direct fast path. In
a proven cohort they use direct streaming request and response adapters without
buffering, parsing, classification work, or byte/classifier permits. Malformed
JSON and unsupported models are delegated to equivalent workers.

When a request is not direct-eligible, the router reserves aggregate byte
capacity, buffers once, and performs one bounded Serde classification in a
bounded `spawn_blocking` slot. It then selects one compatible worker and uses
the same direct response adapter. Body size controls acceptance only; it never
decides whether routing facts are needed.

Global and route-class admission are fail-fast. Each worker has one exact
semaphore per configured class, which is the sole mutable load authority. Round
robin is the default. Least requests snapshots occupancy, orders deterministic
ties, and reserves under one short policy guard; no lock crosses network or
body work. All permits remain held through response EOF/error/drop.

The chat-generation route accepts HTTP/1.1 `POST` without a query, exactly one
valid `Content-Length`, and `application/json` with an optional UTF-8 charset.
It rejects transfer encoding, trailers, expectations, content encoding, route
internal worker-routing headers, ambiguous lengths, and oversized bodies before
admission and dispatch. Media requests without a usable fixed length take the
bounded buffered path. Non-POST methods use the same bounded JSON error and
request-ID contract.

One valid printable `x-request-id` of at most 128 bytes is preserved. A missing
ID is generated. Duplicate, empty, or oversized representable IDs are rejected
and replaced on the bounded error response. Invalid raw HTTP header bytes may
be rejected by the HTTP parser before route dispatch. The canonical value is
sent to the worker and echoed downstream.

Admission is fail-fast. Global, generation, and exact worker permits are
retained until response EOF, response error, or downstream drop. The request
timeout covers upload, connection, classification when required, and upstream
response headers. After headers are committed, response streaming has no
wall-clock deadline: it ends on upstream EOF/error, downstream disconnect, or
process drain. There is one upstream attempt and no queue or retry.

The shared Reqwest client uses HTTP/1.1 pooling and pinned targets. Redirects,
ambient proxies, automatic retries, and automatic response decompression are
disabled. Responses preserve accepted status, encoded bytes, content framing,
and duplicate allowlisted cache headers; hop-by-hop, topology, cookie, and
unapproved headers are removed.

`--help` and `--version` do not require a configuration file.

## Health, readiness, and shutdown

Workers start `Unknown`. One joined task per worker performs status-only probes
through one shared probe semaphore. The deployment defaults mark a worker
unhealthy after three consecutive failures and recover it after two
consecutive successes. Immediate notifications coalesce. Transport or protocol
faults request a probe but do not directly change health; ordinary worker 4xx
responses are relayed and do not mark the worker unhealthy.

Exact `GET /live` reports process liveness. `GET /ready` is registered and
returns `200` only while serving and at least one compatible healthy worker
exists for chat and every enabled media route. Exhausted capacity remains
healthy; draining or unhealthy workers are not dispatchable. No router-local
`/health`, worker CRUD, WebSocket, or metrics route is registered. Disabled
media routes are not installed and return `404`.

On the first `SIGINT` or `SIGTERM`, readiness fails, admission and exact worker
semaphores close, health tasks are cancelled, and owned server/health tasks are
joined within `shutdown.drain_timeout_ms`. A distinct second signal forces a
failed shutdown. Health never terminates worker processes and is not a circuit
breaker.
