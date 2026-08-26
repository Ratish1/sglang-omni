# SGLang-Omni Rust router

`sgl-omni-router` is a standalone Rust router for static SGLang-Omni chat
workers. It serves `POST /v1/chat/completions`, selects compatible healthy
workers from correlated startup profiles, and preserves one direct upstream
attempt with bounded admission and joined shutdown.

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
generation_http = 64

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

Repeat the complete `[[workers]]`, `[workers.capacity]`, and correlated
`[[workers.service_profiles]]` group for each static replica or heterogeneous
worker. The stable worker fields are `worker_id`, `base_url`, and optional
`resolved_ip`; a DNS authority requires a pinned `resolved_ip`, while Host and
TLS SNI continue to use the URL authority. Trust domains never cross.

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

## Routing and resource ownership

At startup the router proves whether a trust-scoped cohort is content-blind:
all generation authorities in that scope must have identical default-model
semantics and identical correlated profile rows. Equal replicas and a sole
route authority can satisfy the proof; worker count by itself cannot.

Every accepted fixed-length request in a proven cohort uses direct streaming
request and response adapters, regardless of body size. The router does not
buffer, parse, spawn classification work, or reserve byte/classifier budgets on
that path. Malformed JSON and unsupported models are delegated to equivalent
workers.

When the proof is absent, the router reserves aggregate byte capacity, buffers
once, and performs one bounded Serde classification in a bounded
`spawn_blocking` slot. It then selects one compatible worker and uses the same
direct response adapter. Body size controls acceptance only; it never decides
whether routing facts are needed.

Global and generation admission are fail-fast. Each worker has one exact
semaphore, which is the sole mutable load authority. Round robin is the
default. Least requests snapshots occupancy, orders deterministic ties, and
reserves under one short generation-policy guard; no lock crosses network or
body work. All permits remain held through response EOF/error/drop.

The route accepts HTTP/1.1 `POST` without a query, exactly one valid
`Content-Length`, and `application/json` with an optional UTF-8 charset. It
rejects transfer encoding, trailers, expectations, content encoding, route
hints, ambiguous lengths, and oversized bodies before admission and dispatch.
Non-POST methods use the same bounded JSON error and request-ID contract.

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
exists in the configured chat trust domain. Exhausted capacity remains
healthy; draining or unhealthy workers are not dispatchable. No router-local
`/health`, worker CRUD, media, WebSocket, or metrics route is registered.

On the first `SIGINT` or `SIGTERM`, readiness fails, admission and exact worker
semaphores close, health tasks are cancelled, and owned server/health tasks are
joined within `shutdown.drain_timeout_ms`. A distinct second signal forces a
failed shutdown. Health never terminates worker processes and is not a circuit
breaker.
