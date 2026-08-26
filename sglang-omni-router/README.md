# SGLang-Omni Rust router

`sgl-omni-router` is the standalone Rust router process for SGLang-Omni. It
currently provides a strict one-worker `POST /v1/chat/completions` relay plus
process liveness and owned graceful shutdown. The configured worker owns JSON,
model, and request-semantic validation; the router owns the HTTP envelope,
request identity, bounds, transport, and resource lifetime.

## Development setup

Install [Rustup](https://rustup.rs/), then enter this directory. The checked-in
`rust-toolchain.toml` selects Rust 1.97.1 with rustfmt and Clippy. Rust 1.90.0
is the minimum supported Rust version and is used only for the separate MSRV
check.

```console
rustup toolchain install 1.90.0 --profile minimal
cargo build --locked
cargo test --locked
cargo +1.90.0 check --workspace --all-targets --all-features --locked
```

Install `cargo-deny` 0.20.2 to run the dependency-policy check locally:

```console
cargo deny --locked check
```

The complete formatting, lint, build, test, documentation, dependency, and
MSRV commands are recorded in `.github/workflows/rust-router.yml`.

## Configuration and CLI

Create `router.toml`:

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

[admission]
global = 128

[http_generation]
streamed_request_max_bytes = 536870912
connect_timeout_ms = 5000
request_timeout_ms = 1800000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 8

[[workers]]
worker_id = "omni-a"
base_url = "http://127.0.0.1:8000/"
```

This branch requires exactly one `[[workers]]` record. An IP-literal
`base_url` is already pinned. A DNS-name authority additionally requires, for
example, `resolved_ip = "127.0.0.1"`; TCP then uses that address while the URL
authority remains the HTTP Host and TLS certificate/SNI identity. Origin URLs
cannot contain credentials, a non-root path, query, or fragment.

All fields are explicit. Unknown, duplicate, and future-only fields are
rejected. Capacities and timeouts must be nonzero and within their documented
bounds. The configuration is limited to 64 KiB and must be UTF-8. Diagnostics
never include the configuration contents or path. The logging filter comes
only from this file; `RUST_LOG` is not read.

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

The route accepts HTTP/1.1 `POST` without a query, exactly one valid
`Content-Length`, and `application/json` with an optional UTF-8 charset. It
rejects transfer encoding, trailers, expectations, content encoding, route
hints, ambiguous lengths, and oversized bodies before dispatch. The request
body is not parsed or reserialized.

One valid printable `x-request-id` of at most 128 bytes is preserved. A missing
ID is generated. Duplicate, empty, or oversized representable IDs are rejected
and replaced on the bounded error response. Invalid raw HTTP header bytes may
be rejected by the HTTP parser before route dispatch. The canonical value is
sent to the worker and echoed downstream.

Admission is fail-fast. One permit is retained until response EOF, response
error, or downstream drop. The request timeout covers upload, connection, and
upstream response headers. After headers are committed, response streaming has
no wall-clock deadline: it ends on upstream EOF/error, downstream disconnect,
or process drain. There is one upstream attempt and no queue or retry.

The shared Reqwest client uses HTTP/1.1 pooling and a pinned target. Redirects,
ambient proxies, automatic retries, and automatic response decompression are
disabled. Responses preserve accepted status, encoded bytes, content framing,
and duplicate allowlisted cache headers; hop-by-hop, topology, cookie, and
unapproved headers are removed.

`--help` and `--version` do not require a configuration file.

## Process behavior

While the listener is serving, exact `GET /live` returns `200` with `live\n`.
No `/ready`, `/health`, worker, routing-policy, media, WebSocket, or metrics
route is registered in this branch.

The listener accepts at most `server.max_connections` active connections. On
the first `SIGINT` or `SIGTERM`, it stops accepting new sockets and drains all
owned connection tasks for at most `shutdown.drain_timeout_ms`. A distinct
second signal forces shutdown. A drain deadline or forced shutdown exits with
a failure status; a completed graceful drain exits successfully.
