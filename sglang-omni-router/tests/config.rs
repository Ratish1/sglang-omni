#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict configuration boundary tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sgl_omni_router::{Config, ConfigError, MAX_CONFIG_BYTES, MAX_DRAIN_TIMEOUT_MS};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-router-config-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated config test directory");
        Self(path)
    }

    fn write(&self, contents: &[u8]) -> PathBuf {
        let path = self.0.join("router.toml");
        fs::write(&path, contents).expect("write isolated config fixture");
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _cleanup_result = fs::remove_dir_all(&self.0);
    }
}

fn valid_config(listen: &str, drain_timeout_ms: u64, filter: &str) -> String {
    format!(
        "schema_version = 1\n\n[server]\nlisten = \"{listen}\"\n\n[shutdown]\ndrain_timeout_ms = {drain_timeout_ms}\n\n[logging]\nformat = \"json\"\nfilter = \"{filter}\"\n\n[admission]\nglobal = 128\n\n[http_generation]\nstreamed_request_max_bytes = 1048576\nconnect_timeout_ms = 1000\nrequest_timeout_ms = 5000\npool_idle_timeout_ms = 30000\npool_max_idle_per_host = 8\n\n[[workers]]\nworker_id = \"worker-a\"\nbase_url = \"http://127.0.0.1:8000/\"\n"
    )
}

fn with_max_connections(config: String, max_connections: u32) -> String {
    config.replace(
        "listen = \"127.0.0.1:30000\"",
        &format!("listen = \"127.0.0.1:30000\"\nmax_connections = {max_connections}"),
    )
}

fn load_bytes(contents: &[u8]) -> Result<Config, ConfigError> {
    let directory = TestDir::new();
    Config::load(&directory.write(contents))
}

#[test]
fn omitted_connection_cap_defaults_to_1024() {
    let config = load_bytes(valid_config("127.0.0.1:30000", 30_000, "info").as_bytes())
        .expect("complete strict configuration should be valid");
    assert_eq!(config.server.listen.to_string(), "127.0.0.1:30000");
    assert_eq!(config.server.max_connections, 1024);
    assert_eq!(config.shutdown.drain_timeout().as_millis(), 30_000);
}

#[test]
fn validates_connection_cap_boundaries() {
    for max_connections in [1, 65_535] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_ok());
    }

    for max_connections in [0, 65_536] {
        let config = with_max_connections(
            valid_config("127.0.0.1:30000", 30_000, "info"),
            max_connections,
        );
        assert!(load_bytes(config.as_bytes()).is_err());
    }
}

#[test]
fn rejects_unknown_duplicate_missing_and_unsupported_schema_fields() {
    let cases = [
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "listen = \"127.0.0.1:30000\"",
            "listen = \"127.0.0.1:30000\"\nunknown = true",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info").replace(
            "filter = \"info\"",
            "filter = \"info\"\nsecret = \"must-not-appear\"",
        ),
        valid_config("127.0.0.1:30000", 30_000, "info")
            + "\n[server]\nlisten = \"127.0.0.1:30001\"\n",
        "schema_version = 1\n".to_owned(),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("schema_version = 1", "schema_version = 2"),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_invalid_address_timeout_format_and_filter() {
    let cases = [
        valid_config("localhost:30000", 30_000, "info"),
        valid_config("127.0.0.1:30000", 0, "info"),
        valid_config(
            "127.0.0.1:30000",
            MAX_DRAIN_TIMEOUT_MS.saturating_add(1),
            "info",
        ),
        valid_config("127.0.0.1:30000", 30_000, "[invalid"),
        valid_config("127.0.0.1:30000", 30_000, &"x".repeat(257)),
        valid_config("127.0.0.1:30000", 30_000, "info")
            .replace("format = \"json\"", "format = \"yaml\""),
    ];

    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn rejects_oversized_and_non_utf8_input_without_echoing_contents() {
    let oversized = vec![b'x'; MAX_CONFIG_BYTES.saturating_add(1)];
    let oversized_error = load_bytes(&oversized).expect_err("oversized config must fail");
    assert!(matches!(oversized_error, ConfigError::TooLarge { .. }));

    let invalid_utf8 = [0xff, b's', b'e', b'c', b'r', b'e', b't'];
    let encoding_error = load_bytes(&invalid_utf8).expect_err("non-UTF-8 config must fail");
    assert!(matches!(encoding_error, ConfigError::Encoding(_)));
    assert!(!encoding_error.to_string().contains("secret"));
}

#[test]
fn read_errors_do_not_disclose_the_config_path() {
    let path = Path::new("/definitely-not-present/secret-router.toml");
    let error = Config::load(path).expect_err("missing config must fail");
    assert!(!error.to_string().contains("secret-router"));
}

#[test]
fn core_http_schema_rejects_unknowns_invalid_bounds_and_nonexact_workers() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info");
    let cases = [
        base.replace("global = 128", "global = 0"),
        base.replace(
            "streamed_request_max_bytes = 1048576",
            "streamed_request_max_bytes = 0",
        ),
        base.replace("connect_timeout_ms = 1000", "connect_timeout_ms = 0"),
        base.replace("pool_max_idle_per_host = 8", "pool_max_idle_per_host = 0"),
        base.replace(
            "pool_max_idle_per_host = 8",
            "pool_max_idle_per_host = 1025",
        ),
        base.replace("worker_id = \"worker-a\"", "worker_id = \"bad worker\""),
        base.replace(
            "base_url = \"http://127.0.0.1:8000/\"",
            "base_url = \"http://worker.invalid:8000/\"",
        ),
        format!(
            "{base}\n[[workers]]\nworker_id = \"worker-b\"\nbase_url = \"http://127.0.0.1:8001/\"\n"
        ),
        base.replace("global = 128", "global = 128\nfuture_limit = 1"),
    ];
    for contents in cases {
        assert!(load_bytes(contents.as_bytes()).is_err());
    }
}

#[test]
fn hostname_worker_requires_and_accepts_one_pinned_ip() {
    let base = valid_config("127.0.0.1:30000", 30_000, "info").replace(
        "base_url = \"http://127.0.0.1:8000/\"",
        "base_url = \"http://worker.invalid:8000/\"\nresolved_ip = \"127.0.0.1\"",
    );
    assert!(load_bytes(base.as_bytes()).is_ok());
}
