from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import router_microbench
import run_candidate
import run_direct_pair
import run_repo_benchmark


class CandidateCommandTests(unittest.TestCase):
    def namespace(self, candidate: str, policy: str) -> argparse.Namespace:
        return argparse.Namespace(
            candidate=candidate,
            rust_binary=Path("/bin/sgl-omni-router"),
            rust_config=Path("asr.toml"),
            python="python3",
            router_port=30000,
            worker_url=["http://127.0.0.1:8011", "http://127.0.0.1:8012"],
            model="model-id",
            policy=policy,
        )

    def test_rust_command_uses_only_binary_and_config(self) -> None:
        self.assertEqual(
            run_candidate.build_router_command(self.namespace("rust", "round_robin")),
            ["/bin/sgl-omni-router", "--config", "asr.toml"],
        )

    def test_python_command_keeps_both_workers_and_policy(self) -> None:
        command = run_candidate.build_router_command(
            self.namespace("python", "least_request")
        )
        self.assertEqual(command.count("http://127.0.0.1:8011"), 1)
        self.assertEqual(command.count("http://127.0.0.1:8012"), 1)
        self.assertEqual(command[command.index("--policy") + 1], "least_request")

    def test_candidate_parser_requires_two_worker_urls(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_candidate.parse_args(
                [
                    "--candidate",
                    "python",
                    "--output-dir",
                    "out",
                    "--worker-url",
                    "http://127.0.0.1:8011",
                    "--model",
                    "m",
                    "--policy",
                    "round_robin",
                    "--",
                    "true",
                ]
            )

    def test_rust_parser_verifies_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "router.toml"
            router_microbench.write_rust_config(
                config, (18011, 18012), 30000, "round_robin", "error"
            )
            arguments = [
                "--candidate",
                "rust",
                "--output-dir",
                "out",
                "--rust-config",
                str(config),
                "--worker-url",
                "http://127.0.0.1:18011",
                "--worker-url",
                "http://127.0.0.1:18012",
                "--model",
                "bench-model",
                "--policy",
                "round_robin",
                "--",
                "true",
            ]
            self.assertEqual(run_candidate.parse_args(arguments).policy, "round_robin")
            arguments[arguments.index("round_robin")] = "least_requests"
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                run_candidate.parse_args(arguments)

    def test_direct_pair_parser_requires_command_and_two_ports(self) -> None:
        args = run_direct_pair.parse_args(
            [
                "--output-dir",
                "out",
                "--worker-port",
                "8011",
                "--worker-port",
                "8012",
                "--",
                "python",
                "bench.py",
                "--port",
                "{router_port}",
            ]
        )
        self.assertEqual(args.worker_port, [8011, 8012])
        self.assertEqual(args.benchmark[-1], "{router_port}")

    def test_repo_benchmark_wrapper_is_narrow_and_preserves_arguments(self) -> None:
        args = run_repo_benchmark.parse_args(
            [
                "benchmarks.eval.benchmark_omni_mmmu",
                "--",
                "--base-url",
                "http://127.0.0.1:30000",
            ]
        )
        self.assertEqual(args.arguments[-1], "http://127.0.0.1:30000")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_repo_benchmark.parse_args(["arbitrary.module"])

    def test_repo_benchmark_wrapper_replaces_only_redundant_waiter(self) -> None:
        fake = SimpleNamespace(wait_for_service=lambda *_args: None)

        def entrypoint() -> int:
            self.assertIsNone(fake.wait_for_service("ignored"))
            self.assertEqual(sys.argv[-1], "16")
            return 0

        fake.main = entrypoint
        with mock.patch.object(
            run_repo_benchmark.importlib, "import_module", return_value=fake
        ):
            self.assertEqual(
                run_repo_benchmark.main(
                    [
                        "benchmarks.eval.benchmark_omni_mmmu",
                        "--",
                        "--max-concurrency",
                        "16",
                    ]
                ),
                0,
            )


class MicrobenchTests(unittest.TestCase):
    def test_payload_has_requested_size_and_valid_json(self) -> None:
        import json

        value = router_microbench.payload(1_048_576)
        self.assertEqual(len(value), 1_048_576)
        self.assertEqual(json.loads(value)["model"], "bench-model")

    def test_oha_command_uses_data_file_not_large_argv(self) -> None:
        command = router_microbench.oha_command(
            "/usr/bin/oha",
            "http://127.0.0.1:30000/v1/chat/completions",
            32,
            1000,
            "large_json",
            Path("large.json"),
        )
        self.assertIn("-D", command)
        self.assertNotIn("-d", command)
        self.assertEqual(
            command[command.index("--output-format") + 1],
            "json",
        )

    def test_parse_oha_extracts_machine_metrics(self) -> None:
        parsed = router_microbench.parse_oha(
            {
                "metrics": {
                    "requests_per_sec": 123.0,
                    "success_rate": 1.0,
                    "latency_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
                },
                "statusCodeDistribution": {"200": 100},
                "errorDistribution": {},
            }
        )
        self.assertEqual(parsed["requests_per_second"], 123.0)
        self.assertEqual(parsed["p99_ms"], 30.0)
        self.assertEqual(parsed["errors"], {})

    def test_parse_oha_converts_legacy_latency_seconds_to_ms(self) -> None:
        parsed = router_microbench.parse_oha(
            {
                "summary": {"requestsPerSec": 10.0, "successRate": 1.0},
                "latencyPercentiles": {"p50": 0.01, "p95": 0.02, "p99": 0.03},
                "statusCodeDistribution": {"200": 10},
                "errorDistribution": {},
            }
        )
        self.assertEqual(parsed["p99_ms"], 30.0)

    def test_generated_rust_config_is_strictly_bounded_and_homogeneous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "router.toml"
            router_microbench.write_rust_config(
                path, (18011, 18012), 30000, "round_robin", "error"
            )
            text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("[[workers]]"), 2)
        self.assertIn('strategy = "round_robin"', text)
        self.assertIn('filter = "error"', text)
        self.assertIn("buffered_request_total_bytes = 1073741824", text)

    def test_proxy_bound_decision_allows_rust_at_direct_ceiling(self) -> None:
        def router(candidate: str, rps: float, cpu: float, p99: float) -> dict:
            return {
                "kind": "router",
                "candidate": candidate,
                "policy": "round_robin",
                "log_filter": "error",
                "scenario": "small_json",
                "concurrency": 1,
                "oha": {
                    "requests_per_second": rps,
                    "p99_ms": p99,
                    "errors": {},
                    "success_rate": 1.0,
                },
                "process_group": {"cpu_seconds": cpu},
            }

        direct = {
            "kind": "direct_pair",
            "scenario": "small_json",
            "concurrency": 1,
            "requests_per_second": 130.0,
            "children": [{"success_rate": 1.0, "errors": {}}],
        }
        decision = router_microbench.decide(
            [
                direct,
                router("python", 100.0, 2.0, 10.0),
                router("rust", 120.0, 1.5, 10.5),
            ]
        )
        self.assertEqual(decision["comparisons"][0]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
