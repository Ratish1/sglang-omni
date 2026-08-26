from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import (
    assert_workers_healthy,
    assert_zero_in_flight,
    benchmark_process_env,
    counter_moved,
    local_process_env,
    parse_proc_stat,
    request_counters,
    substitute_placeholders,
)


class CommonTests(unittest.TestCase):
    def test_substitute_placeholders_is_argument_local(self) -> None:
        command = ["python", "--url", "{router_url}/v1", "--out={output_dir}"]
        self.assertEqual(
            substitute_placeholders(
                command,
                {"router_url": "http://127.0.0.1:30000", "output_dir": "/tmp/run"},
            ),
            ["python", "--url", "http://127.0.0.1:30000/v1", "--out=/tmp/run"],
        )

    def test_loopback_process_environment_removes_proxies(self) -> None:
        with mock.patch.dict(
            "os.environ", {"HTTPS_PROXY": "proxy", "KEEP": "yes"}, clear=True
        ):
            self.assertEqual(local_process_env(), {"KEEP": "yes"})

    def test_benchmark_environment_preserves_proxy_and_bypasses_loopback(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"HTTPS_PROXY": "proxy", "NO_PROXY": "internal.example"},
            clear=True,
        ):
            environment = benchmark_process_env()
        self.assertEqual(environment["HTTPS_PROXY"], "proxy")
        self.assertEqual(
            environment["NO_PROXY"], "internal.example,127.0.0.1,localhost"
        )
        self.assertEqual(environment["no_proxy"], environment["NO_PROXY"])

    def test_proc_stat_parser_handles_spaces_in_process_name(self) -> None:
        fields = ["S", "10", "20", "30"] + ["0"] * 7 + ["7", "11"] + ["0"] * 8 + ["13"]
        self.assertEqual(
            parse_proc_stat("123 (router worker) " + " ".join(fields)),
            (20, 18, 13),
        )

    def test_zero_in_flight_accepts_all_ownership_rows(self) -> None:
        assert_zero_in_flight(
            {
                "admission": [{"class": "global", "in_flight": 0}],
                "workers": [
                    {"capacity": [{"class": "generation_http", "in_flight": 0}]}
                ],
            }
        )

    def test_zero_in_flight_rejects_nonzero_and_missing_shape(self) -> None:
        with self.assertRaisesRegex(AssertionError, "nonzero Rust ownership"):
            assert_zero_in_flight(
                {
                    "admission": [{"in_flight": 1}],
                    "workers": [{"capacity": [{"in_flight": 0}]}],
                }
            )
        with self.assertRaisesRegex(AssertionError, "no admission/capacity"):
            assert_zero_in_flight({"workers": []})

    def test_request_counter_filter_and_movement(self) -> None:
        before = request_counters(
            '# help\nhttp_requests_total{route="chat"} 4\nprocess_cpu_seconds_total 9\n'
        )
        after = request_counters('http_requests_total{route="chat"} 7\n')
        self.assertEqual(before, {'http_requests_total{route="chat"}': 4.0})
        self.assertTrue(counter_moved(before, after))
        self.assertIsNone(counter_moved({}, after))

    def test_post_trial_health_accepts_rust_and_python_shapes(self) -> None:
        assert_workers_healthy(
            {
                "ready": True,
                "workers": [
                    {"worker_id": "a", "health": "healthy", "disposition": "serving"},
                    {"worker_id": "b", "health": "healthy", "disposition": "serving"},
                ],
            },
            "rust",
        )
        assert_workers_healthy(
            {
                "healthy_workers": 2,
                "routable_workers": 2,
                "workers": [
                    {
                        "display_id": "a",
                        "health_state": "healthy",
                        "routable": True,
                        "active_requests": 0,
                    },
                    {
                        "display_id": "b",
                        "health_state": "healthy",
                        "routable": True,
                        "active_requests": 0,
                    },
                ],
            },
            "python",
        )


if __name__ == "__main__":
    unittest.main()
