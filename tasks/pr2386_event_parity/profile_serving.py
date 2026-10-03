"""Exercise request profiling toggles on a running 1.7B streaming server."""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True)
    config = TtsSeedttsBenchmarkConfig(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        meta="zhaochenyang20/seed-tts-eval-arrow",
        host="127.0.0.1",
        stream=True,
        response_format="pcm",
        concurrency=16,
        max_samples=16,
        server_config="examples/configs/qwen3_tts_1_7b.yaml",
    )
    summaries = {}
    event_directories = []
    with managed_omni_server(
        model_path=config.model,
        port=config.port,
        host=config.host,
        log_file=output / "server.log",
        server_config=config.server_config,
        max_running_requests=config.max_running_requests,
        cuda_graph_max_bs=config.cuda_graph_max_bs,
        timeout=1200,
        wait_for_gpu_release=False,
    ):
        with requests.Session() as session:
            session.trust_env = False
            for block, active in (
                ("off", False),
                ("on", True),
                ("off-again", False),
                ("on-again", True),
            ):
                if active:
                    directory = output / block / "events"
                    response = session.post(
                        f"http://127.0.0.1:{config.port}/start_request_profile",
                        json={"run_id": block, "event_dir": str(directory)},
                        timeout=30,
                    )
                    response.raise_for_status()
                    assert response.json()["run_id"] == block
                    event_directories.append(directory)
                    time.sleep(1)
                else:
                    pass
                previous_sizes = {
                    str(path): path.stat().st_size
                    for directory in event_directories
                    for path in directory.glob("*.jsonl")
                }
                config.output_dir = str(output / block)
                started_ns = time.time_ns()
                result = asyncio.run(run_tts_seedtts_benchmark(config))
                finished_ns = time.time_ns()
                assert result["summary"]["completed_requests"] == 16
                assert result["summary"]["failed_requests"] == 0
                summaries[block] = result["summary"]
                if active:
                    response = session.post(
                        f"http://127.0.0.1:{config.port}/stop_request_profile",
                        json={"run_id": block},
                        timeout=30,
                    )
                    response.raise_for_status()
                    time.sleep(1)
                    events = [
                        json.loads(line)
                        for path in directory.glob("*.jsonl")
                        for line in path.read_text().splitlines()
                    ]
                    decode_events = [
                        event
                        for event in events
                        if event["event_name"].startswith("qwen3_tts_vocoder_decode_")
                    ]
                    counts = Counter(event["event_name"] for event in decode_events)
                    assert set(counts) == {
                        f"qwen3_tts_vocoder_decode_{step}"
                        for step in (
                            "enqueued",
                            "dispatched",
                            "launched",
                            "resolved",
                            "committed",
                        )
                    }
                    committed_frames = defaultdict(list)
                    for event in sorted(
                        decode_events, key=lambda event: event["timestamp_ns"]
                    ):
                        assert event["stage"] == "vocoder"
                        assert event["run_id"] == block
                        assert started_ns <= event["timestamp_ns"] <= finished_ns
                        metadata = event["metadata"]
                        assert (
                            0
                            <= metadata["emitted_generated_frames"]
                            <= metadata["generated_frames"]
                        )
                        assert metadata["monotonic_s"] > 0
                        if event["event_name"].endswith("_committed"):
                            assert metadata["samples"] > 0
                            committed_frames[event["request_id"]].append(
                                metadata["emitted_generated_frames"]
                            )
                        else:
                            pass
                    assert len(committed_frames) == 32
                    for frames in committed_frames.values():
                        assert all(
                            right > left for left, right in zip(frames, frames[1:])
                        )
                    summaries[block]["decode_event_counts"] = dict(counts)
                    summaries[block]["profiled_requests_including_warmup"] = len(
                        committed_frames
                    )
                else:
                    assert previous_sizes == {
                        str(path): path.stat().st_size
                        for directory in event_directories
                        for path in directory.glob("*.jsonl")
                    }
    (output / "audit.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
