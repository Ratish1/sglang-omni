# SPDX-License-Identifier: Apache-2.0
"""Compare held audio inputs across two native Thinker cohorts."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch
from audit_qwen3_omni_conditioning_provenance import audit
from compare_qwen3_omni_native_trajectories import difference


def main(root: Path, sample_id: str) -> None:
    torch.set_num_threads(1)
    directories = [root / arm / "captures" for arm in ("first", "second")]
    reports = [
        json.loads((directory / "thinker-input-trial.json").read_text())
        for directory in directories
    ]
    assert reports[0]["capture_sha256"] == reports[1]["capture_sha256"]
    for path, expected_hash in reports[0]["capture_sha256"].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected_hash, path
    snapshots = [
        torch.load(
            directory / "thinker-input-trial.pt",
            map_location="cpu",
            weights_only=True,
        )
        for directory in directories
    ]
    assert torch.equal(snapshots[0]["audio_rows"], snapshots[1]["audio_rows"])
    assert all(
        torch.equal(first, second)
        for first, second in zip(
            snapshots[0]["selected_prefix"],
            snapshots[1]["selected_prefix"],
            strict=True,
        )
    ), "Different cached prefixes cannot isolate the cohort effect"
    original_hidden = [
        torch.load(path, map_location="cpu", weights_only=True)["hidden_states"]
        for path in reports[0]["capture_sha256"]
        if Path(path).name == "thinker-hidden-4.pt"
    ]
    assert len(original_hidden) == 2
    reproduction = {}
    for directory, report, snapshot in zip(
        directories, reports, snapshots, strict=True
    ):
        assert report["request_id"] == f"feedback-trace-native-{sample_id}"
        assert all(
            report[field]
            for field in (
                "native_repeat_exact",
                "alternative_repeats_exact",
                "prefix_and_input_metadata_unchanged",
                "published_native_state",
            )
        )
        native = audit(directory.parent, sample_id)
        assert torch.equal(
            native["thinker_exported_hidden"],
            snapshot["outputs"][0]["hidden_states"][snapshot["audio_rows"]],
        )
        reproduction[directory.parent.name] = [
            difference(
                hidden,
                snapshot["outputs"][index]["hidden_states"][snapshot["audio_rows"]],
            )
            for index, hidden in enumerate(original_hidden, start=1)
        ]
    comparisons = {
        donor: {
            **{
                field: difference(
                    snapshots[0]["outputs"][index][field],
                    snapshots[1]["outputs"][index][field],
                )
                for field in ("hidden_states", "logits")
            },
            "key_value_writes": [
                difference(first, second)
                for first, second in zip(
                    snapshots[0]["writes"][donor],
                    snapshots[1]["writes"][donor],
                    strict=True,
                )
            ],
        }
        for index, donor in enumerate(("first", "second"), start=1)
    }
    result = {
        "sample_id": sample_id,
        "selected_prefix_exact": True,
        "native_handoffs_exact": True,
        "original_hidden_reproduction": reproduction,
        "same_donor_across_cohorts": comparisons,
        "cohorts": [
            {
                "requests": len(report["request_ids"]),
                "tokens": sum(report["extend_lengths"]),
                "mode": report["forward_mode"],
                "cuda_graph": report["cuda_graph"],
                "graph_bucket_tokens": report["graph_bucket_tokens"],
            }
            for report in reports
        ],
        "capture_sha256": {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for directory in directories
            for path in (
                directory / "thinker-input-trial.pt",
                directory / "thinker-input-trial.json",
            )
        },
    }
    with (root / "thinker-cohort-comparison.json").open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sample_id")
    arguments = parser.parse_args()
    main(arguments.root, arguments.sample_id)
else:
    pass
