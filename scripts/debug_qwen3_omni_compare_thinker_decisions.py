"""Compare complete native text trajectories only at identical token prefixes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class CapturedStep(BaseModel):
    phase: str
    captured_graph: bool
    request_ids: list[str]
    output_prefixes: list[list[int]]
    remaining_prefill_chunks: list[int]
    prompt_sha256: list[str]
    sequence_lengths: list[int]
    token_ids: list[int]
    top_ids: list[list[int]]
    top_logits: list[list[float]]
    hidden_sha256: list[str]


@dataclass(kw_only=True)
class Decision:
    step: CapturedStep
    row: int


def read_decisions(directory: Path) -> dict[str, list[Decision]]:
    requests: dict[str, list[Decision]] = {}
    with (directory / "steps.jsonl").open() as manifest:
        for line in manifest:
            step = CapturedStep.model_validate_json(line)
            for row, request_id in enumerate(step.request_ids):
                if step.remaining_prefill_chunks[row] > 0:
                    continue
                else:
                    pass
                history = requests.setdefault(request_id, [])
                prefix = step.output_prefixes[row]
                assert len(prefix) == len(history), request_id
                if history:
                    previous = history[-1]
                    assert prefix == previous.step.output_prefixes[previous.row] + [
                        previous.step.token_ids[previous.row]
                    ]
                else:
                    assert not prefix
                history.append(Decision(step=step, row=row))
    expected = json.loads((directory / "request_ids.json").read_text())
    assert set(requests) == set(expected)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-directory", type=Path, required=True)
    parser.add_argument("--second-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    first_requests = read_decisions(arguments.first_directory)
    second_requests = read_decisions(arguments.second_directory)
    assert set(first_requests) == set(second_requests)
    arguments.output_directory.mkdir(exist_ok=False)
    totals: Counter[str] = Counter()
    with (arguments.output_directory / "requests.jsonl").open("w") as output:
        for request_id, first_history in first_requests.items():
            second_history = second_requests[request_id]
            counts: Counter[str] = Counter()
            differences: list[dict[str, int | bool | list[int] | list[float]]] = []
            for first, second in zip(first_history, second_history):
                first_step, second_step = first.step, second.step
                first_row, second_row = first.row, second.row
                if (
                    first_step.output_prefixes[first_row]
                    != second_step.output_prefixes[second_row]
                ):
                    counts["different_prefix_rows"] += 1
                    continue
                else:
                    pass
                counts["identical_prefix_rows"] += 1
                assert (
                    first_step.prompt_sha256[first_row]
                    == second_step.prompt_sha256[second_row]
                ), request_id
                assert (
                    first_step.sequence_lengths[first_row]
                    == second_step.sequence_lengths[second_row]
                ), request_id
                counts["changed_head_inputs"] += (
                    first_step.hidden_sha256[first_row]
                    != second_step.hidden_sha256[second_row]
                )
                counts["changed_top_logits"] += (
                    first_step.top_ids[first_row] != second_step.top_ids[second_row]
                    or first_step.top_logits[first_row]
                    != second_step.top_logits[second_row]
                )
                first_token = first_step.token_ids[first_row]
                second_token = second_step.token_ids[second_row]
                if first_token != second_token:
                    counts["changed_choices_at_identical_prefix"] += 1
                    differences.append(
                        {
                            "output_position": len(
                                first_step.output_prefixes[first_row]
                            ),
                            "first_token_id": first_token,
                            "second_token_id": second_token,
                            "first_top_ids": first_step.top_ids[first_row],
                            "second_top_ids": second_step.top_ids[second_row],
                            "first_top_logits": first_step.top_logits[first_row],
                            "second_top_logits": second_step.top_logits[second_row],
                            "first_top_tied": first_step.top_logits[first_row][0]
                            == first_step.top_logits[first_row][1],
                            "second_top_tied": second_step.top_logits[second_row][0]
                            == second_step.top_logits[second_row][1],
                            "first_batch_rows": len(first_step.request_ids),
                            "second_batch_rows": len(second_step.request_ids),
                            "first_captured_graph": first_step.captured_graph,
                            "second_captured_graph": second_step.captured_graph,
                        }
                    )
                else:
                    pass
            counts["first_committed_tokens"] = len(first_history)
            counts["second_committed_tokens"] = len(second_history)
            counts["unpaired_tail_rows"] = abs(len(first_history) - len(second_history))
            counts["requests"] = 1
            counts["identical_complete_token_sequences"] = not differences
            assert bool(differences) or len(first_history) == len(second_history)
            totals.update(counts)
            output.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "counts": dict(counts),
                        "choice_differences": differences,
                    }
                )
                + "\n"
            )
    report = {
        "counts": dict(totals),
        "interpretation": "Rows after a prefix fork are counted, not treated as equal-input arithmetic comparisons.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
