"""Build an isolated native expert contribution-rounding diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from flashinfer.jit import env as jit_environment
from flashinfer.jit.cpp_ext import run_ninja

from sglang_omni.models.qwen3_omni.contribution_rounding import (
    FINALIZER_PATH,
    contribution_build_spec,
)

FINALIZER_SHA256 = "7566fe479fc7f34f3e92e1be6941f34d9c343b8e8df996e4b3b3fd960999c604"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--build-jobs", type=int, required=True)
    parser.add_argument(
        "--output-arithmetic",
        choices=("rounded-contributions", "checkpoint-reduction"),
        default="rounded-contributions",
    )
    arguments = parser.parse_args()
    if arguments.build_jobs < 1:
        raise ValueError("Build jobs must be positive")
    else:
        pass
    arguments.output_directory.mkdir(exist_ok=False)
    original_finalizer = jit_environment.FLASHINFER_CSRC_DIR / FINALIZER_PATH
    assert (
        hashlib.sha256(original_finalizer.read_bytes()).hexdigest() == FINALIZER_SHA256
    )
    source_directory = arguments.output_directory.resolve() / "csrc"
    shutil.copytree(jit_environment.FLASHINFER_CSRC_DIR, source_directory)
    patch_filename = (
        "qwen3_omni_checkpoint_reduction.patch"
        if arguments.output_arithmetic == "checkpoint-reduction"
        else "qwen3_omni_bf16_contributions.patch"
    )
    subprocess.run(
        [
            "patch",
            "--batch",
            "--fuzz=0",
            "-p1",
            "-d",
            str(source_directory),
            "-i",
            str(Path(__file__).with_name(patch_filename)),
        ],
        check=True,
    )
    specification = contribution_build_spec(source_directory)
    assert not specification.is_aot
    (arguments.output_directory / "build-specification.json").write_text(
        json.dumps(asdict(specification), default=str, indent=2)
    )
    specification.write_ninja()
    run_ninja(
        specification.build_dir,
        specification.ninja_path,
        verbose=True,
        max_jobs=arguments.build_jobs,
    )
    assert (
        hashlib.sha256(original_finalizer.read_bytes()).hexdigest() == FINALIZER_SHA256
    )
    (arguments.output_directory / "build-result.json").write_text(
        json.dumps(
            {
                "library": str(specification.jit_library_path),
                "library_sha256": hashlib.sha256(
                    specification.jit_library_path.read_bytes()
                ).hexdigest(),
                "original_finalizer_sha256": FINALIZER_SHA256,
                "patched_finalizer_sha256": hashlib.sha256(
                    (source_directory / FINALIZER_PATH).read_bytes()
                ).hexdigest(),
                "output_arithmetic": arguments.output_arithmetic,
                "scope": "Separate diagnostic module with the explicitly selected output arithmetic. Original installed sources and AOT module are unchanged. Numerical equivalence is not established by compilation.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
