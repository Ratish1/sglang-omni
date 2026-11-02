# SPDX-License-Identifier: Apache-2.0
"""Compiled programs kept on disk, so a later boot loads what an earlier one traced."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import platform
from collections.abc import Iterable
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._inductor.runtime.cache_dir_utils import cache_dir

logger = logging.getLogger(__name__)


def source_tree_digest(directories: Iterable[Path]) -> str:
    """A digest of every Python source under the directories."""
    digest = hashlib.sha256()
    for directory in directories:
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.read_bytes())
    return digest.hexdigest()


def has_current_sources(program: AOTCompiledFunction) -> bool:
    """Whether every module this program's trace inlined still reads as it did."""
    return all(
        inspect.getsource(importlib.import_module(source.module)) == source.content
        for source in program.source_info().inlined_sources
    )


class SavedPrograms:
    """The programs of one traced function, under the Inductor cache directory."""

    def __init__(self, name: str, key_parts: Iterable[str]) -> None:
        digest = hashlib.sha256()
        for part in (
            torch.__version__,
            platform.python_version(),
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            str(inductor_config.get_hash()),
            *key_parts,
        ):
            digest.update(part.encode())
        self.directory = Path(cache_dir()) / name / digest.hexdigest()
        self.programs = [
            program
            for program in map(self.load, sorted(self.directory.glob("*.program")))
            if program is not None
        ]

    @staticmethod
    def load(path: Path) -> AOTCompiledFunction | None:
        # note (ratish): a cache must never stop a boot; a program it cannot load is traced again.
        try:
            with path.open("rb") as handle:
                program = torch.compiler.load_compiled_function(handle)
            if has_current_sources(program):
                return program
            else:
                logger.warning(f"Saved program {path} was traced from other sources")
                return None
        except Exception as error:
            logger.warning(f"Could not load the saved program {path}: {error}")
            return None

    def matching(self, *args: object) -> AOTCompiledFunction | None:
        """The first program whose guards pass on these arguments."""
        return next(
            (program for program in self.programs if program.guard_check(*args)), None
        )

    def add(self, program: AOTCompiledFunction, name: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        program.save_compiled_function(str(self.directory / f"{name}.program"))
        self.programs.append(program)
