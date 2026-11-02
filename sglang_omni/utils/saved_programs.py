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


class SavedPrograms:
    """The programs of one traced function, under the Inductor cache directory, keyed on what
    a program's guards do not see: versions, the GPU, the kernels' sources and the caller's parts.
    """

    def __init__(
        self, name: str, kernel_sources: Iterable[Path], key_parts: Iterable[str]
    ) -> None:
        digest = hashlib.sha256()
        for part in (
            torch.__version__,
            platform.python_version(),
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            str(inductor_config.get_hash()),
            *key_parts,
        ):
            digest.update(part.encode())
        for directory in kernel_sources:
            for path in sorted(directory.rglob("*.py")):
                digest.update(path.read_bytes())
        self.directory = Path(cache_dir()) / name / digest.hexdigest()
        self.programs = [
            program
            for program in map(self.load, sorted(self.directory.glob("*.program")))
            if program is not None
        ]
        self.by_shape: dict[tuple[int, ...], AOTCompiledFunction] = {}

    def __getstate__(self) -> dict[str, None]:
        # note (ratish): a trace pickles what its function's self reaches, and a program's guard
        # manager cannot be pickled; no guard reads these programs.
        return {}

    @staticmethod
    def load(path: Path) -> AOTCompiledFunction | None:
        """The program at this path, unless it cannot load or a module its trace inlined changed."""
        # note (ratish): a cache must never stop a boot; a program it cannot load is traced again.
        try:
            with path.open("rb") as handle:
                program = torch.compiler.load_compiled_function(handle)
            for source in program.source_info().inlined_sources:
                module = importlib.import_module(source.module)
                if inspect.getsource(module) != source.content:
                    logger.warning(f"{path} was traced from another {source.module}")
                    return None
                else:
                    pass
            return program
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
