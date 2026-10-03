import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["base_profiler", "torch_profiler"])
def test_profiler_import_preserves_application_logging(module: str) -> None:
    script = r"""
import importlib
import logging
import sys

if sys.argv[1] == 'torch_profiler':
    import torch.profiler
    import sglang_omni.platforms
    import sglang_omni.profiler.base_profiler
else:
    pass

root = logging.getLogger()
root.handlers.clear()
root.setLevel(logging.ERROR)
importlib.import_module('sglang_omni.profiler.' + sys.argv[1])
assert root.level == logging.ERROR
assert not root.handlers
console = logging.StreamHandler(sys.stdout)
root.addHandler(console)
logging.getLogger('application').warning('hidden warning')
logging.getLogger('application').error('visible error')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, module],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "visible error\n"


def test_hub_routing_preserves_application_destinations(tmp_path: Path) -> None:
    script = r"""
import io
import logging
import sys
from pathlib import Path

from huggingface_hub.utils import logging as hf_logging
from sglang_omni.utils.logging import configure_hf_hub_logger

class ApplicationConsole(logging.StreamHandler):
    pass

root = logging.getLogger()
root.handlers.clear()
root.setLevel(logging.INFO)
root_console = logging.StreamHandler(sys.stdout)
root_console.setFormatter(logging.Formatter('root %(levelname)s %(message)s'))
root.addHandler(root_console)
hub = hf_logging.get_logger()
file_destination = logging.FileHandler(sys.argv[1])
custom_output = io.StringIO()
custom_console = ApplicationConsole(custom_output)
hub.addHandler(file_destination)
hub.addHandler(custom_console)
configure_hf_hub_logger()
configure_hf_hub_logger()
hf_logging.get_logger('huggingface_hub.audit').warning('warning retained')
hf_logging.get_logger('huggingface_hub.audit').error('error retained')
file_destination.flush()
assert Path(sys.argv[1]).read_text() == 'warning retained\nerror retained\n'
assert custom_output.getvalue() == 'warning retained\nerror retained\n'
assert root.handlers == [root_console]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "application.log")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "root WARNING warning retained",
        "root ERROR error retained",
    ]
    assert result.stderr == ""
