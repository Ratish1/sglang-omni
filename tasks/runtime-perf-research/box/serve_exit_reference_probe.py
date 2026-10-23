"""Run the fast-exit test's sender under one change and apply the test's assertions.

usage (cwd and PYTHONPATH: the sglang-omni tree under test):
  python3 serve_exit_reference_probe.py <normal|stale|overlap|missing-cleanup>

normal           the sender as the test runs it
stale            a file named after the sender's pid exists in /dev/shm before it sends
overlap          such a file appears while the tensor is being reduced for the send
missing-cleanup  the fast exit skips release_cuda_ipc_file
Prints the sender's report, the foreign file and whether the test's assertions hold.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.unit_test.cli.test_serve_exit import SEND_CUDA_TENSOR

mode = sys.argv[1]
FOREIGN = '"/dev/shm/torch_" + str(os.getpid()) + "_3900310031_999999990"'
source = SEND_CUDA_TENSOR
if mode == "stale":
    source = source.replace(
        'if __name__ == "__main__":\n',
        'if __name__ == "__main__":\n    open(' + FOREIGN + ', "x").close()\n',
    )
elif mode == "overlap" and "    return rebuild, arguments\n" in source:
    source = source.replace(
        "    return rebuild, arguments\n",
        "    open(" + FOREIGN + ', "x").close()\n    return rebuild, arguments\n',
    )
elif mode == "overlap":
    source = source.replace(
        "    files_before_sending = set(glob.glob(pattern))\n",
        "    files_before_sending = set(glob.glob(pattern))\n"
        "    open(" + FOREIGN + ', "x").close()\n',
    )
elif mode == "missing-cleanup":
    source = source.replace(
        'if __name__ == "__main__":\n',
        'if __name__ == "__main__":\n'
        "    import sglang_omni.utils.process as exit_helpers\n"
        "    exit_helpers.release_cuda_ipc_file = lambda: None\n",
    )
else:
    pass
assert mode == "normal" or source != SEND_CUDA_TENSOR, mode
with tempfile.TemporaryDirectory(prefix="serve-exit-probe-") as directory:
    script = Path(directory) / "send_cuda_tensor.py"
    script.write_text(source)
    sender = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    report = json.loads(sender.stdout) if sender.returncode == 0 else None
    foreign = None
    if mode in ("stale", "overlap"):
        candidates = sorted(Path("/dev/shm").glob("torch_*_3900310031_999999990"))
        foreign = [str(path) for path in candidates]
    else:
        pass
    remaining = (
        [] if report is None else [path for path in report if Path(path).exists()]
    )
    if isinstance(report, dict):
        test_assertions_hold = list(report.values()) == [True] and remaining == []
    elif isinstance(report, list):
        test_assertions_hold = bool(report) and remaining == []
    else:
        test_assertions_hold = False
    print(
        json.dumps(
            {
                "mode": mode,
                "returncode": sender.returncode,
                "report": report,
                "foreign_files": foreign,
                "foreign_in_report": bool(
                    report and foreign and set(foreign) & set(report)
                ),
                "remaining_after_exit": remaining,
                "test_assertions_hold": test_assertions_hold,
                "stderr_tail": sender.stderr[-300:] if sender.returncode else "",
            }
        ),
        flush=True,
    )
    for path in foreign or []:
        Path(path).unlink(missing_ok=True)
    if mode == "missing-cleanup":
        for path in remaining:
            Path(path).unlink(missing_ok=True)
    else:
        pass
