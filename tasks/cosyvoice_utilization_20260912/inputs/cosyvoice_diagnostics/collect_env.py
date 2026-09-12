#!/usr/bin/env python3
"""Read-only serving-environment snapshot; Python 3.10+, standard library only.
Run with the SAME Python executable and working directory as the server.
Source defaults are not the live resolved configuration. No model is loaded.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ENV_KEYS = (
    'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'CUDA_MODULE_LOADING',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TORCH_LOGS', 'NVIDIA_TF32_OVERRIDE',
    'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PIPE_DIRECTORY',
    'CUDA_MPS_LOG_DIRECTORY', 'CUDA_DEVICE_MAX_CONNECTIONS',
    'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
)


def command(argv: list[str], cwd: Path | None = None) -> dict:
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                           timeout=20, check=False)
        return {'argv': argv, 'returncode': p.returncode,
                'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': str(exc)}


def source_info(path: Path) -> dict:
    if not path.is_file():
        return {'path': str(path), 'error': 'not found in this installation'}
    raw = path.read_bytes()
    result = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    try:
        tree = ast.parse(raw.decode('utf-8'))
        factories = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('create_'):
                factories[node.name] = ast.unparse(node.args)
        result['factory_source_signatures'] = factories
    except (UnicodeError, SyntaxError) as exc:
        result['parse_error'] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('environment.json'))
    args = parser.parse_args()
    data = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'cwd': str(Path.cwd()),
        'environment_allowlist': {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        'executables': {k: shutil.which(k) for k in ('sgl-omni', 'nsys', 'ncu', 'dcgmi', 'nvidia-smi')},
        'packages': sorted(
            [{'name': d.metadata.get('Name', '<unknown>'), 'version': d.version}
             for d in importlib.metadata.distributions()], key=lambda x: x['name'].lower()),
        'git_cwd': command(['git', 'rev-parse', 'HEAD']),
        'git_status_cwd': command(['git', 'status', '--short']),
        'git_submodules_cwd': command(['git', 'submodule', 'status', '--recursive']),
        'gpu_list': command(['nvidia-smi', '-L']),
        'gpu_query': command(['nvidia-smi', '-q']),
        'gpu_topology': command(['nvidia-smi', 'topo', '-m']),
        'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory', '--format=csv']),
        'cpu_topology': command(['lscpu']),
        'note': 'Factory signatures below are SOURCE defaults, not the live resolved configuration. Attach the effective launch arguments and startup configuration separately; redact secrets.'
    }
    if hasattr(os, 'sched_getaffinity'):
        data['collector_cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for filename in ('/proc/self/cgroup', '/sys/fs/cgroup/cpu.max',
                     '/sys/fs/cgroup/cpu.stat', '/sys/fs/cgroup/cpuset.cpus.effective',
                     '/sys/fs/cgroup/memory.max'):
        path = Path(filename)
        if path.is_file():
            try:
                data[filename] = path.read_text()
            except OSError as exc:
                data[filename] = str(exc)
    try:
        spec = importlib.util.find_spec('sglang_omni')
        if spec is None or not spec.submodule_search_locations:
            data['sglang_omni_source'] = {'error': 'package not found; use the server Python environment'}
        else:
            root = Path(next(iter(spec.submodule_search_locations)))
            data['sglang_omni_package_root'] = str(root.resolve())
            data['git_package_root'] = command(['git', 'rev-parse', 'HEAD'], cwd=root)
            model_root = root / 'models' / 'fun_cosyvoice3'
            data['model_sources'] = [source_info(model_root / f) for f in (
                'config.py', 'stages.py', 'streaming_vocoder.py',
                'model_runner.py', 'request_builders.py', 'flow_estimator_trt.py')]
    except (ImportError, OSError, ValueError) as exc:
        data['sglang_omni_source'] = {'error': str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
