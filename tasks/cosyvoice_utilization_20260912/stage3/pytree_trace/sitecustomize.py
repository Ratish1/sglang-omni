# SPDX-License-Identifier: Apache-2.0
"""Log every pytree node registration with its time and caller.

Put this directory first on PYTHONPATH of a server; the log goes to stderr.
"""

import time
import traceback

import torch.utils._pytree as pytree

original = pytree._private_register_pytree_node


def logged(cls, *args, **kwargs):
    caller = [
        f"{frame.filename.split('site-packages/')[-1]}:{frame.lineno}"
        for frame in traceback.extract_stack()[-8:-1]
    ]
    print(
        f"PYTREE_REGISTER {time.strftime('%H:%M:%S')} "
        f"{cls.__module__}.{cls.__qualname__} total={len(pytree.SUPPORTED_NODES) + 1} "
        f"from {' <- '.join(reversed(caller))}",
        flush=True,
    )
    return original(cls, *args, **kwargs)


pytree._private_register_pytree_node = logged
