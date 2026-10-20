"""An experiment switch for the probe: COSY_GREEN_AR_SMS=N puts the AR's loop and the vocoder on one pair of streams
whose kernels run on disjoint SM partitions, N SMs for the AR and the rest for the vocoder (CUDA green contexts made
by sgl_kernel.spatial, as SGLang's PD multiplexing makes them). Unset, nothing changes. The pair is made once per
process; graphs keep their capture streams and replay into these."""

import os
import sys
import threading

AR_SMS = int(os.environ.get("COSY_GREEN_AR_SMS", "0"))
lock = threading.Lock()
pair = None


def streams():
    """The AR's stream and the vocoder's stream."""
    global pair
    with lock:
        if pair is None:
            import torch
            from sgl_kernel import spatial

            device = torch.cuda.current_device()
            total = spatial.get_sm_available(device)
            pair = spatial.create_greenctx_stream_by_value(AR_SMS, total - AR_SMS, device)
            print(f"cosy_green ar {AR_SMS} vocoder {total - AR_SMS} of {total} SMs", file=sys.stderr, flush=True)
        else:
            pass
    return pair
