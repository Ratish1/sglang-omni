"""Which CUDA streams each thread's device work runs on, and the streams more than one thread uses (10-22, L3's static
c16 gap cost). torch hands out non-priority streams round robin from 32 per device and priority
(c10/cuda/CUDAStream.cpp:19-20, 258-260), so two long lived torch.cuda.Stream objects can be one CUDA stream.

For each label=sqlite (an nsys trace): every graph replay, kernel and copy is tied to the CUDA runtime call with its
correlation id, so to that call's thread; per thread (named by its most frequent NVTX range prefix) the streams it ran
work on with counts, then every stream with its threads.
usage: python3 stream_owners_c20261022.py l2=<c16.sqlite> l3=<c16.sqlite>
"""

import sqlite3
import sys
from collections import Counter, defaultdict

for argument in sys.argv[1:]:
    label, path = argument.split("=", 1)
    db = sqlite3.connect(path)
    thread_of = dict(db.execute("select correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"))
    names = defaultdict(Counter)
    for text, thread in db.execute(
        "select coalesce(n.text, s.value), n.globalTid from NVTX_EVENTS n left join StringIds s on s.id = n.textId "
        "where n.end is not null"
    ):
        names[thread][text.split(" ")[0]] += 1
    work = defaultdict(Counter)
    for table, where in (
        ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "1"),
        ("CUPTI_ACTIVITY_KIND_KERNEL", "graphId is null or graphId = 0"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "1"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "1"),
    ):
        for correlation, stream in db.execute(f"select correlationId, streamId from {table} where {where}"):
            thread = thread_of.get(correlation)
            if thread is not None:
                work[thread][stream] += 1
            else:
                pass
    print(f"== {label} {path}")
    owners = defaultdict(list)
    for thread, streams in sorted(work.items(), key=lambda item: -sum(item[1].values())):
        top = ", ".join(name for name, _ in names[thread].most_common(3)) or "no ranges"
        print(f"  thread {thread % 16777216} ({top}): " + ", ".join(f"stream {s} x{n}" for s, n in streams.most_common(6)))
        for stream in streams:
            owners[stream].append(thread % 16777216)
    for stream, threads in sorted(owners.items()):
        flag = "  <- shared" if len(threads) > 1 else ""
        print(f"  stream {stream}: threads {threads}{flag}")
