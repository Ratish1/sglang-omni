"""benchmarks.eval.benchmark_omni_seedtts, with each streamed chunk's audio length and the playback underrun recorded.

usage: python3 omni_seedtts_chunks.py <the benchmark_omni_seedtts arguments>

The Omni chat client keeps each chunk's arrival time but not its length, so its records carry no
chunk_audio_duration_s or max_playback_underrun_s. This wraps the SSE audio parser to note the frames of each chunk it
appends, and the send function to write both fields with the benchmark's own compute_max_playback_underrun_s over the
chunk arrival offsets. Requests, timing and every other field are the benchmark's.
"""

import contextvars
import functools
import itertools
import sys

import benchmarks.eval.benchmark_omni_seedtts as bench
import benchmarks.tasks.tts as tts
from benchmarks.metrics.playback_continuity import compute_max_playback_underrun_s

DURATIONS: contextvars.ContextVar = contextvars.ContextVar(
    "chunk_durations", default=None
)
collect = tts._collect_chat_streaming_audio
make_send_fn = bench.make_send_fn


def collect_with_lengths(line, pcm_chunks, pcm_format, usage, **kwargs):
    before = len(pcm_chunks)
    pcm_format = collect(line, pcm_chunks, pcm_format, usage, **kwargs)
    durations = DURATIONS.get()
    if durations is not None and pcm_format is not None and len(pcm_chunks) > before:
        rate, channels, width = pcm_format
        durations.extend(
            len(chunk) / (rate * channels * width) for chunk in pcm_chunks[before:]
        )
    else:
        pass
    return pcm_format


def make_send_fn_with_lengths(*args, **kwargs):
    send_fn = make_send_fn(*args, **kwargs)

    @functools.wraps(send_fn)
    async def send(session, sample):
        token = DURATIONS.set([])
        try:
            result = await send_fn(session, sample)
            durations = DURATIONS.get()
        finally:
            DURATIONS.reset(token)
        if result.inter_chunk_s is not None and durations:
            arrivals = [0.0, *itertools.accumulate(result.inter_chunk_s)]
            if len(arrivals) == len(durations):
                result.chunk_audio_duration_s = durations
                result.max_playback_underrun_s = compute_max_playback_underrun_s(
                    arrivals, durations
                )
            else:
                pass
        else:
            pass
        return result

    return send


tts._collect_chat_streaming_audio = collect_with_lengths
bench.make_send_fn = make_send_fn_with_lengths

if __name__ == "__main__":
    sys.argv[0] = bench.__file__
    bench.main()
