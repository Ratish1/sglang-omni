import asyncio
import itertools
import json
import time

from sglang_omni.client.client import Client, StreamedStopTrimmer
from sglang_omni.client.types import GenerateChunk, GenerateRequest, SamplingParams


def partitions(text):
    for cuts in itertools.product((False, True), repeat=max(len(text) - 1, 0)):
        start = 0
        chunks = []
        for index, cut in enumerate(cuts, 1):
            if cut:
                chunks.append(text[start:index])
                start = index
        yield chunks + [text[start:]]


checked = 0
for stop in ("a", "aa", "aba", "abab", "baab", "😀a", "\n\n"):
    texts = ["", "😀a😀", "x\n\ny", "x\n"]
    texts += ["".join(row) for n in range(1, 7) for row in itertools.product("ab", repeat=n)]
    for text in texts:
        matched = text.find(stop)
        expected = text if matched < 0 else text[:matched]
        for chunks in partitions(text):
            trimmer = StreamedStopTrimmer(stop=[stop])
            actual = "".join(trimmer.push(chunk) for chunk in chunks) + trimmer.finish()
            assert actual == expected, (stop, chunks, actual, expected)
            checked += 1


class CumulativeClient(Client):
    def __init__(self):
        self.closed = False

    async def generate(self, request, request_id=None):
        try:
            for text in ("answer ", "<ST"):
                yield GenerateChunk(request_id=request_id, text=text)
            yield GenerateChunk(request_id=request_id, modality="audio", finish_reason="stop")
            yield GenerateChunk(request_id=request_id, text="answer <STOP> tail", finish_reason="stop")
        finally:
            self.closed = True


async def cumulative_final():
    client = CumulativeClient()
    request = GenerateRequest(prompt="x", sampling=SamplingParams(stop=["<STOP>"]))
    chunks = [chunk async for chunk in client.completion_stream(request, request_id="check")]
    assert "".join(chunk.text or "" for chunk in chunks) == "answer "
    assert client.closed
    return {"text": "answer ", "audio_completions": sum(chunk.modality == "audio" for chunk in chunks)}


class CountedStop(str):
    slices = 0
    def __getitem__(self, key):
        type(self).slices += 1
        return super().__getitem__(key)


long_stop = CountedStop("x" * 500_000)
trimmer = StreamedStopTrimmer(stop=[long_stop])
start = time.perf_counter()
assert trimmer.push("hello") == "hello"
duration = time.perf_counter() - start
assert CountedStop.slices <= 5, CountedStop.slices
print(json.dumps({"partition_cases": checked, "cumulative": asyncio.run(cumulative_final()), "long_stop_prefix_slices": CountedStop.slices, "long_stop_seconds": duration}, indent=2))
