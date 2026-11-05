"""Makes one replay of one incremental Codec runner raise, in every process of a real server, because stages are spawned. On the
python path only for graph_fault_read.py. QTTS_GRAPH_FAULT=mode:replay names the runner and which of its replays fails, once.
"""

import os

FAULT = os.environ.get("QTTS_GRAPH_FAULT")
if FAULT:
    from sglang_omni.models.qwen3_tts import incremental_codec_cuda_graph as graphs

    fault_mode, fault_replay = FAULT.split(":")
    replay_count = [0]
    real_capture = graphs.Qwen3TTSIncrementalCodecCudaGraphRunner.capture

    class FaultingGraph:
        def __init__(self, graph) -> None:
            self.graph = graph

        def replay(self) -> None:
            replay_count[0] += 1
            if replay_count[0] == int(fault_replay):
                raise RuntimeError("injected replay failure")
            else:
                self.graph.replay()

        def reset(self) -> None:
            self.graph.reset()

    def capture(self) -> None:
        real_capture(self)
        if self.mode == fault_mode:
            for entry in self.graphs.values():
                entry.graph = FaultingGraph(entry.graph)
        else:
            pass

    graphs.Qwen3TTSIncrementalCodecCudaGraphRunner.capture = capture
else:
    pass
