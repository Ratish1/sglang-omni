"""Compare the warmed ONNX arena with eight admitted reference encodes."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.fun_cosyvoice3.utils import MAX_REFERENCE_SAMPLES, SpeechTokenizerV3
from sglang_omni.utils.checkpoint import resolve_checkpoint


def native_bytes():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


def main():
    torch.set_num_threads(1)
    root = Path(resolve_checkpoint("FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    tokenizer = SpeechTokenizerV3(str(root / "speech_tokenizer_v3.onnx"), device="cuda:0", intra_op_threads=16)
    audio = np.zeros((1, MAX_REFERENCE_SAMPLES), dtype=np.float32)
    tokenizer.extract_speech_token(audio)
    warmed = native_bytes()
    barrier = threading.Barrier(8)

    def encode(index):
        barrier.wait()
        return tokenizer.extract_speech_token(audio.copy()).shape[-1]

    with ThreadPoolExecutor(max_workers=8) as executor:
        lengths = list(executor.map(encode, range(8)))
    after = native_bytes()
    print(json.dumps(dict(warmed_native_bytes=warmed, concurrent_native_bytes=after,
                         retained_growth_bytes=after-warmed, output_lengths=lengths)), flush=True)


if __name__ == "__main__":
    main()
