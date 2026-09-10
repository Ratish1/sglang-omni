"""Untimed qualification entry point; also installed in spawned stage processes."""

import os
import threading
from pathlib import Path

if os.environ.get("MINIMAX_MUSIC3_QUALIFICATION_CAPTURE"):
    import torch

    from sglang_omni.models.minimax_music3.acoustic import MiniMaxMusic3AcousticDecoder

    _original_decode = MiniMaxMusic3AcousticDecoder.decode_with_state
    _active = threading.local()

    def _capture(_module, args, kwargs, output):
        directory, stem = _active.destination
        tensors = {"condition": args[0], "latent": output}
        for name, tensor in tensors.items():
            path = directory / f"{stem}_{name}.pt"
            with path.open("xb") as stream:
                torch.save(tensor.detach().cpu(), stream)

    def _decode(self, hidden, *, seed, chunk_idx, **kwargs):
        directory = Path(os.environ["MINIMAX_MUSIC3_QUALIFICATION_CAPTURE"])
        directory.mkdir(parents=True, exist_ok=True)
        _active.destination = (directory, f"seed{int(seed)}_chunk{chunk_idx:03d}")
        hook = self.dit.register_forward_hook(_capture, with_kwargs=True)
        try:
            return _original_decode(
                self, hidden, seed=seed, chunk_idx=chunk_idx, **kwargs
            )
        finally:
            hook.remove()
            del _active.destination

    MiniMaxMusic3AcousticDecoder.decode_with_state = _decode


if __name__ == "__main__":
    from sglang_omni.cli import app

    app()
