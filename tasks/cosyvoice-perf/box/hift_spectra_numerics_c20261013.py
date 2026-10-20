# Is D's difference from main HiFT's own sensitivity or an error in D? The checkpoint's HiFT (the lightweight
# loader, constants on the device) as main runs it, with D's patch_hift_spectra (box/c1_patch/c1_impl.py), and in
# float64 as the reference, on the same mels, with cuDNN TF32 on (the serving default) and off: the largest difference
# and the SNR against float64 of each float32 variant, and between them.
# usage: python3 hift_spectra_numerics_c20261013.py
import copy
import glob
import math
import sys

import torch

sys.path.insert(0, "/data/c20261013/in/c1_patch")
from c1_impl import patch_hift_spectra  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import stages  # noqa: E402

device = torch.device("cuda")
checkpoint = glob.glob("/root/.cache/huggingface/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/*")[0]
_, hift = stages.load_cosyvoice3_flow_hift_lightweight(checkpoint, device="cuda")
stages.keep_hift_constants_on_device(hift, "cuda")
stages.patch_causal_conv_cache()
patched = copy.deepcopy(hift)
patch_hift_spectra(patched, device)
reference = copy.deepcopy(hift).double()
reference.stft_window = reference.stft_window.double()
sine_generator = reference.m_source.l_sin_gen
sine_generator.sine_waves = sine_generator.sine_waves.double()
sine_generator.rand_ini = sine_generator.rand_ini.double()

torch.manual_seed(0)
mel = torch.randn(4, 80, 400, device=device) * 2 - 4


def snr_db(signal, error):
    return 10 * math.log10(signal.pow(2).mean().item() / max(error.pow(2).mean().item(), 1e-30))


with torch.inference_mode():
    truth, _ = reference.inference(speech_feat=mel.double(), finalize=True)
    print(f"reference: shape {tuple(truth.shape)}, rms {truth.pow(2).mean().sqrt().item():.4f}, max {truth.abs().max().item():.4f}")
    for tf32 in (True, False):
        torch.backends.cudnn.allow_tf32 = tf32
        main, _ = hift.inference(speech_feat=mel, finalize=True)
        spectra, _ = patched.inference(speech_feat=mel, finalize=True)
        main64 = main.double()
        spectra64 = spectra.double()
        print(
            f"cudnn tf32 {tf32}: main against float64 max {(main64 - truth).abs().max().item():.3e} "
            f"snr {snr_db(truth, main64 - truth):.1f} dB | D against float64 max {(spectra64 - truth).abs().max().item():.3e} "
            f"snr {snr_db(truth, spectra64 - truth):.1f} dB | D against main max {(spectra64 - main64).abs().max().item():.3e} "
            f"snr {snr_db(main64, spectra64 - main64):.1f} dB"
        )
