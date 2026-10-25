"""Review-only trained front-end comparison; run inside the task CUDA container."""

import copy
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES, move_conv_padding_to_host
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import Qwen3TTSSpeakerEncoderCudaGraphRunner, SPEAKER_MEL_PADDING, padded_log_mel, reflect_padded_samples

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder


def load_base(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timed(compute):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = compute()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000


def main():
    torch.manual_seed(3177)
    torch.set_num_threads(1)
    root = Path('/data/review3177oct25')
    checkpoint = snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', allow_patterns=['config.json', 'model.safetensors', 'speech_tokenizer/*'])
    config = Qwen3TTSConfig.from_pretrained(checkpoint)
    speaker = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config).eval()
    with safe_open(str(Path(checkpoint)/'model.safetensors'), framework='pt', device='cpu') as handle:
        weights = {name.removeprefix('speaker_encoder.'): handle.get_tensor(name) for name in handle.keys() if name.startswith('speaker_encoder.')}
    speaker.load_state_dict(weights, strict=True)
    del weights
    reference_speaker = copy.deepcopy(speaker).cuda().float().eval()
    speaker = speaker.cuda().bfloat16().eval()
    old_speaker = load_base('review_base_speaker', 'base_speaker.py').Qwen3TTSSpeakerEncoderCudaGraphRunner(speaker, sample_rate=24000)
    new_speaker = Qwen3TTSSpeakerEncoderCudaGraphRunner(speaker, sample_rate=24000)
    buckets = DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(Path(checkpoint)/'speech_tokenizer'), device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    move_conv_padding_to_host(tokenizer.model.encoder)
    reference_codec = copy.deepcopy(tokenizer.model.encoder).float().eval()
    move_conv_padding_to_host(reference_codec)
    old_speaker.capture(buckets, tokenizer.model.encode_downsample_rate)
    new_speaker.capture(buckets, tokenizer.model.encode_downsample_rate)
    old_batcher = load_base('review_base_builders', 'base_builders.py').Qwen3TTSRefCodeBatcher(tokenizer, max_batch_wait_ms=0, graph_bucket_frames=buckets)
    new_batcher = Qwen3TTSRefCodeBatcher(tokenizer, max_batch_wait_ms=0, graph_bucket_frames=buckets)
    waveform, sample_rate = sf.read('/data/review3132oct25/reference.wav', dtype='float32')
    assert sample_rate == 24000 and waveform.ndim == 1
    tiled = np.concatenate((waveform, np.zeros(6000, dtype=np.float32)))
    cases = [len(waveform), 491519, 491520, 491521, 491775, 491776, 983039, 983041, 2112001]
    results = []
    try:
        with torch.inference_mode():
            for count in cases:
                clip = np.resize(tiled, count).astype(np.float32)
                cpu_mel = old_speaker.mel(torch.from_numpy(clip).unsqueeze(0))
                frames = cpu_mel.shape[-1]
                old_embed, old_ms = timed(lambda: old_speaker.embed(clip))
                new_embed, new_ms = timed(lambda: new_speaker.embed(clip))
                fp32_embed = reference_speaker(cpu_mel.cuda().transpose(1,2))[0]
                row = {'samples': count, 'mel_frames': frames, 'speaker_cold_ms': [old_ms, new_ms], 'speaker_exact': torch.equal(old_embed,new_embed), 'speaker_max_abs': float((old_embed-new_embed).abs().max()), 'speaker_cosine_to_fp32': [float(F.cosine_similarity(value.float(),fp32_embed.float(),dim=0)) for value in (old_embed,new_embed)]}
                if frames > max(new_speaker.graphs):
                    width = ((frames+max(new_speaker.graphs)-1)//max(new_speaker.graphs))*max(new_speaker.graphs)
                    reflected=F.pad(torch.from_numpy(clip).cuda().unsqueeze(0),(SPEAKER_MEL_PADDING,SPEAKER_MEL_PADDING),mode='reflect')
                    gpu_mel=padded_log_mel(F.pad(reflected,(0,reflect_padded_samples(width)-reflected.shape[-1])),new_speaker.device_mel_window,new_speaker.device_mel_basis)
                    assert gpu_mel.shape[-1] == width
                    row['mel_max_abs']=float((gpu_mel[...,:frames]-cpu_mel.cuda()).abs().max())
                    row['padded_mel_frames']=width
                old_codes, old_ms = timed(lambda: old_batcher.encode(clip,24000))
                new_codes, new_ms = timed(lambda: new_batcher.encode(clip,24000))
                codec_frames=(count+old_batcher.hop-1)//old_batcher.hop
                assert old_codes.shape == new_codes.shape == (codec_frames,old_batcher.num_quantizers)
                samples=torch.zeros(1,1,codec_frames*old_batcher.hop,device='cuda')
                samples[0,0,:count].copy_(torch.from_numpy(clip))
                fp32_codes=reference_codec.encode(samples,num_quantizers=old_batcher.num_quantizers,return_dict=True).audio_codes[0].transpose(0,1)
                row.update(codec_frames=codec_frames,codec_cold_ms=[old_ms,new_ms],codec_exact=torch.equal(old_codes,new_codes),codec_token_agreement=float((old_codes==new_codes).float().mean()),codec_frame_agreement_to_fp32=[float((codes==fp32_codes).all(1).float().mean()) for codes in (old_codes,new_codes)],codec_unique=int(new_codes.unique().numel()))
                times={name:[] for name in ['speaker_old','speaker_new','codec_old','codec_new']}
                for round_number in range(6):
                    operations=[('speaker_old',lambda:old_speaker.embed(clip)),('speaker_new',lambda:new_speaker.embed(clip)),('codec_old',lambda:old_batcher.encode(clip,24000)),('codec_new',lambda:new_batcher.encode(clip,24000))]
                    for name,compute in operations if round_number%2==0 else reversed(operations):
                        _, elapsed=timed(compute);times[name].append(elapsed)
                row['warm_samples_ms']=times
                row['warm_median_ms']={name:statistics.median(values) for name,values in times.items()}
                results.append(row)
                print(json.dumps(row),flush=True)
                (root/'trained.json').write_text(json.dumps({'checkpoint':checkpoint,'cases':results},indent=2))
    finally:
        old_batcher.close();new_batcher.close()
    assert not old_batcher.thread.is_alive() and not new_batcher.thread.is_alive()
    print('TRAINED CHECKS COMPLETE',flush=True)


if __name__ == '__main__':
    main()
