"""Seed-VC singing conversion with a *voice profile*: several references blended into one voice.

Runs inside Seed-VC's own venv (``services.voice_cover`` starts it as a
subprocess with the Seed-VC checkout as the working directory), so it can
import Seed-VC's ``inference`` module and reuse its model loading.

Seed-VC conditions the diffusion on two things taken from the reference:
a 192-d CAMPPlus timbre vector (``style``) and the reference's own mel +
semantic features used as an in-context prompt. A profile blends both:
the style vector is the weighted mean of the references' vectors, and
the prompt audio is a concatenation of a slice of every reference, each
slice as long as its weight (25 s in all, Seed-VC's prompt limit). The
profile is saved as an ``.npz`` so every cover of a character uses the
exact same voice.

    runner.py profile --output voice.npz --reference a.wav --weight 0.6 --reference b.wav --weight 0.4
    runner.py convert --profile voice.npz --source vocals.wav --output out.wav [--steps 30 ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROMPT_SECONDS = 25.0
MIN_SLICE_SECONDS = 3.0


class Args:
    """What ``inference.load_models`` reads."""

    def __init__(self, f0_condition: bool, fp16: bool = True):
        self.f0_condition = f0_condition
        self.checkpoint = None
        self.config = None
        self.fp16 = fp16


def _seedvc():
    # Seed-VC's script lives in the working directory (the checkout), which a script run by path does not
    # put on sys.path.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    import inference  # noqa: F401
    return inference


def _load(f0_condition: bool, fp16: bool = True):
    inference = _seedvc()
    return inference, inference.load_models(Args(f0_condition, fp16))


def _slices(paths: list[str], weights: list[float], sr: int) -> np.ndarray:
    """Concatenate a slice of every reference, each as long as its weight, 25 s in all."""
    import librosa

    total = float(sum(weights)) or 1.0
    pieces = []
    for path, weight in zip(paths, weights):
        audio = librosa.load(path, sr=sr)[0]
        seconds = max(MIN_SLICE_SECONDS, PROMPT_SECONDS * weight / total)
        take = int(min(len(audio), seconds * sr))
        # Take the middle of the reference: intros and outros are the quietest parts.
        start = max(0, (len(audio) - take) // 2)
        pieces.append(audio[start:start + take].astype(np.float32))
    joined = np.concatenate(pieces) if pieces else np.zeros(sr, dtype=np.float32)
    return joined[: int(PROMPT_SECONDS * sr)]


def build_profile(paths: list[str], weights: list[float], output: str, *, f0_condition: bool = True) -> dict:
    import torch
    import torchaudio

    inference, (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args) = _load(f0_condition)
    device = inference.device
    sr = mel_fn_args["sampling_rate"]
    vectors = []
    for path in paths:
        import librosa
        audio = librosa.load(path, sr=sr)[0][: int(PROMPT_SECONDS * sr)]
        wave = torch.tensor(audio).unsqueeze(0).float().to(device)
        wave_16k = torchaudio.functional.resample(wave, sr, 16000)
        feat = torchaudio.compliance.kaldi.fbank(wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            vectors.append(campplus_model(feat.unsqueeze(0)).squeeze(0).float().cpu().numpy())
    total = float(sum(weights)) or 1.0
    style = sum(v * (w / total) for v, w in zip(vectors, weights))
    prompt = _slices(paths, weights, sr)
    np.savez(output, style=style.astype(np.float32), prompt=prompt, sr=np.int64(sr),
             weights=np.asarray(weights, dtype=np.float32), f0_condition=np.bool_(f0_condition))
    return {"references": len(paths), "prompt_seconds": round(len(prompt) / sr, 1), "sr": sr}


def convert(profile_path: str, source: str, output: str, *, steps: int, cfg_rate: float, length_adjust: float,
            semitone_shift: int, auto_f0: bool) -> float:
    """Seed-VC's ``main`` with the profile in place of the reference file; returns the RTF."""
    import torch

    # Seed-VC's main runs under no_grad: the vocoder refuses inference-mode tensors otherwise.
    with torch.no_grad():
        return _convert(profile_path, source, output, steps=steps, cfg_rate=cfg_rate, length_adjust=length_adjust,
                        semitone_shift=semitone_shift, auto_f0=auto_f0)


def _convert(profile_path: str, source: str, output: str, *, steps: int, cfg_rate: float, length_adjust: float,
             semitone_shift: int, auto_f0: bool) -> float:
    import librosa
    import torch
    import torchaudio

    profile = np.load(profile_path)
    f0_condition = bool(profile["f0_condition"])
    inference, (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args) = _load(f0_condition)
    device = inference.device
    fp16 = inference.fp16
    sr = int(profile["sr"])
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(librosa.load(source, sr=sr)[0]).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(profile["prompt"]).unsqueeze(0).float().to(device)
    style2 = torch.tensor(profile["style"]).unsqueeze(0).float().to(device)
    started = time.time()

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        parts, buffer, traversed = [], None, 0
        while traversed < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed:traversed + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed:traversed + 16000 * (30 - overlapping_time)]],
                                  dim=-1)
            part = semantic_fn(chunk)
            parts.append(part if traversed == 0 else part[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(parts, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)
    mel = mel_fn(source_audio)
    mel2 = mel_fn(ref_audio)
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

    if f0_condition:
        F0_ori = torch.from_numpy(f0_fn(ori_waves_16k[0], thred=0.03)).to(device)[None]
        F0_alt = torch.from_numpy(f0_fn(converted_waves_16k[0], thred=0.03)).to(device)[None]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0:
            voiced_ori, voiced_alt = F0_ori[F0_ori > 1], F0_alt[F0_alt > 1]
            if voiced_ori.numel() and voiced_alt.numel():
                median_ori = torch.median(torch.log(voiced_ori + 1e-5))
                median_alt = torch.median(torch.log(voiced_alt + 1e-5))
                shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_alt + median_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if semitone_shift:
            shifted_f0_alt[F0_alt > 1] = inference.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], semitone_shift)
    else:
        F0_ori, shifted_f0_alt = None, None

    cond, *_ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, *_ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    max_source_window = max_context_window - mel2.size(2)
    processed, chunks, previous = 0, [], None
    total_chunks = max(1, -(-cond.size(1) // max(1, max_source_window - overlap_frame_len)))
    while processed < cond.size(1):
        chunk_cond = cond[:, processed:processed + max_source_window]
        is_last = processed + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(device),
                                            mel2, style2, None, steps, inference_cfg_rate=cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        print(f"chunk {len(chunks) + 1}/{total_chunks}", flush=True)
        if processed == 0 and is_last:
            chunks.append(vc_wave[0].cpu().numpy())
            break
        if processed == 0:
            chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
        elif is_last:
            chunks.append(inference.crossfade(previous.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            break
        else:
            chunks.append(inference.crossfade(previous.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                              overlap_wave_len))
        previous = vc_wave[0, -overlap_wave_len:]
        processed += vc_target.size(2) - overlap_frame_len
    wave = torch.tensor(np.concatenate(chunks))[None, :].float()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torchaudio.save(output, wave.cpu(), sr)
    rtf = (time.time() - started) / max(1, wave.size(-1)) * sr
    print(f"RTF: {rtf:.3f}", flush=True)
    return rtf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("profile")
    build.add_argument("--reference", action="append", required=True)
    build.add_argument("--weight", action="append", type=float, default=[])
    build.add_argument("--output", required=True)
    build.add_argument("--no-f0", action="store_true", help="speech checkpoint (22 kHz) instead of the singing one")
    conv = sub.add_parser("convert")
    conv.add_argument("--profile", required=True)
    conv.add_argument("--source", required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--steps", type=int, default=30)
    conv.add_argument("--cfg-rate", type=float, default=0.7)
    conv.add_argument("--length-adjust", type=float, default=1.0)
    conv.add_argument("--semitone-shift", type=int, default=0)
    conv.add_argument("--auto-f0", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "profile":
        weights = list(args.weight) + [1.0] * (len(args.reference) - len(args.weight))
        facts = build_profile(args.reference, weights[: len(args.reference)], args.output,
                              f0_condition=not args.no_f0)
        print(f"profile: {facts}", flush=True)
        return 0
    convert(args.profile, args.source, args.output, steps=args.steps, cfg_rate=args.cfg_rate,
            length_adjust=args.length_adjust, semitone_shift=args.semitone_shift, auto_f0=args.auto_f0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
