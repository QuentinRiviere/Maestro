"""Score a converted vocal stem: timbre match to the target voice, pitch fidelity to the source, singing MOS.

Runs in Seed-VC's venv next to ``seedvc_runner.py`` (same checkout: its CAMPPlus speaker encoder and RMVPE pitch
tracker are the judges' ears), prints one JSON object per candidate.

    python seedvc_judge.py --source <source_vocals.wav> --reference <ref.wav> [--weight W] ... \\
        --candidate <name>=<vocals.wav> [...] [--mos]

Scores:
- ``timbre``: cosine similarity between the candidate's CAMPPlus embedding and the weighted target embedding
  (1 = same voice); ``timbre_source`` the same against the source singer (lower = more converted).
- ``pitch_corr``: Pearson correlation of log-F0 on frames both voiced; ``pitch_cents``: median absolute
  deviation in cents; ``voiced_agreement``: share of frames where both agree on voiced/unvoiced.
- ``mos``: SingMOS (South-Twilight/SingMOS, singing-voice MOS predictor, 1-5) when ``--mos`` is given,
  UTMOS22 as the fallback when it cannot load.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

JUDGE_SECONDS = 60
SR_16K = 16000


def _seedvc():
    here = os.getcwd()
    if here not in sys.path:
        sys.path.insert(0, here)
    import inference  # noqa: F401 - Seed-VC's own module, loaded from its checkout
    return inference


def _load():
    inference = _seedvc()
    return inference, inference.load_models(type("A", (), {"f0_condition": True, "checkpoint": None, "config": None,
                                                           "fp16": True})())


def _audio16k(path: str) -> np.ndarray:
    import librosa
    return librosa.load(path, sr=SR_16K)[0][: JUDGE_SECONDS * SR_16K].astype(np.float32)


def embed(campplus_model, audio16k: np.ndarray, device) -> np.ndarray:
    """CAMPPlus speaker embedding over 20 s windows, averaged (one window when shorter)."""
    import torch
    import torchaudio

    window = 20 * SR_16K
    pieces = [audio16k[i:i + window] for i in range(0, max(1, len(audio16k) - SR_16K), window)] or [audio16k]
    vectors = []
    for piece in pieces:
        if len(piece) < 2 * SR_16K:
            continue
        wave = torch.tensor(piece).unsqueeze(0).float().to(device)
        feat = torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0, sample_frequency=SR_16K)
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            vectors.append(campplus_model(feat.unsqueeze(0)).squeeze(0).float().cpu().numpy())
    vector = np.mean(vectors, axis=0) if vectors else np.zeros(192, dtype=np.float32)
    return vector / (np.linalg.norm(vector) or 1.0)


def pitch(f0_fn, audio16k: np.ndarray, device) -> np.ndarray:
    import torch
    # Seed-VC calls its RMVPE as f0_fn(wave_1d, thred=0.03).
    wave = torch.tensor(audio16k).float().to(device)
    with torch.no_grad():
        f0 = f0_fn(wave, thred=0.03)
    return np.asarray(f0.float().cpu().numpy() if hasattr(f0, "cpu") else f0, dtype=np.float64).reshape(-1)


def pitch_scores(source_f0: np.ndarray, candidate_f0: np.ndarray) -> dict:
    n = min(len(source_f0), len(candidate_f0))
    a, b = source_f0[:n], candidate_f0[:n]
    voiced = (a > 0) & (b > 0)
    agreement = float(np.mean((a > 0) == (b > 0))) if n else 0.0
    if voiced.sum() < 10:
        return {"pitch_corr": 0.0, "pitch_cents": None, "voiced_agreement": round(agreement, 3)}
    la, lb = np.log2(a[voiced]), np.log2(b[voiced])
    corr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else 0.0
    cents = float(np.median(np.abs(lb - la)) * 1200)
    return {"pitch_corr": round(corr, 3), "pitch_cents": round(cents, 1), "voiced_agreement": round(agreement, 3)}


def mos_predictor():
    """SingMOS (singing) first, UTMOS22 (speech naturalness) as the fallback; returns (name, fn(audio16k))."""
    import torch
    import torchaudio

    # s3prl (SingMOS's backbone) still calls torchaudio.set_audio_backend, gone since torchaudio 2.9.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    try:
        model = torch.hub.load("South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True)
        model.eval()

        def singmos(audio16k: np.ndarray) -> float:
            wave = torch.from_numpy(audio16k).unsqueeze(0)
            with torch.no_grad():
                return float(model(wave, torch.tensor([wave.shape[1]])).reshape(-1)[0])
        return "singmos", singmos
    except Exception as error:  # noqa: BLE001 - fall back to the speech predictor
        print(f"[judge] SingMOS unavailable ({str(error)[:120]}), using UTMOS22", file=sys.stderr)
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model.eval()

    def utmos(audio16k: np.ndarray) -> float:
        wave = torch.from_numpy(audio16k).unsqueeze(0)
        with torch.no_grad():
            return float(model(wave, SR_16K).reshape(-1)[0])
    return "utmos22", utmos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="the source vocals the candidates were converted from")
    parser.add_argument("--reference", action="append", default=[], help="a target voice reference (repeat)")
    parser.add_argument("--profile", default="", help="a Seed-VC voice profile (.npz) whose style vector is the target")
    parser.add_argument("--output", default="", help="write the scores as a JSON list here too")
    parser.add_argument("--weight", action="append", type=float, default=[], help="its weight (repeat)")
    parser.add_argument("--candidate", action="append", default=[], help="name=path of a converted stem (repeat)")
    parser.add_argument("--mos", action="store_true", help="also predict a MOS (downloads the predictor once)")
    args = parser.parse_args()
    import torch

    inference, (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args) = _load()
    device = inference.device
    weights = list(args.weight) + [1.0] * (len(args.reference) - len(args.weight))
    total = float(sum(weights[: len(args.reference)])) or 1.0
    target = np.zeros(192, dtype=np.float32)
    if args.profile:
        target = np.load(args.profile)["style"].astype(np.float32)
    else:
        for path, weight in zip(args.reference, weights):
            target += embed(campplus_model, _audio16k(path), device) * (weight / total)
    target /= np.linalg.norm(target) or 1.0
    source16k = _audio16k(args.source)
    source_vector = embed(campplus_model, source16k, device)
    source_f0 = pitch(f0_fn, source16k, device)
    predictor = mos_predictor() if args.mos else None
    results = []
    for item in args.candidate:
        name, _, path = item.partition("=")
        audio = _audio16k(path)
        vector = embed(campplus_model, audio, device)
        scores = {"candidate": name, "seconds": round(len(audio) / SR_16K, 1),
                  "timbre": round(float(vector @ target), 3), "timbre_source": round(float(vector @ source_vector), 3),
                  **pitch_scores(source_f0, pitch(f0_fn, audio, device))}
        if predictor:
            scores[predictor[0]] = round(predictor[1](audio), 2)
        results.append(scores)
        print(json.dumps(scores), flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
