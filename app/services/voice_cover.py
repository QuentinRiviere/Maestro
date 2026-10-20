"""Worker job: a song cover sung by a character's voice.

The job (``raw_params["voice_cover"]``, see ``services.director.voice_cover``
on the controller) takes a song and the target voice, either a ready
profile (``cover_profile``, an ``.npz``) or one or more references with
weights (``cover_references`` / ``cover_weights``):

1. BS-RoFormer splits the song into a vocal stem and an instrumental
   (``preprocessing.extract_vocals.get_stems``, the separator wgp already
   ships for audio-driven models).
2. Seed-VC converts the vocal stem to the voice, zero-shot, with its
   singing checkpoint (44.1 kHz, F0-conditioned; rap goes through the
   same model, the F0 track carries the flow). The voice is a *profile*
   (``services.seedvc_runner``): the references' CAMPPlus timbre vectors
   blended by weight plus a prompt made of a slice of each, so a
   character's voice is one fixed file reused by every cover. Seed-VC
   runs as a subprocess in its own venv under ``$MAESTRO_DATA/tools/seed-vc``
   (cloned at a pinned commit, the venv layered on the worker's packages
   through a ``.pth`` link), so its pins never touch the worker.
3. The converted vocals are mixed back over the instrumental.

Outputs in the job's ``out_dir``: ``cover.wav`` (the mix), ``vocals.wav``
(the converted stem), ``source_vocals.wav``, ``instrumental.wav`` and
``voice_profile.npz`` (the profile used, for the controller to keep).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
SEED_VC_EXTRA_PACKAGES = ("descript-audio-codec", "jiwer", "pydub", "soundfile", "python-dotenv")
DEFAULT_STEPS = 30
DEFAULT_CFG_RATE = 0.7
MAX_STEPS = 100
MODES = ("sing", "rap")
ENGINES = ("seedvc", "rvc")
PROFILE_NAME = "voice_profile.npz"
_PROGRESS = re.compile(r"chunk (\d+)/(\d+)")
_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seedvc_runner.py")


class CoverError(RuntimeError):
    pass


def cover_config(raw: Mapping[str, Any] | None) -> dict:
    """The job's settings with defaults and bounds."""
    raw = dict(raw or {})

    def number(key: str, default: float, low: float, high: float, cast=float):
        try:
            value = cast(raw.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(low, min(high, value))

    mode = str(raw.get("mode") or "sing")
    engine = str(raw.get("engine") or "seedvc")
    return {
        "name": re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw.get("name") or "cover")).strip("_")[:60] or "cover",
        "mode": mode if mode in MODES else "sing",
        # seedvc: zero-shot from the profile; rvc: the voice's trained RVC model (services.voice_train).
        "engine": engine if engine in ENGINES else "seedvc",
        "steps": number("steps", DEFAULT_STEPS, 4, MAX_STEPS, int),
        "cfg_rate": number("cfg_rate", DEFAULT_CFG_RATE, 0.0, 1.0),
        "semitone_shift": number("semitone_shift", 0, -12, 12, int),
        # Rap keeps the rapper's own pitch level: the flow is the performance, not a key to transpose.
        "auto_f0": bool(raw.get("auto_f0", False)) and mode != "rap",
        "length_adjust": number("length_adjust", 1.0, 0.5, 2.0),
        "vocal_gain_db": number("vocal_gain_db", 0.0, -12.0, 12.0),
        "index_rate": number("index_rate", 0.75, 0.0, 1.0),
        "protect": number("protect", 0.33, 0.0, 0.5),
    }


def voice_weights(references: list[str], weights: Any) -> list[float]:
    """One positive weight per reference (missing ones 1.0), normalised to sum 1."""
    values = []
    for index in range(len(references)):
        try:
            value = float(weights[index]) if isinstance(weights, (list, tuple)) and index < len(weights) else 1.0
        except (TypeError, ValueError):
            value = 1.0
        values.append(max(0.0, value))
    total = sum(values)
    if total <= 0:
        values, total = [1.0] * len(references), float(len(references) or 1)
    return [round(v / total, 4) for v in values]


def data_root() -> str:
    return os.environ.get("MAESTRO_DATA") or os.getcwd()


def seed_vc_dir() -> str:
    return os.path.join(data_root(), "tools", "seed-vc")


def _venv_python(folder: str) -> str:
    return os.path.join(folder, ".venv", "bin", "python")


def ensure_seed_vc(report: Callable[[str], Any] | None = None) -> str:
    """Clone Seed-VC at its pinned commit and build its venv once; returns the folder."""
    folder = seed_vc_dir()
    say = report or (lambda message: None)
    if not os.path.isfile(os.path.join(folder, "inference.py")):
        say("Installing Seed-VC")
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", SEED_VC_REPO, folder], check=True)
        subprocess.run(["git", "-C", folder, "checkout", "--quiet", SEED_VC_COMMIT], check=True)
    python = _venv_python(folder)
    if not os.path.isfile(python):
        say("Installing Seed-VC's packages")
        subprocess.run([sys.executable, "-m", "venv", os.path.join(folder, ".venv")], check=True)
        # The venv sees the worker's packages (torch, transformers, librosa, whisper…) through a .pth link
        # and only adds what Seed-VC needs on top.
        site = next((p for p in sys.path if p.endswith("site-packages")), "")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site = os.path.join(folder, ".venv", "lib", version, "site-packages")
        os.makedirs(venv_site, exist_ok=True)
        with open(os.path.join(venv_site, "maestro.pth"), "w", encoding="utf-8") as handle:
            handle.write(site + "\n")
        subprocess.run([python, "-m", "pip", "install", "-q", *SEED_VC_EXTRA_PACKAGES], check=True)
    return folder


def _run_runner(arguments: list[str], *, report: Callable[[str, float], Any] | None = None,
                is_cancelled: Callable[[], bool] | None = None, label: str = "Converting the voice") -> None:
    """Run ``seedvc_runner.py`` in Seed-VC's venv, mirroring its chunk progress; raises on failure."""
    folder = ensure_seed_vc(lambda message: report and report(message, 0.0))
    env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME") or os.path.join(data_root(), "hf"),
           "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen([_venv_python(folder), _RUNNER, *arguments], cwd=folder, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail: list[str] = []
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(256), ""):
        if is_cancelled and is_cancelled():
            process.kill()
            raise CoverError("Cancelled")
        tail.append(chunk)
        tail = tail[-40:]
        found = _PROGRESS.findall(chunk)
        if found and report:
            step, total = found[-1]
            if int(total):
                report(f"{label}, part {step} of {total}", 100.0 * int(step) / int(total))
    process.wait()
    if process.returncode != 0:
        lines = [line for line in "".join(tail).strip().splitlines() if line.strip()]
        raise CoverError("Seed-VC failed: " + (lines[-1] if lines else f"exit {process.returncode}")[:300])


def build_profile(references: list[str], weights: list[float], output: str, *,
                  report: Callable[[str, float], Any] | None = None,
                  is_cancelled: Callable[[], bool] | None = None) -> str:
    """Blend the references into one voice profile (``.npz``); returns its path."""
    arguments = ["profile", "--output", output]
    for path, weight in zip(references, weights):
        arguments += ["--reference", path, "--weight", str(weight)]
    _run_runner(arguments, report=report, is_cancelled=is_cancelled, label="Building the voice")
    if not os.path.isfile(output):
        raise CoverError("Seed-VC produced no voice profile")
    return output


def convert_vocals(source: str, profile: str, output: str, config: Mapping[str, Any], *,
                   report: Callable[[str, float], Any] | None = None,
                   is_cancelled: Callable[[], bool] | None = None) -> str:
    """Seed-VC's singing conversion of ``source`` to the profile's voice; returns the converted wav."""
    arguments = ["convert", "--profile", profile, "--source", source, "--output", output,
                 "--steps", str(config["steps"]), "--cfg-rate", str(config["cfg_rate"]),
                 "--length-adjust", str(config["length_adjust"]), "--semitone-shift", str(config["semitone_shift"])]
    if config["auto_f0"]:
        arguments.append("--auto-f0")
    _run_runner(arguments, report=report, is_cancelled=is_cancelled)
    if not os.path.isfile(output):
        raise CoverError("Seed-VC produced no file")
    return output


def mix(vocals: str, instrumental: str, output: str, *, vocal_gain_db: float = 0.0) -> str:
    """Sum the converted vocals over the instrumental (resampled to it), peak-normalised when it clips."""
    import librosa
    import numpy as np
    import soundfile as sf

    bed, rate = sf.read(instrumental, dtype="float32", always_2d=True)
    voice, voice_rate = librosa.load(vocals, sr=rate, mono=True)
    voice = voice.astype(np.float32) * float(10 ** (vocal_gain_db / 20))
    length = max(bed.shape[0], voice.shape[0])
    out = np.zeros((length, bed.shape[1]), dtype=np.float32)
    out[: bed.shape[0]] += bed
    out[: voice.shape[0]] += voice[:, None]
    peak = float(np.abs(out).max() or 0.0)
    if peak > 0.98:
        out *= 0.98 / peak
    sf.write(output, out, rate, subtype="PCM_16")
    return output


def run_cover_job(job: dict, raw_params: Mapping[str, Any]) -> bool:
    """Separate, convert and remix the song in ``raw_params``; publish the files as the job's outputs."""
    import wgp
    from services.job_lifecycle import finish_job, is_cancel_requested, update_job

    config = cover_config(raw_params.get("voice_cover"))
    song = str(raw_params.get("cover_song") or "")
    profile = str(raw_params.get("cover_profile") or "")
    references = [str(p) for p in (raw_params.get("cover_references") or []) if p]
    if not references and raw_params.get("cover_reference"):
        references = [str(raw_params["cover_reference"])]
    weights = voice_weights(references, raw_params.get("cover_weights"))
    rvc_model = str(raw_params.get("cover_rvc_model") or "")
    rvc_index = str(raw_params.get("cover_rvc_index") or "")
    use_rvc = config["engine"] == "rvc"
    out_dir = job.get("out_dir") or wgp.save_path
    os.makedirs(out_dir, exist_ok=True)

    def report(message: str, percent: float, **fields: Any) -> bool:
        return update_job(job, message=message, phase="Cover", progress=round(percent, 1), **fields)

    work_dir = tempfile.mkdtemp(prefix="voice_cover_", dir=out_dir)
    try:
        if not song or not os.path.isfile(song):
            raise CoverError("The song file is missing")
        if profile and not os.path.isfile(profile):
            raise CoverError("The voice profile file is missing")
        if use_rvc and not os.path.isfile(rvc_model):
            raise CoverError("The voice's RVC model file is missing")
        if not use_rvc and not profile and not references:
            raise CoverError("The cover needs a voice: a profile or at least one reference")
        for path in references:
            if not os.path.isfile(path):
                raise CoverError(f"The voice reference {os.path.basename(path)} is missing")
        names = {"cover": "cover.wav", "vocals": "vocals.wav", "source_vocals": "source_vocals.wav",
                 "instrumental": "instrumental.wav", "profile": PROFILE_NAME}
        cancelled = lambda: is_cancel_requested(job)  # noqa: E731
        if use_rvc:
            names.pop("profile")
        elif profile:
            shutil.copyfile(profile, os.path.join(out_dir, names["profile"]))
        else:
            report("Building the voice", 1.0)
            build_profile(references, weights, os.path.join(out_dir, names["profile"]),
                          report=lambda m, p: report(m, 1.0), is_cancelled=cancelled)
        if cancelled():
            return False
        report("Separating the vocals from the music", 5.0)
        from preprocessing.extract_vocals import get_stems
        source_vocals, instrumental = get_stems(song, os.path.join(work_dir, "stems"))
        if cancelled():
            return False
        report("Converting the voice", 12.0)
        if use_rvc:
            from services.voice_train import convert_vocals_rvc, infer_config
            converted = convert_vocals_rvc(source_vocals, rvc_model, rvc_index, os.path.join(work_dir, "converted.wav"),
                                           infer_config(config), report=lambda m, p: report(m, 12.0),
                                           is_cancelled=cancelled)
        else:
            converted = convert_vocals(source_vocals, os.path.join(out_dir, names["profile"]),
                                       os.path.join(work_dir, "converted.wav"), config,
                                       report=lambda m, p: report(m, 12.0 + 0.83 * p), is_cancelled=cancelled)
        report("Mixing the cover", 96.0)
        mix(converted, instrumental, os.path.join(out_dir, names["cover"]), vocal_gain_db=config["vocal_gain_db"])
        shutil.copyfile(converted, os.path.join(out_dir, names["vocals"]))
        shutil.copyfile(source_vocals, os.path.join(out_dir, names["source_vocals"]))
        shutil.copyfile(instrumental, os.path.join(out_dir, names["instrumental"]))
        finish_job(job, "completed", progress=100, step=0, total_steps=0, phase="", message="Done",
                   output_files=list(names.values()))
        print(f"[Voice cover] {config['name']}: {names['cover']} ({config['engine']}, {config['steps']} steps)")
        return job.get("status") == "completed"
    except CoverError as error:
        if str(error) == "Cancelled" or is_cancel_requested(job):
            return False
        finish_job(job, "failed", error=str(error), message=f"Error: {error}")
        return False
    except Exception as error:  # noqa: BLE001 - recorded on the job
        import traceback
        traceback.print_exc()
        if not is_cancel_requested(job):
            finish_job(job, "failed", error=str(error)[:400], message=f"Error: {str(error)[:200]}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
