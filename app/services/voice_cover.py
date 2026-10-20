"""Worker job: a song cover sung by a character's voice.

The job (``raw_params["voice_cover"]``, see ``services.director.voice_cover``
on the controller) takes a song and a short reference of the target voice:

1. BS-RoFormer splits the song into a vocal stem and an instrumental
   (``preprocessing.extract_vocals.get_stems``, the separator wgp already
   ships for audio-driven models).
2. Seed-VC converts the vocal stem to the reference's timbre, zero-shot,
   with its singing checkpoint (44.1 kHz, F0-conditioned). Seed-VC runs
   as a subprocess in its own venv under ``$MAESTRO_DATA/tools/seed-vc``
   (cloned at a pinned commit, the venv layered on the worker's packages
   through a ``.pth`` link), so its pins never touch the worker.
3. The converted vocals are mixed back over the instrumental.

Outputs in the job's ``out_dir``: ``cover.wav`` (the mix), ``vocals.wav``
(the converted stem), ``source_vocals.wav`` and ``instrumental.wav``.
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
_PROGRESS = re.compile(r"(\d+)/(\d+) \[")


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

    return {
        "name": re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw.get("name") or "cover")).strip("_")[:60] or "cover",
        "steps": number("steps", DEFAULT_STEPS, 4, MAX_STEPS, int),
        "cfg_rate": number("cfg_rate", DEFAULT_CFG_RATE, 0.0, 1.0),
        "semitone_shift": number("semitone_shift", 0, -12, 12, int),
        "auto_f0": bool(raw.get("auto_f0", False)),
        "length_adjust": number("length_adjust", 1.0, 0.5, 2.0),
        "vocal_gain_db": number("vocal_gain_db", 0.0, -12.0, 12.0),
    }


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


def convert_vocals(source: str, reference: str, out_dir: str, config: Mapping[str, Any], *,
                   report: Callable[[str, float], Any] | None = None,
                   is_cancelled: Callable[[], bool] | None = None) -> str:
    """Run Seed-VC's singing conversion of ``source`` to ``reference``'s voice; returns the converted wav."""
    folder = ensure_seed_vc(lambda message: report and report(message, 0.0))
    os.makedirs(out_dir, exist_ok=True)
    command = [
        _venv_python(folder), "inference.py", "--source", source, "--target", reference, "--output", out_dir,
        "--diffusion-steps", str(config["steps"]), "--length-adjust", str(config["length_adjust"]),
        "--inference-cfg-rate", str(config["cfg_rate"]), "--f0-condition", "True",
        "--auto-f0-adjust", "True" if config["auto_f0"] else "False", "--semi-tone-shift", str(config["semitone_shift"]),
    ]
    env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME") or os.path.join(data_root(), "hf"),
           "PYTHONUNBUFFERED": "1"}
    before = set(os.listdir(out_dir))
    process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace")
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
                report(f"Converting the voice, step {step} of {total}", 100.0 * int(step) / int(total))
    process.wait()
    if process.returncode != 0:
        text = "".join(tail).strip().splitlines()
        raise CoverError("Seed-VC failed: " + (text[-1] if text else f"exit {process.returncode}")[:300])
    made = [n for n in os.listdir(out_dir) if n not in before and n.lower().endswith(".wav")]
    if not made:
        raise CoverError("Seed-VC produced no file")
    return os.path.join(out_dir, sorted(made, key=lambda n: os.path.getmtime(os.path.join(out_dir, n)))[-1])


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
    reference = str(raw_params.get("cover_reference") or "")
    out_dir = job.get("out_dir") or wgp.save_path
    os.makedirs(out_dir, exist_ok=True)

    def report(message: str, percent: float, **fields: Any) -> bool:
        return update_job(job, message=message, phase="Cover", progress=round(percent, 1), **fields)

    work_dir = tempfile.mkdtemp(prefix="voice_cover_", dir=out_dir)
    try:
        for label, path in (("song", song), ("voice reference", reference)):
            if not path or not os.path.isfile(path):
                raise CoverError(f"The {label} file is missing")
        report("Separating the vocals from the music", 2.0)
        from preprocessing.extract_vocals import get_stems
        source_vocals, instrumental = get_stems(song, os.path.join(work_dir, "stems"))
        if is_cancel_requested(job):
            return False
        report("Converting the voice", 10.0)
        converted = convert_vocals(source_vocals, reference, os.path.join(work_dir, "vc"), config,
                                   report=lambda m, p: report(m, 10.0 + 0.85 * p),
                                   is_cancelled=lambda: is_cancel_requested(job))
        report("Mixing the cover", 96.0)
        names = {"cover": "cover.wav", "vocals": "vocals.wav", "source_vocals": "source_vocals.wav",
                 "instrumental": "instrumental.wav"}
        mix(converted, instrumental, os.path.join(out_dir, names["cover"]), vocal_gain_db=config["vocal_gain_db"])
        shutil.copyfile(converted, os.path.join(out_dir, names["vocals"]))
        shutil.copyfile(source_vocals, os.path.join(out_dir, names["source_vocals"]))
        shutil.copyfile(instrumental, os.path.join(out_dir, names["instrumental"]))
        finish_job(job, "completed", progress=100, step=0, total_steps=0, phase="", message="Done",
                   output_files=list(names.values()))
        print(f"[Voice cover] {config['name']}: {names['cover']} ({config['steps']} steps)")
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
