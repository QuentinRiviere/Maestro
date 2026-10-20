"""A permanent character voice: an RVC model trained with Applio on the worker, and conversion with it.

Seed-VC gives a voice in one shot from a few references but drifts a little
between songs; RVC (retrieval-based voice conversion) trained on a few
minutes of that voice (its references plus the vocals Seed-VC already sang)
is the stable version of it. Applio (MIT) is cloned once under
``$MAESTRO_DATA/tools/applio`` at a pinned commit, with a venv layered on the
worker's packages like Seed-VC's, and driven through its ``core.py`` CLI:

- ``voice_train`` job: ``preprocess`` (slice the dataset), ``extract``
  (RMVPE pitch + ContentVec features), ``train`` (from the HiFi-GAN
  pretrained pair, so a few hundred epochs over a few minutes of audio are
  enough) and ``index`` (FAISS retrieval index). Outputs ``<name>.pth`` and
  ``<name>.index``.
- ``convert_vocals_rvc``: ``infer`` on a vocal stem with that pair, used by
  the cover job when its voice carries an RVC model (``engine: "rvc"``).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT = "5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1"
# Applio's engine requirements minus what the worker already ships (torch, transformers, librosa, numpy...).
APPLIO_EXTRA_PACKAGES = ("faiss-cpu==1.14.3", "stftpitchshift==2.0", "soxr==1.1.0", "edge-tts==7.2.8", "tensorboard==2.21.0",
                         "beautifulsoup4==4.15.0", "sounddevice==0.5.5", "webrtcvad-wheels==2.0.14",
                         "torchcrepe==0.0.24", "torchfcpe==0.0.4")
SAMPLE_RATES = (32000, 40000, 48000)
DEFAULT_EPOCHS = 300
MAX_EPOCHS = 2000
# Applio's training lines read "<name> | epoch=12 | step=108 | batch=3/9".
_EPOCH = re.compile(r"epoch=(\d+)")
_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class TrainError(RuntimeError):
    pass


def data_root() -> str:
    return os.environ.get("MAESTRO_DATA") or os.getcwd()


def applio_dir() -> str:
    return os.path.join(data_root(), "tools", "applio")


def _python(folder: str) -> str:
    return os.path.join(folder, ".venv", "bin", "python")


def train_config(raw: Mapping[str, Any] | None) -> dict:
    raw = dict(raw or {})

    def number(key: str, default: float, low: float, high: float, cast=int):
        try:
            value = cast(raw.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(low, min(high, value))

    rate = number("sample_rate", 40000, 32000, 48000)
    return {
        "name": _NAME.sub("_", str(raw.get("name") or "voice")).strip("_")[:60] or "voice",
        "epochs": number("epochs", DEFAULT_EPOCHS, 1, MAX_EPOCHS),
        "batch_size": number("batch_size", 8, 1, 50),
        "sample_rate": rate if rate in SAMPLE_RATES else 40000,
        "f0_method": str(raw.get("f0_method") or "rmvpe") if raw.get("f0_method") in ("crepe", "rmvpe", "fcpe") else "rmvpe",
        "embedder": "contentvec",
        "vocoder": "HiFi-GAN",
        "save_every_epoch": number("save_every_epoch", 50, 1, MAX_EPOCHS),
    }


def infer_config(raw: Mapping[str, Any] | None) -> dict:
    raw = dict(raw or {})

    def number(key: str, default: float, low: float, high: float):
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(low, min(high, value))

    return {
        "pitch": int(number("semitone_shift", 0, -24, 24)),
        "index_rate": number("index_rate", 0.75, 0.0, 1.0),
        "protect": number("protect", 0.33, 0.0, 0.5),
        "volume_envelope": number("volume_envelope", 1.0, 0.0, 1.0),
        "f0_method": "rmvpe",
    }


def ensure_applio(report: Callable[[str], Any] | None = None) -> str:
    """Clone Applio at its pinned commit, build its venv and fetch its models once; returns the folder."""
    folder = applio_dir()
    say = report or (lambda message: None)
    if not os.path.isfile(os.path.join(folder, "core.py")):
        say("Installing Applio")
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", APPLIO_REPO, folder], check=True)
        subprocess.run(["git", "-C", folder, "checkout", "--quiet", APPLIO_COMMIT], check=True)
    _patch_dataloader(folder)
    python = _python(folder)
    if not os.path.isfile(python):
        say("Installing Applio's packages")
        subprocess.run([sys.executable, "-m", "venv", os.path.join(folder, ".venv")], check=True)
        site = next((p for p in sys.path if p.endswith("site-packages")), "")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site = os.path.join(folder, ".venv", "lib", version, "site-packages")
        os.makedirs(venv_site, exist_ok=True)
        with open(os.path.join(venv_site, "maestro.pth"), "w", encoding="utf-8") as handle:
            handle.write(site + "\n")
        subprocess.run([python, "-m", "pip", "install", "-q", *APPLIO_EXTRA_PACKAGES], check=True)
    if not os.path.isfile(os.path.join(folder, "rvc", "models", "predictors", "rmvpe.pt")):
        say("Downloading Applio's models")
        subprocess.run([python, "core.py", "prerequisites", "--no-exe"], cwd=folder, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return folder


def _patch_dataloader(folder: str) -> None:
    """Applio's training DataLoader spawns 4 workers; with Docker's 64 MB /dev/shm they deadlock, so the count
    becomes the ``APPLIO_NUM_WORKERS`` environment setting (0 here)."""
    path = os.path.join(folder, "rvc", "train", "train.py")
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return
    workers = 'int(os.environ.get("APPLIO_NUM_WORKERS", "4"))'
    changes = [
        ('num_workers=2 if device.type == "mps" else 4,', f"num_workers={workers},"),
        ('        persistent_workers=True,\n        prefetch_factor=2 if device.type == "mps" else 8,\n',
         f"        persistent_workers={workers} > 0,\n"
         f"        prefetch_factor=(2 if device.type == \"mps\" else 8) if {workers} > 0 else None,\n"),
    ]
    patched = text
    for old, new in changes:
        patched = patched.replace(old, new, 1)
    if patched != text:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(patched)


def _run_cli(arguments: list[str], *, report: Callable[[str, float], Any] | None = None,
             is_cancelled: Callable[[], bool] | None = None, label: str = "Training", total: int = 0) -> str:
    """Run one ``core.py`` command in Applio's venv; mirrors epoch progress out of ``total``; returns the output tail."""
    folder = ensure_applio(lambda message: report and report(message, 0.0))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "APPLIO_NUM_WORKERS": os.environ.get("APPLIO_NUM_WORKERS", "0")}
    process = subprocess.Popen([_python(folder), "core.py", *arguments], cwd=folder, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail: list[str] = []
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(256), ""):
        if is_cancelled and is_cancelled():
            process.kill()
            raise TrainError("Cancelled")
        tail.append(chunk)
        tail = tail[-60:]
        found = _EPOCH.findall(chunk)
        if found and report and total:
            epoch = int(found[-1])
            report(f"{label}, epoch {epoch} of {total}", 100.0 * epoch / total)
    process.wait()
    text = "".join(tail)
    if process.returncode != 0:
        lines = [line for line in text.strip().splitlines() if line.strip()]
        raise TrainError(f"Applio {arguments[0]} failed: " + (lines[-1] if lines else f"exit {process.returncode}")[:300])
    return text


def _newest(pattern: str) -> str:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else ""


def train_voice(dataset: list[str], config: Mapping[str, Any], out_dir: str, *,
                report: Callable[[str, float], Any] | None = None,
                is_cancelled: Callable[[], bool] | None = None) -> tuple[str, str]:
    """Train an RVC model on the dataset files; returns the (.pth, .index) copied into ``out_dir``."""
    folder = ensure_applio(lambda message: report and report(message, 0.0))
    name = config["name"]
    rate = str(config["sample_rate"])
    logs = os.path.join(folder, "logs", name)
    shutil.rmtree(logs, ignore_errors=True)
    dataset_dir = tempfile.mkdtemp(prefix=f"dataset_{name}_", dir=out_dir)
    try:
        for index, path in enumerate(dataset):
            shutil.copyfile(path, os.path.join(dataset_dir, f"{index:03d}{os.path.splitext(path)[1].lower() or '.wav'}"))
        say = report or (lambda message, percent: None)
        say("Slicing the dataset", 1.0)
        _run_cli(["preprocess", "--model-name", name, "--dataset-path", dataset_dir, "--sample-rate", rate,
                  "--cpu-cores", str(min(8, os.cpu_count() or 4)), "--cut-preprocess", "Automatic"],
                 report=lambda m, p: say(m, 1.0), is_cancelled=is_cancelled, label="Slicing")
        say("Extracting pitch and features", 4.0)
        _run_cli(["extract", "--model-name", name, "--f0-method", config["f0_method"], "--cpu-cores",
                  str(min(8, os.cpu_count() or 4)), "--gpu", "0", "--sample-rate", rate,
                  "--embedder-model", config["embedder"]],
                 report=lambda m, p: say(m, 4.0), is_cancelled=is_cancelled, label="Extracting")
        say("Training", 8.0)
        _run_cli(["train", "--model-name", name, "--total-epoch", str(config["epochs"]), "--save-every-epoch",
                  str(min(config["save_every_epoch"], config["epochs"])), "--save-only-latest", "--batch-size",
                  str(config["batch_size"]), "--gpu", "0", "--sample-rate", rate, "--pretrained", "--vocoder",
                  config["vocoder"], "--cleanup"],
                 report=lambda m, p: say(m, 8.0 + 0.88 * p), is_cancelled=is_cancelled, label="Training",
                 total=int(config["epochs"]))
        # The final weights are "<name>_<epochs>e_<steps>s.pth"; G_/D_ files are the raw checkpoints.
        model = _newest(os.path.join(logs, f"{name}_*e_*s.pth"))
        index = _newest(os.path.join(logs, "*.index"))
        if not index:
            say("Building the retrieval index", 97.0)
            _run_cli(["index", "--model-name", name], report=lambda m, p: say(m, 97.0), is_cancelled=is_cancelled)
            index = _newest(os.path.join(logs, "*.index"))
        if not model:
            raise TrainError("Applio produced no model file")
        pth_out = os.path.join(out_dir, f"{name}.pth")
        index_out = os.path.join(out_dir, f"{name}.index")
        shutil.copyfile(model, pth_out)
        if index:
            shutil.copyfile(index, index_out)
        return pth_out, index_out if index else ""
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def convert_vocals_rvc(source: str, model: str, index: str, output: str, config: Mapping[str, Any], *,
                       report: Callable[[str, float], Any] | None = None,
                       is_cancelled: Callable[[], bool] | None = None) -> str:
    """Applio's inference of ``source`` with the RVC pair; returns the converted wav."""
    arguments = ["infer", "--input-path", source, "--output-path", output, "--pth-path", model,
                 "--f0-method", config["f0_method"], "--pitch", str(config["pitch"]),
                 "--index-rate", str(config["index_rate"]), "--protect", str(config["protect"]),
                 "--volume-envelope", str(config["volume_envelope"]), "--export-format", "WAV"]
    if index and os.path.isfile(index):
        arguments += ["--index-path", index]
    if report:
        report("Converting the voice (RVC)", 0.0)
    _run_cli(arguments, report=None, is_cancelled=is_cancelled, label="Converting")
    if not os.path.isfile(output):
        raise TrainError("Applio produced no file")
    return output


def run_train_job(job: dict, raw_params: Mapping[str, Any]) -> bool:
    """Train the voice in ``raw_params`` (``voice_train`` settings, ``train_dataset`` files); publish the pair."""
    import wgp
    from services.job_lifecycle import finish_job, is_cancel_requested, update_job

    config = train_config(raw_params.get("voice_train"))
    dataset = [str(p) for p in (raw_params.get("train_dataset") or []) if p]
    out_dir = job.get("out_dir") or wgp.save_path
    os.makedirs(out_dir, exist_ok=True)

    def report(message: str, percent: float, **fields: Any) -> bool:
        return update_job(job, message=message, phase="Voice training", progress=round(percent, 1), **fields)

    try:
        if not dataset:
            raise TrainError("The training needs audio files")
        for path in dataset:
            if not os.path.isfile(path):
                raise TrainError(f"The dataset file {os.path.basename(path)} is missing")
        model, index = train_voice(dataset, config, out_dir, report=report,
                                   is_cancelled=lambda: is_cancel_requested(job))
        outputs = [os.path.basename(model)] + ([os.path.basename(index)] if index else [])
        finish_job(job, "completed", progress=100, step=0, total_steps=0, phase="", message="Done",
                   output_files=outputs)
        print(f"[Voice training] {config['name']}: {', '.join(outputs)} ({config['epochs']} epochs)")
        return job.get("status") == "completed"
    except TrainError as error:
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
