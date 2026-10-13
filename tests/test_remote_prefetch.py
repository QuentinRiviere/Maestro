"""Boot-time R2 prefetch and the streamed s5cmd pull it relies on."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import textwrap
import threading
import time

import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import boot_events, prefetch  # noqa: E402
from services.remote_worker.checkpoint_cache import CheckpointCacheError, R2CheckpointCache  # noqa: E402

# A stand-in s5cmd: ``cat`` and ``run`` against <bucket root>/<bucket>/...,
# printing one line per finished copy like the real one.
FAKE_S5CMD = textwrap.dedent('''\
    #!{python}
    import os, shlex, shutil, sys
    root = os.environ["FAKE_R2_ROOT"]
    args = sys.argv[3:]
    def local(url):
        return os.path.join(root, url[len("s3://"):]) if url.startswith("s3://") else url
    if args[0] == "cat":
        path = local(args[1])
        if not os.path.exists(path):
            sys.exit(1)
        sys.stdout.write(open(path).read())
    elif args[0] == "run":
        for line in open(args[1]):
            parts = shlex.split(line)
            if not parts:
                continue
            src, dst = parts[-2], parts[-1]
            os.makedirs(os.path.dirname(local(dst)), exist_ok=True)
            shutil.copyfile(local(src), local(dst))
            print(f"cp {{src}} {{dst}}", flush=True)
''')


@pytest.fixture
def r2(tmp_path, monkeypatch):
    script = tmp_path / "s5cmd"
    script.write_text(FAKE_S5CMD.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    bucket = tmp_path / "r2"
    monkeypatch.setenv("FAKE_R2_ROOT", str(bucket))
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "kid")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("MAESTRO_DATA", str(tmp_path / "data"))
    return script, bucket


def _publish(bucket, model_type, files, *, tamper=None):
    """Put files and a hashed manifest in the fake bucket."""
    entries = []
    for path, data in files.items():
        target = bucket / "videogen" / "maestro" / "ckpts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tamper[path] if tamper and path in tamper else data)
        entries.append({"root": "ckpts", "path": path, "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = bucket / "videogen" / "maestro" / "manifests" / f"{model_type}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"model_type": model_type, "files": entries, "links": []}))


def _cache(script, tmp_path):
    return R2CheckpointCache(bucket="videogen", prefix="maestro", endpoint_url="https://r2.example",
                             roots={"ckpts": str(tmp_path / "data" / "ckpts")}, s5cmd=str(script),
                             require_hashes=True)


def test_streamed_pull_verifies_every_file(r2, tmp_path):
    script, bucket = r2
    _publish(bucket, "m", {"a.safetensors": b"A" * 64, "sub/b.safetensors": b"B" * 32})
    cache = _cache(script, tmp_path)
    assert cache.hydrate("m") == 2
    assert (tmp_path / "data" / "ckpts" / "sub" / "b.safetensors").read_bytes() == b"B" * 32
    assert cache.hydrate("m") == 0


def test_streamed_pull_removes_a_tampered_file(r2, tmp_path):
    script, bucket = r2
    _publish(bucket, "m", {"a.safetensors": b"A" * 64}, tamper={"a.safetensors": b"X" * 64})
    with pytest.raises(CheckpointCacheError, match="do not match"):
        _cache(script, tmp_path).hydrate("m")
    assert not (tmp_path / "data" / "ckpts" / "a.safetensors").exists()


def test_prefetch_reports_boot_events_and_releases_waiters(r2, tmp_path, capsys):
    script, bucket = r2
    _publish(bucket, "m", {"a.safetensors": b"A" * 64})
    assert prefetch.prefetch(["m", "missing"], _cache(script, tmp_path)) == 0
    events = boot_events.parse_events(
        {"timestamp": "", "message": line} for line in capsys.readouterr().out.splitlines())
    assert [(e["model_type"], e["state"]) for e in events if e["state"] != "progress"] == [
        ("m", "started"), ("m", "done"), ("missing", "started"), ("missing", "done")]
    final = [e for e in events if e["state"] == "progress"][-1]
    assert (final["model_type"], final["done_gigabytes"]) == ("m", 0.0)
    started = time.time()
    prefetch.wait_for("m")  # finished: returns at once
    assert time.time() - started < 1


def test_maestro_waits_for_a_running_prefetch(r2, tmp_path):
    marker_dir = tmp_path / "data" / ".prefetch"
    marker_dir.mkdir(parents=True)
    running = marker_dir / "m.running"
    running.touch()
    threading.Timer(0.3, lambda: os.replace(running, marker_dir / "m.done")).start()
    started = time.time()
    prefetch.wait_for("m", timeout=5)
    assert 0.2 < time.time() - started < 3


def test_pull_progress_reaches_the_total(r2, tmp_path):
    script, bucket = r2
    _publish(bucket, "m", {"a.safetensors": b"A" * 4096, "b.safetensors": b"B" * 1024})
    seen = []
    assert _cache(script, tmp_path).hydrate("m", on_progress=seen.append) == 2
    assert seen[-1]["done_bytes"] == seen[-1]["total_bytes"] == 5120
    assert seen[-1]["files_done"] == seen[-1]["files_total"] == 2
