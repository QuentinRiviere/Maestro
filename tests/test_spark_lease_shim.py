"""docker/spark-stack/maestro_worker_lease.py against a fake worker HTTP server."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHIM = os.path.join(_ROOT, "docker", "spark-stack", "maestro_worker_lease.py")
TOKEN = "lease-token"


def _load_shim():
    spec = importlib.util.spec_from_file_location("maestro_worker_lease", _SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWorker:
    def __init__(self, token=TOKEN, release_delay=0.0):
        self.calls = []
        self.paused = True
        self.release_delay = release_delay
        self.release_done = threading.Event()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self):
                worker.calls.append((self.command, self.path, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != f"Bearer {token}":
                    return self._reply(401, {"detail": "Unauthorized"})
                if self.path == "/api/v1/worker/acquire":
                    worker.paused = False
                    return self._reply(200, {"paused": False, "loaded_models": []})
                if self.path == "/api/v1/worker/state":
                    return self._reply(200, {"paused": worker.paused})
                if self.path == "/api/v1/worker/release":
                    worker.paused = True
                    time.sleep(worker.release_delay)  # a render finishing
                    worker.release_done.set()
                    return self._reply(200, {"paused": True, "released": ["ltx2"], "handed_back": ["j1"]})
                return self._reply(404, {})

            do_GET = _handle
            do_POST = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self):
        return [path for _method, path, _auth in self.calls]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class LeaseShimTests(unittest.TestCase):
    def setUp(self):
        self.shim = _load_shim()
        self.env = dict(os.environ)
        os.environ["MAESTRO_WORKER_TOKEN"] = TOKEN
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self.env)

    def _run_in_thread(self, argv):
        stop = threading.Event()
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("code", self.shim.run(argv, stop)))
        thread.start()
        return stop, thread, result

    def test_acquires_then_releases_on_stop(self):
        worker = FakeWorker()
        self.addCleanup(worker.close)
        stop, thread, result = self._run_in_thread(["--url", worker.url, "--check-interval", "0.05"])
        self.assertTrue(_wait_for(lambda: "/api/v1/worker/acquire" in worker.paths()))
        self.assertFalse(worker.paused)
        stop.set()
        thread.join(timeout=5)
        self.assertEqual(result["code"], 0)
        self.assertEqual(worker.paths()[-1], "/api/v1/worker/release")
        self.assertTrue(worker.paused)
        self.assertTrue(all(auth == f"Bearer {TOKEN}" for _m, _p, auth in worker.calls))

    def test_resumes_a_worker_paused_behind_its_back(self):
        worker = FakeWorker()
        self.addCleanup(worker.close)
        stop, thread, result = self._run_in_thread(["--url", worker.url, "--check-interval", "0.05"])
        self.assertTrue(_wait_for(lambda: "/api/v1/worker/acquire" in worker.paths()))
        worker.paused = True  # e.g. the worker container restarted paused
        self.assertTrue(_wait_for(lambda: worker.paths().count("/api/v1/worker/acquire") >= 2))
        stop.set()
        thread.join(timeout=5)
        self.assertEqual(result["code"], 0)

    def test_waits_for_a_worker_that_is_not_up_yet_and_releases_a_stopped_one(self):
        url = f"http://127.0.0.1:{_free_port()}"
        stop, thread, result = self._run_in_thread(["--url", url, "--retry", "0.05"])
        time.sleep(0.2)
        self.assertTrue(thread.is_alive())  # keeps retrying the acquire
        stop.set()
        thread.join(timeout=5)
        # Nothing listens at release time: the worker holds no memory.
        self.assertEqual(result["code"], 0)

    def test_wrong_token_exits_with_an_error(self):
        worker = FakeWorker(token="other")
        self.addCleanup(worker.close)
        stop, thread, result = self._run_in_thread(["--url", worker.url])
        thread.join(timeout=5)
        self.assertEqual(result["code"], 2)

    @unittest.skipIf(os.name != "posix", "SIGTERM semantics")
    def test_sigterm_exits_only_after_the_release_returned(self):
        worker = FakeWorker(release_delay=0.8)
        self.addCleanup(worker.close)
        env = {**os.environ, "MAESTRO_WORKER_URL": worker.url, "MAESTRO_WORKER_TOKEN": TOKEN}
        process = subprocess.Popen([sys.executable, _SHIM, "--check-interval", "0.1"], env=env,
                                   stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(_wait_for(lambda: "/api/v1/worker/acquire" in worker.paths()))
            sent = time.monotonic()
            process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=10)
            self.assertEqual(code, 0)
            self.assertTrue(worker.release_done.is_set())
            self.assertGreaterEqual(time.monotonic() - sent, 0.7)
        finally:
            if process.poll() is None:
                process.kill()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
