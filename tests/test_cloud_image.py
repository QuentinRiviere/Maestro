"""Cloud image providers (fal.ai, Replicate) against mocked HTTP."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import cloud_image  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"


def load_spec(model_type):
    with open(os.path.join(_APP_DIR, "defaults", f"{model_type}.json"), encoding="utf-8") as handle:
        return json.load(handle)["model"]["cloud"]


class Recorder:
    def __init__(self, handler):
        self.requests = []
        self._handler = handler

    def __call__(self, request):
        self.requests.append(request)
        return self._handler(request)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def run(spec, params, recorder, **kwargs):
    with recorder.client() as http:
        return cloud_image.generate(spec, params, api_key="k-123", http=http, sleep=lambda _: None, **kwargs)


class BuildInputTests(unittest.TestCase):
    def test_aspect_ratio_count_and_seed(self):
        spec = load_spec("cloud_fal_flux_pro_ultra")
        payload = cloud_image.build_input(spec, {"prompt": "a fox", "resolution": "1280x720",
                                                 "batch_size": 9, "seed": 7}, [])
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(payload["num_images"], 4)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["output_format"], "png")

    def test_random_seed_is_omitted_and_size_object(self):
        spec = load_spec("cloud_fal_seedream_4")
        payload = cloud_image.build_input(spec, {"prompt": "x", "resolution": "832x1248", "seed": -1}, ["u1"])
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["image_size"], {"width": 832, "height": 1248})
        self.assertEqual(payload["image_urls"], ["u1"])

    def test_single_image_field(self):
        spec = load_spec("cloud_replicate_flux_kontext_pro")
        payload = cloud_image.build_input(spec, {"prompt": "x", "resolution": "1024x1024"}, ["a", "b"])
        self.assertEqual(payload["input_image"], "a")


class ApiKeyTests(unittest.TestCase):
    def test_stored_key_then_environment(self):
        self.assertEqual(cloud_image.provider_api_key("fal", {"fal_api_key": "stored"}, {"FAL_KEY": "env"}), "stored")
        self.assertEqual(cloud_image.provider_api_key("fal", {}, {"FAL_KEY": "env"}), "env")
        self.assertEqual(cloud_image.provider_api_key("replicate", {}, {"REPLICATE_API_TOKEN": "r"}), "r")
        self.assertEqual(cloud_image.provider_api_key("replicate", {}, {}), "")

    def test_missing_key_fails_before_any_request(self):
        recorder = Recorder(lambda request: httpx.Response(500))
        with recorder.client() as http, self.assertRaisesRegex(cloud_image.CloudImageError, "No fal.ai API key"):
            cloud_image.generate(load_spec("cloud_fal_nano_banana"), {"prompt": "x"}, api_key="", http=http)
        self.assertEqual(recorder.requests, [])


class FalTests(unittest.TestCase):
    def test_queue_flow_with_edit_endpoint(self):
        polls = iter([{"status": "IN_QUEUE", "queue_position": 2}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED"}])

        def handler(request):
            url = str(request.url)
            if request.method == "POST":
                return httpx.Response(200, json={"request_id": "r1", "status_url": "https://queue.fal.run/s",
                                                 "response_url": "https://queue.fal.run/r",
                                                 "cancel_url": "https://queue.fal.run/c"})
            if url.endswith("/s"):
                return httpx.Response(200, json=next(polls))
            if url.endswith("/r"):
                return httpx.Response(200, json={"images": [{"url": "https://cdn.fal.media/out.png"}]})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        recorder = Recorder(handler)
        messages = []
        with tempfile.NamedTemporaryFile(suffix=".png") as ref:
            ref.write(PNG)
            ref.flush()
            images = run(load_spec("cloud_fal_nano_banana"),
                         {"prompt": "make it night", "resolution": "1024x1024", "image_refs": [ref.name]},
                         recorder, update=lambda **u: messages.append(u.get("message")))
        submit = recorder.requests[0]
        self.assertEqual(str(submit.url), "https://queue.fal.run/fal-ai/nano-banana/edit")
        self.assertEqual(submit.headers["authorization"], "Key k-123")
        body = json.loads(submit.content)
        self.assertTrue(body["image_urls"][0].startswith("data:image/png;base64,"))
        self.assertEqual(images[0].data, PNG)
        self.assertEqual(images[0].extension, ".png")
        self.assertIn("Queued on fal.ai (position 2)…", messages)

    def test_text_only_uses_base_endpoint_and_reports_errors(self):
        def handler(request):
            return httpx.Response(422, json={"detail": [{"msg": "prompt too long"}]})

        recorder = Recorder(handler)
        with self.assertRaisesRegex(cloud_image.CloudImageError, "prompt too long"):
            run(load_spec("cloud_fal_nano_banana"), {"prompt": "x"}, recorder)
        self.assertEqual(str(recorder.requests[0].url), "https://queue.fal.run/fal-ai/nano-banana")

    def test_cancel_stops_the_remote_request(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"status_url": "https://q/s", "response_url": "https://q/r",
                                                 "cancel_url": "https://q/c"})
            return httpx.Response(200, json={})

        recorder = Recorder(handler)
        with self.assertRaises(cloud_image.CloudImageCancelled):
            run(load_spec("cloud_fal_flux_pro_ultra"), {"prompt": "x"}, recorder, is_cancelled=lambda: True)
        self.assertEqual((recorder.requests[-1].method, str(recorder.requests[-1].url)), ("PUT", "https://q/c"))


class ReplicateTests(unittest.TestCase):
    def test_prediction_flow_uploads_references(self):
        polls = iter([{"status": "starting"}, {"status": "succeeded", "output": "https://replicate.delivery/o.jpg"}])

        def handler(request):
            url = str(request.url)
            if url.endswith("/v1/files"):
                return httpx.Response(201, json={"urls": {"get": "https://api.replicate.com/v1/files/f1"}})
            if url.endswith("/predictions"):
                return httpx.Response(201, json={"urls": {"get": "https://api.replicate.com/v1/predictions/p1",
                                                          "cancel": "https://api.replicate.com/v1/predictions/p1/cancel"}})
            if url.endswith("/p1"):
                return httpx.Response(200, json=next(polls))
            return httpx.Response(200, content=b"jpg", headers={"content-type": "image/jpeg"})

        recorder = Recorder(handler)
        with tempfile.NamedTemporaryFile(suffix=".png") as ref:
            ref.write(PNG)
            ref.flush()
            images = run(load_spec("cloud_replicate_nano_banana"),
                         {"prompt": "x", "resolution": "1280x720", "image_refs": [ref.name]}, recorder)
        submit = next(r for r in recorder.requests if str(r.url).endswith("/predictions"))
        self.assertEqual(str(submit.url), "https://api.replicate.com/v1/models/google/nano-banana/predictions")
        self.assertEqual(submit.headers["authorization"], "Bearer k-123")
        self.assertEqual(json.loads(submit.content)["input"]["image_input"], ["https://api.replicate.com/v1/files/f1"])
        self.assertEqual(json.loads(submit.content)["input"]["aspect_ratio"], "16:9")
        self.assertEqual(images[0].extension, ".jpg")

    def test_failed_prediction(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"urls": {"get": "https://api.replicate.com/v1/predictions/p1"}})
            return httpx.Response(200, json={"status": "failed", "error": "NSFW content detected"})

        with self.assertRaisesRegex(cloud_image.CloudImageError, "NSFW content detected"):
            run(load_spec("cloud_replicate_flux_pro_ultra"), {"prompt": "x"}, Recorder(handler))

    def test_bad_key(self):
        with self.assertRaisesRegex(cloud_image.CloudImageError, "rejected the API key"):
            run(load_spec("cloud_replicate_seedream_4"), {"prompt": "x"}, Recorder(lambda r: httpx.Response(401)))

    def test_required_reference_image(self):
        spec = dict(load_spec("cloud_replicate_flux_kontext_pro"), requires_image=True)
        with self.assertRaisesRegex(cloud_image.CloudImageError, "reference image"):
            run(spec, {"prompt": "x"}, Recorder(lambda r: httpx.Response(500)))


class ModelDefinitionTests(unittest.TestCase):
    def test_every_cloud_definition_is_valid(self):
        names = [n for n in os.listdir(os.path.join(_APP_DIR, "defaults")) if n.startswith("cloud_")]
        self.assertEqual(len(names), 8)
        for name in names:
            spec = load_spec(name[:-5])
            self.assertIsNotNone(cloud_image.cloud_spec({"cloud": spec}), name)
            cloud_image.build_input(spec, {"prompt": "x", "resolution": "1024x1024"}, ["u"])


if __name__ == "__main__":
    unittest.main()
