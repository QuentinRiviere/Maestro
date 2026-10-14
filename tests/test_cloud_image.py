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


class RangeTests(unittest.TestCase):
    def test_values_are_clamped_to_provider_ranges(self):
        spec = load_spec("cloud_replicate_flux_dev_lora")
        payload = cloud_image.build_input(spec, {"prompt": "x", "guidance_scale": 12.0, "num_inference_steps": 80}, [])
        self.assertEqual((payload["guidance"], payload["num_inference_steps"]), (10.0, 50))
        self.assertEqual(cloud_image.clamp(spec, "lora_scale", 5.0), 3.0)
        self.assertEqual(cloud_image.clamp(spec, "unknown", 5.0), 5.0)


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

    def test_poll_and_download_retry_after_a_read_timeout_but_submit_does_not(self):
        timeouts = {"/s": 1, "out.png": 1}

        def handler(request):
            url = str(request.url)
            for key in timeouts:
                if url.endswith(key) and timeouts[key]:
                    timeouts[key] -= 1
                    raise httpx.ReadTimeout("The read operation timed out", request=request)
            if request.method == "POST":
                return httpx.Response(200, json={"status_url": "https://q/s", "response_url": "https://q/r",
                                                 "cancel_url": "https://q/c"})
            if url.endswith("/s"):
                return httpx.Response(200, json={"status": "COMPLETED"})
            if url.endswith("/r"):
                return httpx.Response(200, json={"images": [{"url": "https://cdn.fal.media/out.png"}]})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        images = run(load_spec("cloud_fal_nano_banana"), {"prompt": "x"}, Recorder(handler))
        self.assertEqual(images[0].data, PNG)

        def submit_times_out(request):
            raise httpx.ReadTimeout("The read operation timed out", request=request)

        recorder = Recorder(submit_times_out)
        with self.assertRaisesRegex(cloud_image.CloudImageError, "timed out"):
            run(load_spec("cloud_fal_nano_banana"), {"prompt": "x"}, recorder)
        self.assertEqual(len(recorder.requests), 1)

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


def lora_file(directory, name="style.safetensors", size=32):
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(b"x" * size)
    return path


class LoraTests(unittest.TestCase):
    def test_scales_take_the_first_phase_value(self):
        self.assertEqual(cloud_image.parse_lora_scales("0.8 1.2;0.5", 3), [0.8, 1.2, 1.0])
        self.assertEqual(cloud_image.parse_lora_scales(None, 2), [1.0, 1.0])

    def test_fal_uploads_once_then_reuses_the_cached_url(self):
        def handler(request):
            url = str(request.url)
            if url == cloud_image.FAL_TOKEN_URL:
                return httpx.Response(200, json={"token": "t", "token_type": "Bearer",
                                                 "base_url": "https://v3.fal.media"})
            if url == "https://v3.fal.media/files/upload":
                assert request.headers["authorization"] == "Bearer t"
                assert request.headers["x-fal-file-name"] == "style.safetensors"
                return httpx.Response(200, json={"access_url": "https://v3.fal.media/style.safetensors"})
            if url == "https://queue.fal.run/fal-ai/flux-lora":
                return httpx.Response(200, json={"status_url": "https://q/s", "response_url": "https://q/r"})
            if url == "https://q/s":
                return httpx.Response(200, json={"status": "COMPLETED"})
            if url == "https://q/r":
                return httpx.Response(200, json={"images": [{"url": "https://cdn/o.png"}]})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        spec = load_spec("cloud_fal_flux_dev_lora")
        params = {"prompt": "x", "resolution": "1024x1024", "num_inference_steps": 20, "guidance_scale": 4}
        with tempfile.TemporaryDirectory() as tmp:
            cache = cloud_image.UploadCache(os.path.join(tmp, "uploads.json"))
            lora = cloud_image.CloudLora(lora_file(tmp), scale=0.7)
            for _ in range(2):
                recorder = Recorder(handler)
                run(spec, params, recorder, loras=[lora], upload_cache=cache, wall_clock=lambda: 1000.0)
                submit = next(r for r in recorder.requests if str(r.url).endswith("/flux-lora"))
                body = json.loads(submit.content)
                self.assertEqual(body["loras"], [{"path": "https://v3.fal.media/style.safetensors", "scale": 0.7}])
                self.assertEqual((body["num_inference_steps"], body["guidance_scale"]), (20, 4.0))
                uploads = [r for r in recorder.requests if str(r.url).endswith("/files/upload")]
            self.assertEqual(uploads, [])  # second run reused the cached URL

    def test_fal_multipart_upload_for_large_files(self):
        def handler(request):
            url = str(request.url)
            if url == cloud_image.FAL_TOKEN_URL:
                return httpx.Response(200, json={"token": "t", "token_type": "Bearer", "base_url": "https://cdn"})
            if url == "https://cdn/files/upload/multipart":
                return httpx.Response(200, json={"access_url": "https://cdn/f", "uploadId": "u"})
            if request.method == "PUT":
                return httpx.Response(200, headers={"etag": f"e{url[-1]}"})
            if url == "https://cdn/f/multipart/u/complete":
                return httpx.Response(200, json={})
            return httpx.Response(500)

        recorder = Recorder(handler)
        original = (cloud_image.FAL_MULTIPART_THRESHOLD, cloud_image.FAL_MULTIPART_CHUNK)
        cloud_image.FAL_MULTIPART_THRESHOLD, cloud_image.FAL_MULTIPART_CHUNK = 10, 16
        try:
            with tempfile.TemporaryDirectory() as tmp, recorder.client() as http:
                lora = cloud_image.CloudLora(lora_file(tmp, size=40))
                url = cloud_image.FalClient("k", http).upload_lora(lora)
        finally:
            cloud_image.FAL_MULTIPART_THRESHOLD, cloud_image.FAL_MULTIPART_CHUNK = original
        self.assertEqual(url, "https://cdn/f")
        complete = json.loads(recorder.requests[-1].content)
        self.assertEqual(complete["parts"], [{"partNumber": n, "etag": f"e{n}"} for n in (1, 2, 3)])

    def test_replicate_uses_civitai_ids_and_token(self):
        def handler(request):
            if str(request.url).endswith("/predictions"):
                return httpx.Response(201, json={"urls": {"get": "https://api.replicate.com/v1/predictions/p"}})
            if str(request.url).endswith("/p"):
                return httpx.Response(200, json={"status": "succeeded", "output": ["https://r/o.png"]})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        recorder = Recorder(handler)
        with tempfile.TemporaryDirectory() as tmp:
            loras = [cloud_image.CloudLora(lora_file(tmp, "a.safetensors"), 0.9, civitai_model_id=123),
                     cloud_image.CloudLora(lora_file(tmp, "b.safetensors"), 0.5, civitai_model_id=456)]
            run(load_spec("cloud_replicate_flux_dev_lora"), {"prompt": "x", "guidance_scale": 3},
                recorder, loras=loras, civitai_token="civ-tok")
        body = json.loads(recorder.requests[0].content)["input"]
        self.assertEqual(body["lora_weights"], "https://civitai.com/models/123")
        self.assertEqual((body["extra_lora"], body["extra_lora_scale"]), ("https://civitai.com/models/456", 0.5))
        self.assertEqual(body["civitai_api_token"], "civ-tok")
        self.assertEqual(body["guidance"], 3.0)

    def test_r2_url_takes_precedence_over_provider_uploads(self):
        def handler(request):
            if str(request.url).endswith("/predictions"):
                return httpx.Response(201, json={"urls": {"get": "https://api.replicate.com/v1/predictions/p"}})
            if str(request.url).endswith("/p"):
                return httpx.Response(200, json={"status": "succeeded", "output": ["https://r/o.png"]})
            if str(request.url).endswith("/v1/files"):
                raise AssertionError("no provider upload expected")
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        recorder = Recorder(handler)
        with tempfile.TemporaryDirectory() as tmp:
            lora = cloud_image.CloudLora(lora_file(tmp), 0.6, civitai_model_id=123)
            run(load_spec("cloud_replicate_flux_dev_lora"), {"prompt": "x"}, recorder, loras=[lora],
                civitai_token="civ", lora_url=lambda item: f"https://r2.example/{item.name}?sig=1")
        body = json.loads(recorder.requests[0].content)["input"]
        self.assertEqual(body["lora_weights"], "https://r2.example/style.safetensors?sig=1")
        self.assertNotIn("civitai_api_token", body)

    def test_replicate_refuses_large_files_without_civitai_source(self):
        spec = load_spec("cloud_replicate_flux_dev_lora")
        with tempfile.TemporaryDirectory() as tmp:
            lora = cloud_image.CloudLora(lora_file(tmp))
            original = cloud_image.REPLICATE_FILE_LIMIT
            cloud_image.REPLICATE_FILE_LIMIT = 10
            try:
                with self.assertRaisesRegex(cloud_image.CloudImageError, "100 MB"):
                    run(spec, {"prompt": "x"}, Recorder(lambda r: httpx.Response(500)), loras=[lora])
            finally:
                cloud_image.REPLICATE_FILE_LIMIT = original

    def test_lora_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = cloud_image.CloudLora(lora_file(tmp))
            with self.assertRaisesRegex(cloud_image.CloudImageError, "does not accept LoRAs"):
                run(load_spec("cloud_fal_nano_banana"), {"prompt": "x"}, Recorder(lambda r: httpx.Response(500)),
                    loras=[lora])
            with self.assertRaisesRegex(cloud_image.CloudImageError, "at most 2"):
                run(load_spec("cloud_replicate_flux_dev_lora"), {"prompt": "x"},
                    Recorder(lambda r: httpx.Response(500)), loras=[lora] * 3)


class ModelDefinitionTests(unittest.TestCase):
    def test_every_cloud_definition_is_valid(self):
        names = [n for n in os.listdir(os.path.join(_APP_DIR, "defaults")) if n.startswith("cloud_")]
        self.assertEqual(len(names), 13)
        for name in names:
            spec = load_spec(name[:-5])
            self.assertIsNotNone(cloud_image.cloud_spec({"cloud": spec}), name)
            cloud_image.build_input(spec, {"prompt": "x", "resolution": "1024x1024"}, ["u"])


if __name__ == "__main__":
    unittest.main()
