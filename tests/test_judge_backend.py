"""The Director judges on a local OpenAI-compatible vision server (judge_backend)."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.director import judge_backend, storyboard_judge, video_judge  # noqa: E402

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def write_image(path: str, size=(1600, 900), colour=(200, 40, 40)) -> str:
    Image.new("RGB", size, colour).save(path)
    return path


def write_video(path: str, seconds: float = 4.0) -> str:
    """A tiny synthetic mp4 (needs ffmpeg)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10",
                    "-t", f"{seconds}", "-pix_fmt", "yuv420p", path], check=True, capture_output=True, timeout=60)
    return path


def answer(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class SettingsTests(unittest.TestCase):
    def test_default_is_fal_with_its_key(self):
        backend = judge_backend.settings({"fal_api_key": "f"}, environ={})
        self.assertEqual((backend.provider, backend.key, backend.available), ("fal", "f", True))
        self.assertFalse(judge_backend.settings({}, environ={}).available)
        self.assertEqual(backend.resolve_model("google/gemini-3.8-flash"), "google/gemini-3.8-flash")
        self.assertEqual(backend.media_noun, "video")

    def test_openai_settings_and_defaults(self):
        backend = judge_backend.settings({"director_judge_provider": "openai",
                                          "director_judge_url": "https://spark.tailnet/v1/",
                                          "director_judge_api_key": "k"}, environ={})
        self.assertEqual(backend.provider, "openai")
        self.assertEqual(backend.url, "https://spark.tailnet/v1")
        self.assertEqual((backend.key, backend.model), ("k", "judge"))
        self.assertTrue(backend.available)
        self.assertEqual(backend.resolve_model("google/gemini-3.8-flash"), "judge")
        self.assertEqual(backend.media_noun, "image")
        # No URL: the backend is configured but unusable.
        self.assertFalse(judge_backend.settings({"director_judge_provider": "openai"}, environ={}).available)

    def test_environment_fills_unset_settings_and_stored_values_win(self):
        environ = {"MAESTRO_JUDGE_PROVIDER": "openai", "MAESTRO_JUDGE_URL": "http://env/v1",
                   "MAESTRO_JUDGE_API_KEY": "env-key", "MAESTRO_JUDGE_MODEL": "env-model"}
        backend = judge_backend.settings({}, environ=environ)
        self.assertEqual((backend.provider, backend.url, backend.key, backend.model),
                         ("openai", "http://env/v1", "env-key", "env-model"))
        stored = judge_backend.settings({"director_judge_url": "http://stored/v1", "director_judge_model": "m"},
                                        environ=environ)
        self.assertEqual((stored.url, stored.key, stored.model), ("http://stored/v1", "env-key", "m"))
        self.assertEqual(judge_backend.settings({"director_judge_provider": "bogus"}, environ={}).provider, "fal")

    def test_chat_completions_url(self):
        self.assertEqual(judge_backend.chat_completions_url("http://h:8080/v1/"), "http://h:8080/v1/chat/completions")
        self.assertEqual(judge_backend.chat_completions_url("http://h:8080"), "http://h:8080/v1/chat/completions")


class AskTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.backend = judge_backend.JudgeBackend(provider="openai", url="http://judge/v1", key="secret",
                                                  model="judge")

    def test_image_request_body_shape(self):
        image = write_image(os.path.join(self.dir, "a.png"))
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=answer('{"pass": true}'))

        with client(handler) as http:
            text, cost = self.backend.ask(http, "secret", "judge", "SYS", "Shot plan", [image], lambda _: None,
                                          max_tokens=400, timeout=30)
        self.assertEqual((text, cost), ('{"pass": true}', 0.0))
        request = requests[0]
        self.assertEqual(str(request.url), "http://judge/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer secret")
        body = json.loads(request.content)
        self.assertEqual((body["model"], body["temperature"], body["max_tokens"], body["stream"]),
                         ("judge", 0.2, 400, False))
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        parts = body["messages"][1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "Shot plan"})
        self.assertEqual(parts[1], {"type": "text", "text": "Image 1:"})
        self.assertEqual(parts[2]["type"], "image_url")
        url = parts[2]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as sent:
            self.assertEqual(sent.size, (768, 432))

    def test_retry_once_on_server_error_and_think_blocks_are_stripped(self):
        image = write_image(os.path.join(self.dir, "a.png"))
        statuses = iter([503, 200])
        sleeps = []

        def handler(request):
            status = next(statuses)
            if status != 200:
                return httpx.Response(status, text="loading")
            return httpx.Response(200, json=answer("<think>hmm</think>\n{\"best\": 2}"))

        with client(handler) as http:
            text, _ = self.backend.ask(http, "", "judge", "S", "P", [image], sleeps.append, max_tokens=10, timeout=5)
        self.assertEqual(text, '{"best": 2}')
        self.assertEqual(sleeps, [2.0])

    def test_persistent_server_error_raises(self):
        image = write_image(os.path.join(self.dir, "a.png"))
        with client(lambda request: httpx.Response(500, text="boom")) as http, \
                self.assertRaises(video_judge.JudgeError):
            self.backend.ask(http, "", "judge", "S", "P", [image], lambda _: None, max_tokens=10, timeout=5)

    def test_timeout_is_retried_then_raised(self):
        image = write_image(os.path.join(self.dir, "a.png"))
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("slow")

        with client(handler) as http, self.assertRaises(video_judge.JudgeError):
            self.backend.ask(http, "", "judge", "S", "P", [image], lambda _: None, max_tokens=10, timeout=5)
        self.assertEqual(len(calls), 2)

    def test_missing_url_fails_clearly(self):
        backend = judge_backend.JudgeBackend(provider="openai", url="")
        with client(lambda request: httpx.Response(200)) as http, self.assertRaises(video_judge.JudgeError):
            backend.ask(http, "", "judge", "S", "P", [], lambda _: None)

    def test_video_is_sent_as_frames_with_timestamps_in_the_prompt(self):
        """Frame extraction is mocked; the real ffmpeg path is covered below."""
        video = os.path.join(self.dir, "clip.mp4")
        with open(video, "wb") as handle:
            handle.write(b"\x00")
        frames = [write_image(os.path.join(self.dir, f"f{n}.jpg"), size=(320, 180)) for n in range(3)]

        def extract(path, dest_dir, count=8, long_side=768):
            self.assertEqual(path, video)
            return [(0.0, frames[0]), (1.5, frames[1]), (3.0, frames[2])]

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=answer("ok"))

        with patch.object(judge_backend, "extract_frames", extract), client(handler) as http:
            self.backend.ask(http, "", "judge", "S", "Video 1 is take A", [video], lambda _: None)
        parts = json.loads(requests[0].content)["messages"][1]["content"]
        self.assertEqual(parts[0]["text"], "Video 1 is take A\nVideo 1 is shown as 3 frames at t=0.0s, 1.5s, 3.0s.\n"
                                           + judge_backend.SILENT_FRAMES_NOTE)
        self.assertEqual(parts[1]["text"], "Video 1, frames in order:")
        self.assertEqual([p["text"] for p in parts[2::2]], ["t=0.0s", "t=1.5s", "t=3.0s"])
        self.assertEqual(sum(1 for p in parts if p["type"] == "image_url"), 3)

    def test_data_uri_video_is_decoded_before_extraction(self):
        seen = []

        def extract(path, dest_dir, count=8, long_side=768):
            with open(path, "rb") as handle:
                seen.append(handle.read())
            return [(0.0, write_image(os.path.join(self.dir, "f.jpg"), size=(64, 64)))]

        with patch.object(judge_backend, "extract_frames", extract), \
                client(lambda request: httpx.Response(200, json=answer("ok"))) as http:
            self.backend.ask(http, "", "judge", "S", "P", ["data:video/mp4;base64," + base64.b64encode(b"mp4!").decode()],
                             lambda _: None)
        self.assertEqual(seen, [b"mp4!"])

    @unittest.skipUnless(FFMPEG, "ffmpeg/ffprobe not installed")
    def test_real_frames_are_evenly_spaced_and_downscaled(self):
        video = write_video(os.path.join(self.dir, "clip.mp4"), seconds=4.0)
        frames = judge_backend.extract_frames(video, self.dir, count=4, long_side=160)
        self.assertEqual([t for t, _ in frames], [0.0, 1.0, 2.0, 3.0])
        with Image.open(frames[0][1]) as frame:
            self.assertEqual(frame.size, (160, 90))

    def test_frame_times_are_capped(self):
        self.assertEqual(judge_backend.frame_times(6.4, 8), [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6])
        self.assertEqual(len(judge_backend.frame_times(100.0, 40)), judge_backend.MAX_VIDEO_FRAMES)
        self.assertEqual(judge_backend.frame_times(None, 3), [0.0, 1.0, 2.0])


class StoryboardJudgeBackendTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_openai_backend_sends_images_not_videos(self):
        shot = {"path": write_image(os.path.join(self.dir, "shot.png")), "framing": "close-up",
                "cast_sheets": [{"name": "Mina", "path": write_image(os.path.join(self.dir, "mina.png"))}]}
        backend = judge_backend.JudgeBackend(provider="openai", url="http://judge/v1", model="judge")
        seen = {}

        def ask(http, key, model, system, prompt, media, sleep, **kwargs):
            seen.update(system=system, prompt=prompt, media=media, model=model)
            return '{"pass": true, "issues": [], "fix_hint": ""}', 0.0

        with patch.object(storyboard_judge, "still_as_video", side_effect=AssertionError("no mp4 conversion")):
            verdict = storyboard_judge.check_shot(shot, fal_key="", model="judge", ask=ask, http=object(),
                                                  backend=backend)
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["cost_usd"], 0.0)
        self.assertEqual(seen["media"], [shot["path"], shot["cast_sheets"][0]["path"]])
        self.assertIn("image 2 = Mina", seen["prompt"])
        self.assertIn("shown as an image, image 1", seen["system"])
        self.assertNotIn("video", seen["system"].split("Check:")[0].lower().replace("video pipeline", "")
                         .replace("video render", ""))

    def test_rank_candidates_numbers_images_with_the_openai_backend(self):
        paths = [write_image(os.path.join(self.dir, f"c{n}.png")) for n in range(2)]
        backend = judge_backend.JudgeBackend(provider="openai", url="http://judge/v1")
        seen = {}

        def ask(http, key, model, system, prompt, media, sleep, **kwargs):
            seen.update(system=system, prompt=prompt, media=media)
            return '{"best": 2, "verdicts": [{"index": 1, "verdict": "flagged"}, {"index": 2, "verdict": "pass"}]}', 0.0

        ranking = storyboard_judge.rank_candidates({"framing": "wide"}, paths, fal_key="", ask=ask, http=object(),
                                                   backend=backend)
        self.assertEqual(ranking["best"], 2)
        self.assertEqual(seen["media"], paths)
        self.assertIn("image 1 = candidate 1, image 2 = candidate 2", seen["prompt"])
        self.assertEqual(seen["system"], storyboard_judge.rank_prompt("image"))

    def test_fal_backend_still_converts_stills_to_video(self):
        shot = {"path": write_image(os.path.join(self.dir, "shot.png"))}
        seen = {}

        def ask(http, key, model, system, prompt, media, sleep, **kwargs):
            seen.update(media=media, system=system)
            return '{"pass": true}', 0.002

        with patch.object(storyboard_judge, "still_as_video", side_effect=lambda src, dest: src), \
                patch.object(storyboard_judge, "data_uri", side_effect=lambda path: "data:video/mp4;base64,AA"):
            storyboard_judge.check_shot(shot, fal_key="k", ask=ask, http=object())
        self.assertEqual(seen["media"], ["data:video/mp4;base64,AA"])
        self.assertIs(seen["system"], storyboard_judge.VISION_PROMPT)

    def test_settings_enable_vision_without_a_fal_key_on_the_local_backend(self):
        services = {"director_judge_provider": "openai", "director_judge_url": "http://judge/v1"}
        with patch.dict(os.environ, {}, clear=False):
            config = storyboard_judge.settings(services, {}, "")
        self.assertTrue(config["vision"])
        self.assertEqual((config["provider"], config["model"]), ("openai", "judge"))
        self.assertEqual(config["backend"].provider, "openai")
        fal = storyboard_judge.settings({}, {}, "fal-key")
        self.assertEqual((fal["provider"], fal["fal_key"], fal["vision"]), ("fal", "fal-key", True))


class VideoJudgeBackendTests(unittest.TestCase):
    def test_review_loops_use_the_backend_and_record_it(self):
        backend = judge_backend.JudgeBackend(provider="openai", url="http://judge/v1", model="judge")
        calls = []

        def judge(url, brief, **kwargs):
            calls.append((url, kwargs.get("backend")))
            return {"score": 7, "cost_usd": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "s1.mp4")
            with open(clip, "wb") as handle:
                handle.write(b"\x00")
            with patch.object(video_judge, "cut_segment", lambda source, start, end, dest: source):
                review = video_judge.review_clips([{"video_filename": "s1.mp4"}], [], tmp, fal_key="",
                                                  publish=lambda p: p, judge=judge, backend=backend)
        self.assertEqual(review["backend"], "openai")
        self.assertEqual(calls, [(clip, backend)])
        self.assertEqual(review["cost_usd"], 0.0)
        self.assertEqual(video_judge.review_clips([], [], "/nowhere", fal_key="k", publish=lambda p: p)["backend"],
                         "fal")

    def test_compare_takes_asks_the_backend(self):
        backend = judge_backend.JudgeBackend(provider="openai", url="http://judge/v1", model="judge")
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            winner = "A" if len(requests) == 1 else "B"
            return httpx.Response(200, json=answer(f'{{"winner": "{winner}", "confidence": 0.9}}'))

        with tempfile.TemporaryDirectory() as tmp:
            a = write_image(os.path.join(tmp, "a.png"), size=(64, 64))
            b = write_image(os.path.join(tmp, "b.png"), size=(64, 64))
            with client(handler) as http:
                result = video_judge.compare_takes(a, b, {"label": "hook"}, fal_key="", model="judge", http=http,
                                                   sleep=lambda _: None, backend=backend)
        self.assertEqual((result["winner"], result["cost_usd"]), ("A", 0.0))
        self.assertEqual([len(r["messages"][1]["content"]) for r in requests], [5, 5])


class LoadJsonObjectTests(unittest.TestCase):
    def test_a_clean_object_with_text_around_it(self):
        self.assertEqual(judge_backend.load_json_object('Here: {"score": 7, "issues": []} done {"x": 1}'),
                         {"score": 7, "issues": []})

    def test_trailing_commas_are_dropped(self):
        self.assertEqual(judge_backend.load_json_object('{"score": 7, "issues": [1, 2,],}'),
                         {"score": 7, "issues": [1, 2]})

    def test_an_answer_cut_by_the_token_limit_keeps_its_complete_elements(self):
        text = '{"score": 4, "scores": {"rhythm": 5, "coverage": 7}, "issues": [{"shot": 1, "severity": "major"}, {"shot": 2, "sev'
        self.assertEqual(judge_backend.load_json_object(text),
                         {"score": 4, "scores": {"rhythm": 5, "coverage": 7}, "issues": [{"shot": 1, "severity": "major"}]})

    def test_braces_inside_strings_do_not_close_the_object(self):
        self.assertEqual(judge_backend.load_json_object('{"note": "a } in text", "score": 3}'),
                         {"note": "a } in text", "score": 3})

    def test_no_json_and_junk_raise(self):
        with self.assertRaises(video_judge.JudgeError):
            judge_backend.load_json_object("no braces here", "The episode judge")
        with self.assertRaises(video_judge.JudgeError):
            judge_backend.load_json_object("{ this is not json at all", "The episode judge")

    def test_the_episode_parser_survives_a_cut_answer(self):
        text = '{"score": 4, "verdict": "needs_polish", "scores": {}, "defects": [{"shot": 1, "at_sec": 2, "severity": "major", "description": "x"}, {"shot": 2, "at_s'
        verdict = video_judge.parse_episode_verdict(text, 20.0, [])
        self.assertEqual(verdict["score"], 4)
        self.assertEqual(len(verdict["defects"]), 1)


class BadEscapeTest(unittest.TestCase):
    def test_a_prose_escape_in_a_string_is_dropped(self):
        # Run f36ffbb5: the episode judge wrote "the model\'s inability" and the review failed.
        text = "Here: {\"summary\": \"hides the model\\'s inability to hold\", \"score\": 5}\n```"
        self.assertEqual(judge_backend.load_json_object(text), {"summary": "hides the model's inability to hold", "score": 5})
        # Real escapes survive.
        self.assertEqual(judge_backend.load_json_object('{"a": "line\\nbreak \\"q\\""}'), {"a": 'line\nbreak "q"'})


if __name__ == "__main__":
    unittest.main()
