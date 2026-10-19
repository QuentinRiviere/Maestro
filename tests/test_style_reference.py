"""Director style references: local editing stats, chunking, parsing, planner block, storage and routes.

No paid API is called: every model call is a fake ``ask``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import bibles  # noqa: E402
from services.director import style_reference as sr  # noqa: E402
from services.director.video_judge import FAL_ROUTER, JudgeError  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_color_video(path: str, segments: list[tuple[str, float]], *, audio: bool = True) -> str:
    """Solid colour segments joined by hard cuts (lavfi), with a tone when ``audio``."""
    command = ["ffmpeg", "-v", "error", "-y"]
    for colour, seconds in segments:
        command += ["-f", "lavfi", "-i", f"color=c={colour}:s=320x240:d={seconds}:r=24"]
    total = sum(seconds for _, seconds in segments)
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={total}"]
    inputs = "".join(f"[{i}:v]" for i in range(len(segments)))
    command += ["-filter_complex", f"{inputs}concat=n={len(segments)}:v=1:a=0[v]", "-map", "[v]"]
    if audio:
        command += ["-map", f"{len(segments)}:a", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return path


def grammar_answer(**extra) -> str:
    body = {
        "rhythm": {"target_asl_sec": 2.4, "pace_with_emotion": "Cuts quicken as the argument escalates.",
                   "notes": "Long holds on silences."},
        "shot_mix": {"close-up": 40, "medium": 35, "wide": 25},
        "patterns": ["Shot/reverse-shot over the shoulder", "Reaction close-up after each key line"],
        "scene_openings": "Open on an insert of a moving object, then a wide.",
        "scene_closings": "Close on a held close-up of the listener.",
        "dialogue_coverage": "Over-the-shoulder pairs, singles for confessions.",
        "camera_movement": "Static, slow push-in on realisations.",
        "lighting_color": "Warm practicals, cool window fill.",
        "music_silence": "Music drops out before the key line.",
        "rules": ["Hold a reaction close-up 2-3 s after every key line.",
                  "Open each scene on an insert of hands or feet before the wide.",
                  "Push in slowly only on a realisation.", "Keep exchanges at 2-3 s per shot.",
                  "End each scene on the listener, not the speaker."],
        "summary_markdown": "## Rhythm\nFast exchanges, long silences.",
    }
    body.update(extra)
    return json.dumps(body)


def shots_answer(count: int) -> str:
    sizes = ["close_up", "medium", "wide", "insert", "ots"]
    return "Here you go:\n```json\n" + json.dumps({"shots": [
        {"shot": n + 1, "shot_size": sizes[n % len(sizes)], "angle": "eye level", "movement": "static",
         "subject": "a woman", "action": "she turns", "cut_reason": "reaction", "transition": "cut",
         "lighting": "warm", "audio": "dialogue"} for n in range(count)]}) + "\n```"


class LocalStatsTests(unittest.TestCase):
    def test_shot_stats_from_cuts(self):
        stats = sr.shot_stats([2.0, 5.0, 5.5, 12.0], 20.0)
        self.assertEqual(stats["shot_count"], 5)
        self.assertEqual(stats["shot_lengths"], [2.0, 3.0, 0.5, 6.5, 8.0])
        self.assertEqual(stats["asl_sec"], 4.0)
        self.assertEqual(stats["median_sec"], 3.0)
        self.assertEqual(stats["cuts_per_minute"], 12.0)
        buckets = {d["bucket"]: d["count"] for d in stats["distribution"]}
        self.assertEqual(buckets, {"<1s": 1, "1-2s": 0, "2-4s": 2, "4-8s": 1, "8s+": 1})
        self.assertEqual(len(stats["pacing"]), 2)
        first = stats["pacing"][0]
        self.assertEqual((first["start"], first["end"], first["cuts"]), (0.0, 10.0, 3))
        self.assertEqual(first["asl_sec"], 2.5)  # 4 shots overlap the first 10 s

    def test_clean_cuts_drops_flashes(self):
        self.assertEqual(sr.clean_cuts([0.1, 2.0, 2.1, 4.0, 9.95], 10.0), [2.0, 4.0])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg is not installed")
    def test_local_analysis_finds_the_known_cuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_color_video(os.path.join(tmp, "cuts.mp4"),
                                    [("red", 2), ("blue", 3), ("green", 1.5), ("white", 2.5)])
            stats = sr.local_analysis(path)
        self.assertAlmostEqual(stats["duration_sec"], 9.0, delta=0.1)
        self.assertAlmostEqual(stats["fps"], 24.0, delta=0.01)
        self.assertTrue(stats["has_audio"])
        self.assertEqual(len(stats["cuts"]), 3)
        for found, expected in zip(stats["cuts"], [2.0, 5.0, 6.5]):
            self.assertAlmostEqual(found, expected, delta=0.05)
        self.assertEqual(stats["shot_count"], 4)
        self.assertAlmostEqual(stats["asl_sec"], 2.25, delta=0.05)
        self.assertAlmostEqual(stats["cuts_per_minute"], 20.0, delta=0.5)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg is not installed")
    def test_encode_chunk_keeps_sound_and_downscales(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_color_video(os.path.join(tmp, "src.mp4"), [("red", 2), ("blue", 2)])
            out = sr.encode_chunk(path, 1.0, 3.0, os.path.join(tmp, "chunk.mp4"))
            facts = sr.probe(out)
        self.assertTrue(facts["has_audio"])
        self.assertLessEqual(facts["width"], sr.CHUNK_WIDTH)
        self.assertAlmostEqual(facts["duration_sec"], 2.0, delta=0.15)


class ChunkTests(unittest.TestCase):
    def test_chunks_align_on_cuts_and_stay_under_the_cap(self):
        shots = sr.shot_spans([20.0, 45.0, 70.0, 100.0, 110.0], 130.0)
        chunks = sr.plan_chunks(shots, max_sec=60)
        self.assertEqual([c["shots"] for c in chunks], [[0, 1], [2, 3], [4, 5]])
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 60)
            self.assertIn(chunk["start"], [s["start"] for s in shots])

    def test_a_long_shot_gets_its_own_capped_chunk(self):
        shots = sr.shot_spans([5.0, 95.0], 100.0)
        chunks = sr.plan_chunks(shots, max_sec=60)
        self.assertEqual([c["shots"] for c in chunks], [[0], [1], [2]])
        self.assertEqual((chunks[1]["start"], chunks[1]["end"]), (5.0, 65.0))

    def test_only_the_analysed_minutes_are_chunked(self):
        shots = sr.shot_spans([float(t) for t in range(10, 300, 10)], 300.0)
        chunks = sr.plan_chunks(shots, max_sec=60, limit_sec=120)
        self.assertLess(max(c["start"] for c in chunks), 120)

    def test_local_chunks_group_by_stills_and_long_shots_take_two(self):
        shots = sr.shot_spans([1.0, 2.0, 6.0, 7.0, 8.0], 20.0)
        self.assertEqual(sr.still_times(shots[0]), [0.5])
        self.assertEqual(sr.still_times(shots[2]), [3.0, 5.0])
        chunks = sr.plan_local_chunks(shots, max_stills=4)
        # shots: 1 s (1 still), 1 s (1), 4 s (2) | 1 s (1), 1 s (1), 12 s (2)
        self.assertEqual([c["shots"] for c in chunks], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual((chunks[1]["start"], chunks[1]["end"]), (6.0, 20.0))

    def test_stills_note_maps_images_to_shots(self):
        note = sr.stills_note([(1, 12.5, "a.jpg"), (2, 13.0, "b.jpg"), (2, 14.0, "c.jpg")], 12.0)
        self.assertIn("Image 1 = shot 1 at 0.5 s, Image 2 = shot 2 at 1.0 s, Image 3 = shot 2 at 2.0 s", note)
        self.assertIn("silent stills", note)

    def test_chunk_prompt_lists_times_relative_to_the_chunk(self):
        shots = sr.shot_spans([20.0, 45.0], 60.0)
        prompt = sr.chunk_prompt({"start": 20.0, "end": 60.0, "shots": [1, 2]}, shots, 2, 3, 60.0)
        self.assertIn("1: 0.00-25.00 s", prompt)
        self.assertIn("2: 25.00-40.00 s", prompt)


class ParsingTests(unittest.TestCase):
    def test_chunk_shots_from_fenced_json_with_aliases(self):
        records = sr.parse_chunk_shots(shots_answer(2), {"start": 10, "end": 20, "shots": [4, 5]})
        self.assertEqual(sorted(records), [4, 5])
        self.assertEqual(records[4]["shot_size"], "close_up")
        self.assertEqual(records[4]["angle"], "eye_level")
        self.assertEqual(records[4]["transition"], "hard_cut")
        self.assertEqual(records[5]["shot_size"], "medium")

    def test_chunk_shots_accept_a_bare_list_and_unknown_values(self):
        text = json.dumps([{"shot_size": "ECU", "movement": "spin", "cut_reason": "??"}, {"shot": 9}])
        records = sr.parse_chunk_shots(text, {"start": 0, "end": 5, "shots": [0, 1]})
        self.assertEqual(records[0]["shot_size"], "extreme_close_up")
        self.assertEqual(records[0]["movement"], "static")
        self.assertEqual(records[0]["cut_reason"], "other")
        self.assertEqual(list(records), [0])  # shot 9 is not in the chunk

    def test_unusable_answers_raise(self):
        for text in ("no json here", "{\"shots\": []}", "{broken"):
            with self.assertRaises(sr.StyleReferenceError):
                sr.parse_chunk_shots(text, {"start": 0, "end": 5, "shots": [0]})

    def test_grammar_parsing_normalises_and_needs_rules(self):
        grammar, summary = sr.parse_grammar("Sure!\n" + grammar_answer())
        self.assertEqual(grammar["rhythm"]["target_asl_sec"], 2.4)
        self.assertEqual(grammar["shot_mix"], {"close_up": 40.0, "medium": 35.0, "wide": 25.0})
        self.assertEqual(len(grammar["rules"]), 5)
        self.assertTrue(summary.startswith("## Rhythm"))
        with self.assertRaises(sr.StyleReferenceError):
            sr.parse_grammar(grammar_answer(rules=[]))

    def test_rules_given_as_text_are_split(self):
        grammar, _ = sr.parse_grammar(grammar_answer(rules="1. Cut on the look.\n- Hold the silence."))
        self.assertEqual(grammar["rules"], ["Cut on the look.", "Hold the silence."])


class FakeAsk:
    """Scripted model answers; records each call (router, prompt, video count)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, http, key, model, system, prompt, urls, sleep, **kwargs):
        self.calls.append({"router": kwargs.get("router", FAL_ROUTER), "system": system, "prompt": prompt,
                           "videos": len(urls), "key": key})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def fake_local(cuts, duration):
    def local(path):
        return {"duration_sec": duration, "fps": 24.0, "has_audio": True, **sr.shot_stats(cuts, duration)}
    return local


class AnalyseTests(unittest.TestCase):
    def run_analysis(self, ask, cuts=(30.0, 50.0, 80.0), duration=100.0, fal_key="key", **kwargs):
        progress = []
        with patch.object(sr, "encode_chunk", lambda src, a, b, dest, **kw: dest), \
                patch.object(sr, "data_uri", lambda path: "data:video/mp4;base64,AAAA"):
            result = sr.analyse_reference("video.mp4", fal_key=fal_key, ask=ask, http=object(),
                                          sleep=lambda s: None, local=fake_local(list(cuts), duration),
                                          on_progress=lambda message, partial: progress.append((message, partial)),
                                          **kwargs)
        return result, progress

    def test_two_chunks_retry_and_synthesis(self):
        ask = FakeAsk([("garbage", 0.01), (shots_answer(2), 0.02), (shots_answer(2), 0.02),
                       (grammar_answer(), 0.005)])
        result, progress = self.run_analysis(ask)
        self.assertEqual([c["videos"] for c in ask.calls], [1, 1, 1, 0])
        self.assertEqual(ask.calls[-1]["router"], sr.TEXT_ROUTER)
        self.assertIn("Measured editing statistics", ask.calls[-1]["prompt"])
        self.assertIn("Shot list of this excerpt (2 shots)", ask.calls[0]["prompt"])
        self.assertAlmostEqual(result["cost_usd"], 0.055)
        self.assertEqual(len(result["shots"]), 4)
        self.assertTrue(all(s.get("shot_size") for s in result["shots"]))
        self.assertEqual(result["stats"]["shot_count"], 4)
        # Measured shot mix replaces the model's estimate.
        self.assertEqual(result["grammar"]["shot_mix"], {"close_up": 50.0, "medium": 50.0})
        self.assertEqual(result["chunk_errors"], [])
        self.assertIn("stats", progress[0][1])

    def test_a_failed_chunk_is_recorded_and_text_router_falls_back(self):
        ask = FakeAsk([(shots_answer(2), 0.02), JudgeError("fal.ai error 500: down"),
                       JudgeError("fal.ai error 404: no such app"), (grammar_answer(), 0.01)])
        result, _ = self.run_analysis(ask)
        self.assertEqual(len(result["chunk_errors"]), 1)
        self.assertEqual([c["router"] for c in ask.calls[-2:]], [sr.TEXT_ROUTER, FAL_ROUTER])
        self.assertEqual(sum(1 for s in result["shots"] if s.get("shot_size")), 2)

    def test_no_key_keeps_stats_and_raises(self):
        progress = []
        with self.assertRaises(sr.StyleReferenceError):
            sr.analyse_reference("v.mp4", fal_key="", ask=FakeAsk([]), local=fake_local([1.0], 2.0),
                                 on_progress=lambda m, p: progress.append(p))
        self.assertEqual(progress[0]["stats"]["shot_count"], 2)

    def test_local_backend_sends_stills_for_nothing(self):
        class Backend:
            provider = "openai"
            available = True
            model = ""

            def __init__(self):
                self.ask = FakeAsk([(shots_answer(2), 0.0), (shots_answer(2), 0.0), (grammar_answer(), 0.0)])

        backend = Backend()
        made = []

        def fake_stills(source, chunk, shots, dest_dir, **kw):
            stills = [(n, float(shots[i]["start"]) + 0.5, os.path.join(dest_dir, f"{n}.jpg"))
                      for n, i in enumerate(chunk["shots"], 1)]
            made.append(len(stills))
            return stills

        with patch.object(sr, "shot_stills", fake_stills), \
                patch.object(sr, "plan_local_chunks", lambda shots, **kw: sr.plan_chunks(shots, max_sec=60)):
            result, _ = self.run_analysis(backend.ask, fal_key="fal-key", backend=backend)
        self.assertEqual(made, [2, 2])
        # The local server authenticates with its own key, never fal's.
        self.assertEqual({c["key"] for c in backend.ask.calls}, {""})
        self.assertEqual([c["videos"] for c in backend.ask.calls], [2, 2, 0])
        self.assertIn("Image 1 = shot 1 at 0.5 s", backend.ask.calls[0]["prompt"])
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertEqual(result["model"], sr.LOCAL_MODEL)
        self.assertTrue(all(s.get("shot_size") for s in result["shots"]))

    def test_local_backend_without_a_url_keeps_stats_and_raises(self):
        class Backend:
            provider = "openai"
            available = False
            model = "analyst"
            ask = FakeAsk([])

        with self.assertRaises(sr.StyleReferenceError):
            sr.analyse_reference("v.mp4", fal_key="", backend=Backend(), local=fake_local([1.0], 2.0))

    def test_bad_synthesis_twice_fails(self):
        ask = FakeAsk([(shots_answer(2), 0.0), (shots_answer(2), 0.0), ("nope", 0.0), ("still nope", 0.0)])
        with self.assertRaises(sr.StyleReferenceError):
            self.run_analysis(ask)


def reference(ref_id="ref_a", label="Argument scene", tags=(), rules=None, status="done", enabled=True):
    return {
        "id": ref_id, "label": label, "status": status, "enabled": enabled, "scene_tags": list(tags),
        "stats": {"asl_sec": 2.6, "median_sec": 2.1, "cuts_per_minute": 23.0,
                  "distribution": [{"bucket": "<1s", "percent": 10.0}, {"bucket": "1-2s", "percent": 40.0}]},
        "grammar": {"rhythm": {"target_asl_sec": 2.4, "pace_with_emotion": "Faster as tension rises."},
                    "shot_mix": {"close_up": 45.0, "medium": 30.0}, "patterns": ["Reaction after each line"],
                    "dialogue_coverage": "OTS pairs", "rules": rules or ["Hold reactions 2-3 s.",
                                                                          "Open on an insert."]},
    }


class PlannerBlockTests(unittest.TestCase):
    def test_block_carries_rhythm_and_rules(self):
        block = sr.planner_block([reference()])
        self.assertIn("REFERENCE STAGING GRAMMAR", block)
        self.assertIn("never copy", block)
        self.assertIn("target average shot 2.4 s", block)
        self.assertIn("23 cuts/min", block)
        self.assertIn("- Rule: Hold reactions 2-3 s.", block)
        self.assertIn("close-up 45%", block)

    def test_block_is_bounded(self):
        long_rules = [f"Rule number {n} " + "x" * 300 for n in range(10)]
        refs = [reference(f"ref_{n}", f"Ref {n}", rules=long_rules) for n in range(6)]
        single = sr.reference_block(refs[0])
        self.assertLessEqual(len(single), sr.MAX_REFERENCE_CHARS)
        self.assertIn("Rule: Rule number 0", single)
        block = sr.planner_block(refs)
        self.assertLessEqual(len(block), sr.MAX_BLOCK_CHARS)
        self.assertIn("Ref 0", block)

    def test_tagged_references_matching_the_brief_come_first(self):
        refs = [reference("ref_r", "Romance", tags=["romance"]), reference("ref_u", "Any"),
                reference("ref_a", "Fight", tags=["argument"])]
        block = sr.planner_block(refs, brief="Two sisters argue about the inheritance.")
        self.assertLess(block.index("Fight"), block.index("Any"))
        self.assertLess(block.index("Any"), block.index("Romance"))
        self.assertIn("for scenes: argument", block)
        self.assertIn("names the scenes it guides", block)

    def test_selection(self):
        bible = {"style_references": [reference("ref_a"), reference("ref_b", enabled=False),
                                      reference("ref_c", status="failed"), reference("ref_d", rules=[""])]}
        bible["style_references"][3]["grammar"]["rules"] = []
        self.assertEqual([r["id"] for r in sr.select_references(bible, "auto")], ["ref_a"])
        self.assertEqual([r["id"] for r in sr.select_references(bible, "")], ["ref_a"])
        self.assertEqual(sr.select_references(bible, "off"), [])
        self.assertEqual([r["id"] for r in sr.select_references(bible, "ref_b")], ["ref_b"])
        self.assertEqual(sr.select_references(bible, "ref_c"), [])
        self.assertEqual(sr.planner_block([]), "")


class BibleStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible = bibles.save_bible({"name": "Seoul Nights"})
        self.video = os.path.join(self.tmp.name, "ref.mp4")
        Path(self.video).write_bytes(b"fake video")

    def test_references_survive_a_text_save_and_restarts_mark_interrupted(self):
        ref = sr.copy_into_bible(self.bible["id"], self.video, label="Fight", scene_tags="Argument, tension")
        self.assertEqual(ref["scene_tags"], ["argument", "tension"])
        stored = bibles.get_bible(self.bible["id"])
        self.assertEqual(len(stored["style_references"]), 1)
        # A client PUT never carries (or drops) references.
        saved = bibles.save_bible({**stored, "style_references": []}, bible_id=self.bible["id"])
        self.assertEqual([r["id"] for r in saved["style_references"]], [ref["id"]])
        sr._set(self.bible["id"], ref["id"], status="analysing")
        self.assertEqual(sr.get_reference(self.bible["id"], ref["id"])["status"], "failed")
        self.assertEqual(sr.get_reference(self.bible["id"], ref["id"])["error"], "Interrupted")
        self.assertEqual(bibles.list_bibles()[0]["style_references"], 1)

    def test_edit_and_delete(self):
        ref = sr.copy_into_bible(self.bible["id"], self.video)
        with self.assertRaises(bibles.BibleError):
            sr.update_reference(self.bible["id"], ref["id"], {"rules": "x"})
        sr._set(self.bible["id"], ref["id"], status="done", grammar={"rules": ["a"], "rhythm": {}})
        edited = sr.update_reference(self.bible["id"], ref["id"], {
            "label": "New", "enabled": False, "scene_tags": ["romance"], "rules": "- One\n- Two\n"})
        self.assertEqual((edited["label"], edited["enabled"], edited["scene_tags"]), ("New", False, ["romance"]))
        self.assertEqual(edited["grammar"]["rules"], ["One", "Two"])
        path = bibles.video_path(self.bible["id"], ref["file"])
        sr.save_shots(self.bible["id"], ref, [{"index": 0}])
        self.assertEqual(sr.delete_reference(self.bible["id"], ref["id"]), [])
        self.assertFalse(path.exists())
        self.assertFalse(sr.shots_path(self.bible["id"], ref).exists())

    def test_pipeline_records_the_choice_and_the_block(self):
        from services import director_pipeline as pipeline

        ref = sr.copy_into_bible(self.bible["id"], self.video, label="Fight")
        sr._set(self.bible["id"], ref["id"], status="done", **{k: v for k, v in reference().items()
                                                               if k in ("stats", "grammar")})
        params = {"bible_id": self.bible["id"], "scene_description": "A quarrel."}
        block = pipeline._apply_style_references("p1", params)
        self.assertIn("REFERENCE STAGING GRAMMAR", block)
        self.assertEqual(params["director_style_reference"], "auto")
        self.assertEqual(params["_director_style_references"][0]["id"], ref["id"])
        self.assertEqual(params["_director_style_references"][0]["target_asl_sec"], 2.4)
        params = {"bible_id": self.bible["id"], "director_style_reference": "off"}
        self.assertEqual(pipeline._apply_style_references("p1", params), "")
        self.assertEqual(params["_director_style_references"], [])
        self.assertEqual(pipeline._apply_style_references("p1", {}), "")


class RouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible = bibles.save_bible({"name": "Seoul Nights"})
        api = FastAPI()
        sr.register_routes(api, get_fal_key=lambda: "test-key", get_model=lambda: "test/model")
        self.client = TestClient(api)
        # The uploads of these tests are not videos: no proxy.
        proxy = patch.object(sr, "ensure_proxy", lambda bible_id, reference, **kw: dict(reference))
        proxy.start()
        self.addCleanup(proxy.stop)
        self.base = f"/api/v1/director/bibles/{self.bible['id']}/style-references"

    def fake_analyse(self, path, *, fal_key, model, on_progress, backend=None, graphics="ignore", known_shots=None):
        self.seen = (fal_key, model)
        self.graphics = graphics
        on_progress("Measured", {"stats": {"asl_sec": 2.5, "shot_count": 2}})
        return {"stats": {"asl_sec": 2.5, "shot_count": 2, "cuts": [1.0], "shot_lengths": [1.0, 1.0]},
                "shots": [{"index": 0, "shot_size": "close_up"}, {"index": 1}],
                "grammar": {"rhythm": {"target_asl_sec": 2.5}, "rules": ["Cut on the look."]},
                "summary": "## Summary", "cost_usd": 0.04, "model": model, "chunk_errors": []}

    def wait_for(self, ref_id, states=("done", "failed")):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ref = self.client.get(f"{self.base}/{ref_id}").json()["reference"]
            if ref["status"] in states:
                return ref
            time.sleep(0.05)
        self.fail("analysis did not finish")

    def test_crud_with_a_mocked_model(self):
        with patch.object(sr, "analyse_reference", self.fake_analyse):
            response = self.client.post(self.base, files={"file": ("scene.mp4", b"fake video", "video/mp4")},
                                        data={"label": "Fight", "scene_tags": "argument, tension"})
            self.assertEqual(response.status_code, 200, response.text)
            ref = response.json()["reference"]
            self.assertIn(ref["status"], ("queued", "analysing", "done"))
            done = self.wait_for(ref["id"])
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(self.seen, ("test-key", "test/model"))
        self.assertEqual(done["cost_usd"], 0.04)
        self.assertEqual(done["described_shots"], 1)
        self.assertEqual(len(done["shots"]), 2)
        self.assertEqual(done["scene_tags"], ["argument", "tension"])

        listed = self.client.get(self.base).json()["references"]
        self.assertEqual([r["id"] for r in listed], [ref["id"]])
        self.assertNotIn("shots", listed[0])
        self.assertNotIn("cuts", listed[0]["stats"])
        self.assertNotIn("style_references", json.dumps(listed))

        video = self.client.get(f"{self.base}/{ref['id']}/video")
        self.assertEqual((video.status_code, video.content), (200, b"fake video"))

        edited = self.client.patch(f"{self.base}/{ref['id']}", json={"enabled": False, "rules": ["A", "B"]})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["reference"]["grammar"]["rules"], ["A", "B"])
        self.assertFalse(edited.json()["reference"]["enabled"])

        with patch.object(sr, "analyse_reference", self.fake_analyse):
            again = self.client.post(f"{self.base}/{ref['id']}/analyse")
            self.assertEqual(again.status_code, 200)
            self.assertEqual(self.wait_for(ref["id"])["grammar"]["rules"], ["Cut on the look."])

        deleted = self.client.delete(f"{self.base}/{ref['id']}")
        self.assertEqual(deleted.json(), {"deleted": ref["id"], "references": []})
        self.assertEqual(self.client.get(f"{self.base}/{ref['id']}").status_code, 404)

    def test_failures_are_recorded(self):
        def broken(path, **kwargs):
            kwargs["on_progress"]("Measured", {"stats": {"asl_sec": 3.0}})
            raise sr.StyleReferenceError("model down")

        with patch.object(sr, "analyse_reference", broken):
            ref = self.client.post(self.base, files={"file": ("a.mov", b"x", "video/quicktime")}).json()["reference"]
            failed = self.wait_for(ref["id"])
        self.assertEqual((failed["status"], failed["error"]), ("failed", "model down"))
        self.assertEqual(failed["stats"]["asl_sec"], 3.0)

    def test_upload_validation(self):
        bad = self.client.post(self.base, files={"file": ("notes.txt", b"x", "text/plain")})
        self.assertEqual(bad.status_code, 400)
        empty = self.client.post(self.base, files={"file": ("a.mp4", b"", "video/mp4")})
        self.assertEqual(empty.status_code, 400)
        unknown = self.client.post("/api/v1/director/bibles/nope/style-references",
                                   files={"file": ("a.mp4", b"x", "video/mp4")})
        self.assertEqual(unknown.status_code, 404)
        with patch.object(sr, "MAX_UPLOAD_BYTES", 4):
            big = self.client.post(self.base, files={"file": ("a.mp4", b"0123456789", "video/mp4")},
                                   data={"analyse": "false"})
        self.assertEqual(big.status_code, 413)
        self.assertEqual(self.client.get(self.base).json()["references"], [])
        leftovers = [p.name for p in bibles._folder(self.bible["id"]).iterdir() if p.name.startswith("styleref_")]
        self.assertEqual(leftovers, [])
        self.assertEqual(self.client.get(f"{self.base}/missing").status_code, 404)
        self.assertEqual(self.client.patch(f"{self.base}/missing", json={"label": "x"}).status_code, 404)



class CreditsTests(unittest.TestCase):
    def test_flagged_shots_leave_the_stats_and_the_synthesis(self):
        shots = sr.shot_spans([10.0, 20.0, 30.0], 40.0)
        shots[0]["credits"] = True
        shots[3]["credits"] = True
        kept, trimmed = sr.without_credits(shots, 40.0)
        self.assertEqual([s["index"] for s in kept], [1, 2])
        self.assertEqual((trimmed["credits_shots"], trimmed["credits_sec"], trimmed["shot_count"]), (2, 20.0, 2))
        self.assertEqual(trimmed["asl_sec"], 10.0)
        self.assertEqual(sr.without_credits(sr.shot_spans([10.0], 20.0), 20.0)[1], {})

    def test_parse_keeps_the_credits_flag(self):
        answer = json.dumps({"shots": [{"shot": 1, "shot_size": "wide", "credits": True},
                                       {"shot": 2, "shot_size": "close_up", "credits": False}]})
        records = sr.parse_chunk_shots(answer, {"shots": [0, 1]})
        self.assertTrue(records[0].get("credits"))
        self.assertNotIn("credits", records[1])

    def test_analysis_drops_credits_from_the_grammar(self):
        first = json.dumps({"shots": [{"shot": 1, "shot_size": "wide", "credits": True},
                                      {"shot": 2, "shot_size": "close_up"}]})
        ask = FakeAsk([(first, 0.0), (shots_answer(2), 0.0), (grammar_answer(), 0.0)])
        progress = []
        with patch.object(sr, "encode_chunk", lambda src, a, b, dest, **kw: dest), \
                patch.object(sr, "data_uri", lambda path: "data:video/mp4;base64,AAAA"):
            result = sr.analyse_reference("video.mp4", fal_key="key", ask=ask, http=object(), sleep=lambda s: None,
                                          local=fake_local([30.0, 50.0, 80.0], 100.0),
                                          on_progress=lambda m, p: progress.append(m))
        self.assertEqual(result["stats"]["credits_shots"], 1)
        self.assertEqual(result["stats"]["shot_count"], 3)
        self.assertTrue(result["shots"][0].get("credits"))
        self.assertNotIn("1 | 0.0s", ask.calls[-1]["prompt"])  # the credits shot is not in the breakdown
        self.assertIn("2 | 30.0s", ask.calls[-1]["prompt"])
        self.assertTrue(any(m.startswith("Left out 1 credits shots") for m in progress))


def merged_answer(**extra) -> str:
    body = json.loads(grammar_answer())
    body["rules"] = [f"Rule {n}." for n in range(1, 15)]
    body["scene_notes"] = {"dispute": "Cut faster, 1.5 s per shot.", "Confession!": "Hold the listener."}
    body.update(extra)
    return json.dumps(body)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible = bibles.save_bible({"name": "Seoul Nights"})
        self.bid = self.bible["id"]

    def put_references(self, references):
        folder = bibles._folder(self.bid)
        folder.mkdir(parents=True, exist_ok=True)
        for r in references:
            (folder / f"{r['id']}.mp4").write_bytes(b"video")

        def change(bible):
            bible["style_references"] = [{"file": f"{r['id']}.mp4", "updated_at": "t0", **r} for r in references]
        bibles._mutate(self.bid, change)
        return bibles.get_bible(self.bid)

    def fake_merge(self, references, *, fal_key, model, backend=None):
        self.merged_from = [r["id"] for r in references]
        grammar, summary = sr.parse_grammar(merged_answer(), max_rules=sr.MERGED_RULES, scene_notes=True)
        return {"grammar": grammar, "summary": summary, "cost_usd": 0.0, "model": "analyst"}

    def test_parse_grammar_keeps_fourteen_rules_and_scene_notes(self):
        grammar, _ = sr.parse_grammar(merged_answer(), max_rules=sr.MERGED_RULES, scene_notes=True)
        self.assertEqual(len(grammar["rules"]), 14)
        self.assertEqual(grammar["scene_notes"], {"dispute": "Cut faster, 1.5 s per shot.",
                                                  "confession": "Hold the listener."})
        self.assertNotIn("scene_notes", sr.parse_grammar(merged_answer())[0])

    def test_merge_grammars_is_one_text_call_on_the_backend(self):
        class Backend:
            provider = "openai"
            available = True
            model = "analyst"
            ask = FakeAsk([("nope", 0.0), (merged_answer(), 0.0)])

        backend = Backend()
        result = sr.merge_grammars([reference("ref_a", tags=["dispute"]), reference("ref_b")], fal_key="",
                                   backend=backend, http=object(), sleep=lambda s: None)
        self.assertEqual([c["videos"] for c in backend.ask.calls], [0, 0])
        self.assertIn("2 references to consolidate", backend.ask.calls[0]["prompt"])
        self.assertIn("for dispute scenes", backend.ask.calls[0]["prompt"])
        self.assertEqual(backend.ask.calls[0]["system"], sr.MERGE_SYSTEM_PROMPT)
        self.assertEqual((result["model"], result["cost_usd"], len(result["grammar"]["rules"])), ("analyst", 0.0, 14))

    def test_one_reference_needs_no_merge_and_drops_a_stale_one(self):
        self.put_references([reference("ref_a")])
        bibles._mutate(self.bid, lambda b: b.__setitem__("style_grammar", {"status": "done", "sources": []}))
        self.assertIsNone(sr.start_merge(self.bid, fal_key="k", merge=self.fake_merge, wait=True))
        self.assertIsNone(bibles.get_bible(self.bid)["style_grammar"])

    def test_merge_runs_once_per_set_of_references_and_goes_stale_on_edits(self):
        bible = self.put_references([reference("ref_a", tags=["dispute"]), reference("ref_b"),
                                     reference("ref_c", status="failed")])
        record = sr.start_merge(self.bid, fal_key="k", merge=self.fake_merge, wait=True)
        self.assertEqual(record["status"], "merging")
        stored = bibles.get_bible(self.bid)["style_grammar"]
        self.assertEqual(stored["status"], "done")
        self.assertEqual(self.merged_from, ["ref_a", "ref_b"])
        self.assertEqual([s["id"] for s in stored["sources"]], ["ref_a", "ref_b"])
        self.assertEqual(stored["grammar"]["scene_notes"]["dispute"], "Cut faster, 1.5 s per shot.")
        # Fresh: the same set of references reuses it.
        references = sr.merge_candidates(bibles.get_bible(self.bid))
        self.assertIsNotNone(sr.merged_grammar(bibles.get_bible(self.bid), references))
        self.merged_from = None
        self.assertEqual(sr.start_merge(self.bid, fal_key="k", merge=self.fake_merge, wait=True)["status"], "done")
        self.assertIsNone(self.merged_from)
        # An edit (tags) makes it stale until the next merge; the planner falls back to stacking meanwhile.
        sr.update_reference(self.bid, "ref_b", {"scene_tags": "confession"})
        bible = bibles.get_bible(self.bid)
        self.assertIsNone(sr.merged_grammar(bible, sr.merge_candidates(bible)))
        sr.start_merge(self.bid, fal_key="k", merge=self.fake_merge, wait=True)
        bible = bibles.get_bible(self.bid)
        self.assertEqual(self.merged_from, ["ref_a", "ref_b"])
        self.assertIsNotNone(sr.merged_grammar(bible, sr.merge_candidates(bible)))
        # A reference chosen by id never uses the merged grammar.
        self.assertIsNone(sr.merged_grammar(bible, sr.select_references(bible, "ref_a")))

    def test_a_failed_merge_is_recorded(self):
        self.put_references([reference("ref_a"), reference("ref_b")])

        def broken(references, **kw):
            raise sr.StyleReferenceError("model down")

        sr.start_merge(self.bid, fal_key="k", merge=broken, wait=True)
        stored = bibles.get_bible(self.bid)["style_grammar"]
        self.assertEqual((stored["status"], stored["error"]), ("failed", "model down"))

    def test_the_merged_grammar_survives_a_bible_save(self):
        self.put_references([reference("ref_a"), reference("ref_b")])
        sr.start_merge(self.bid, fal_key="k", merge=self.fake_merge, wait=True)
        bibles.save_bible({"name": "Seoul Nights 2"}, bible_id=self.bid)
        stored = bibles.get_bible(self.bid)["style_grammar"]
        self.assertEqual(stored["status"], "done")
        self.assertEqual(len(stored["grammar"]["rules"]), 14)


class MergedBlockTests(unittest.TestCase):
    def merged(self):
        grammar, summary = sr.parse_grammar(merged_answer(), max_rules=sr.MERGED_RULES, scene_notes=True)
        return {"status": "done", "grammar": grammar, "summary": summary}

    def test_the_merged_grammar_replaces_the_stack(self):
        references = [reference(f"ref_{n}", label=f"Ref {n}", tags=["dispute"] if n == 2 else ()) for n in range(6)]
        stacked = sr.planner_block(references)
        merged = sr.planner_block(references, merged=self.merged(), brief="A confession at the café.")
        self.assertIn("all 6 references combined", merged)
        self.assertNotIn("Ref 1", merged)
        self.assertNotIn("Each reference names the scenes", merged)
        self.assertIn("- Rule: Rule 14.", merged)
        self.assertIn("Ref 1", stacked)
        self.assertNotIn("Left out", merged)
        self.assertLessEqual(len(merged), sr.MAX_BLOCK_CHARS)
        # The note of the tag the brief names comes before the other one.
        self.assertLess(merged.index("For confession scenes"), merged.index("For dispute scenes"))

    def test_a_merged_record_without_rules_stacks(self):
        references = [reference("ref_a"), reference("ref_b", label="Second")]
        block = sr.planner_block(references, merged={"status": "done", "grammar": {"rules": []}})
        self.assertIn("Second", block)



class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
        self.folder = bibles._folder(self.bid)
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, name, data=b"video"):
        (self.folder / name).write_bytes(data)
        return sr.add_reference(self.bid, str(self.folder / name), source_filename=name)

    def test_a_large_upload_is_replaced_by_its_proxy_once(self):
        reference = self.add("styleref_1.mkv")
        made = []

        def fake_probe(path, **kw):
            return {"duration_sec": 90.0, "fps": 25.0, "width": 1920, "height": 1080, "has_audio": True}

        def fake_proxy(source, dest, **kw):
            made.append((os.path.basename(source), os.path.basename(dest), kw.get("height")))
            Path(dest).write_bytes(b"small")
            return dest

        with patch.object(sr, "probe", fake_probe), patch.object(sr, "make_proxy", fake_proxy):
            updated = sr.ensure_proxy(self.bid, reference)
            again = sr.ensure_proxy(self.bid, updated)
        self.assertEqual(made, [("styleref_1.mkv", "styleref_1_480p.mp4", 480)])
        self.assertEqual(updated["file"], "styleref_1_480p.mp4")
        self.assertEqual(updated["source"], {"width": 1920, "height": 1080, "bytes": 5, "duration_sec": 90.0})
        self.assertFalse((self.folder / "styleref_1.mkv").exists())
        self.assertTrue((self.folder / "styleref_1_480p.mp4").exists())
        self.assertEqual(again["file"], "styleref_1_480p.mp4")
        stored = sr.get_reference(self.bid, reference["id"])
        self.assertEqual((stored["file"], stored["source"]["height"]), ("styleref_1_480p.mp4", 1080))

    def test_a_small_mp4_is_kept(self):
        reference = self.add("styleref_2.mp4")
        with patch.object(sr, "probe", lambda path, **kw: {"duration_sec": 5.0, "width": 854, "height": 480,
                                                           "has_audio": False}), \
                patch.object(sr, "make_proxy", lambda *a, **kw: self.fail("no proxy needed")):
            updated = sr.ensure_proxy(self.bid, reference)
        self.assertEqual(updated["file"], "styleref_2.mp4")
        self.assertEqual(updated["source"]["height"], 480)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg needed")
    def test_make_proxy_scales_to_480p_and_keeps_mono_sound(self):
        source = os.path.join(self.tmp.name, "big.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=1:r=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-c:a", "aac", "-shortest", source], check=True, capture_output=True)
        dest = sr.make_proxy(source, os.path.join(self.tmp.name, "small.mp4"))
        facts = sr.probe(dest)
        self.assertEqual((facts["width"], facts["height"], facts["has_audio"]), (854, 480, True))



class GraphicsTests(unittest.TestCase):
    def analyse(self, graphics, shots_json):
        ask = FakeAsk([(shots_json, 0.0), (shots_answer(2), 0.0), (grammar_answer(graphics="Captions in the lower third."), 0.0)])
        with patch.object(sr, "encode_chunk", lambda src, a, b, dest, **kw: dest), \
                patch.object(sr, "data_uri", lambda path: "data:video/mp4;base64,AAAA"):
            result = sr.analyse_reference("video.mp4", fal_key="key", ask=ask, http=object(), sleep=lambda s: None,
                                          local=fake_local([30.0, 50.0, 80.0], 100.0), graphics=graphics)
        return ask, result

    def test_ignore_tells_the_analyst_to_leave_graphics_out(self):
        answer = json.dumps({"shots": [{"shot": 1, "shot_size": "wide", "graphics": "channel logo"},
                                       {"shot": 2, "shot_size": "close_up", "graphics": "none"}]})
        ask, result = self.analyse("ignore", answer)
        self.assertIn("are NOT part of the staging", ask.calls[0]["system"])
        self.assertNotIn("graphics column", ask.calls[-1]["system"])
        self.assertNotIn("| graphics", ask.calls[-1]["prompt"])
        self.assertEqual(result["shots"][0].get("graphics"), "channel logo")  # described, kept for the user
        self.assertNotIn("graphics", result["shots"][1])

    def test_include_adds_the_graphics_column_and_section(self):
        answer = json.dumps({"shots": [{"shot": 1, "shot_size": "wide", "graphics": "caption top left"},
                                       {"shot": 2, "shot_size": "close_up"}]})
        ask, result = self.analyse("include", answer)
        self.assertIn("On-screen graphics are part of the grammar", ask.calls[0]["system"])
        self.assertIn("graphics column", ask.calls[-1]["system"])
        self.assertIn("| graphics", ask.calls[-1]["prompt"])
        self.assertIn("| caption top left", ask.calls[-1]["prompt"])
        self.assertEqual(result["grammar"]["graphics"], "Captions in the lower third.")
        block = sr.reference_block({"label": "Vlog", "grammar": result["grammar"], "stats": None, "scene_tags": []},
                                   3000)
        self.assertIn("- Graphics: Captions in the lower third.", block)

    def test_parse_graphics_and_the_field_on_the_reference(self):
        self.assertEqual([sr.parse_graphics(v) for v in (True, False, "include", "IGNORE", "yes", "", None)],
                         ["include", "ignore", "include", "ignore", "include", "ignore", "ignore"])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles"):
            bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
            folder = bibles._folder(bid)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "styleref_1.mp4").write_bytes(b"video")
            reference = sr.add_reference(bid, str(folder / "styleref_1.mp4"), graphics="include")
            self.assertEqual(reference["graphics"], "include")
            self.assertEqual(sr.get_reference(bid, reference["id"])["graphics"], "include")
            updated = sr.update_reference(bid, reference["id"], {"graphics": False})
            self.assertEqual(updated["graphics"], "ignore")



class FakeFetchHttp:
    """The video fetch service as the controller sees it: one job, a few polls, a file."""

    def __init__(self, polls=("downloading", "done"), fail_status=None, file=b"mp4 bytes"):
        self.polls = list(polls)
        self.fail_status = fail_status
        self.file = file
        self.calls = []

    class _Response:
        def __init__(self, status_code, body=None, text=""):
            self.status_code, self._body, self.text = status_code, body, text

        def json(self):
            return self._body

    def post(self, url, headers=None, json=None, **kw):
        self.calls.append(("POST", url, headers, json))
        if self.fail_status:
            return self._Response(self.fail_status, text="no")
        return self._Response(200, {"job": {"id": "job1", "status": "queued"}})

    def get(self, url, headers=None, **kw):
        self.calls.append(("GET", url, headers, None))
        status = self.polls.pop(0) if self.polls else "done"
        body = {"job": {"id": "job1", "status": status, "progress": "Downloading 40%", "title": "Cafe scene",
                        "error": "blocked" if status == "failed" else ""}}
        return self._Response(200, body)

    def stream(self, method, url, headers=None, **kw):
        self.calls.append(("STREAM", url, headers, None))
        http = self

        class Stream:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_bytes(self, size):
                yield http.file
        return Stream()

    def delete(self, url, headers=None, **kw):
        self.calls.append(("DELETE", url, headers, None))
        return self._Response(200, {"deleted": "job1"})


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
        sr._FETCHES.clear()

    def test_a_link_becomes_a_reference_and_starts_its_analysis(self):
        http = FakeFetchHttp()
        started = []
        fetch = sr.fetch_reference(self.bid, "https://youtu.be/abc123", fetcher=("https://fetch.test/", "tok"),
                                   scene_tags="dispute", graphics="include", http=http, sleep=lambda s: None,
                                   start=lambda bible_id, ref_id: started.append((bible_id, ref_id)), wait=True)
        self.assertEqual(fetch["status"], "done")  # wait=True: the record after the thread
        done = sr.list_fetches(self.bid)[0]
        self.assertEqual((done["status"], done["error"]), ("done", ""))
        self.assertEqual(http.calls[0][:2], ("POST", "https://fetch.test/api/fetch"))
        self.assertEqual(http.calls[0][2], {"Authorization": "Bearer tok"})
        self.assertEqual(http.calls[0][3], {"url": "https://youtu.be/abc123", "max_height": 480})
        self.assertEqual([c[0] for c in http.calls[1:]], ["GET", "GET", "STREAM", "DELETE"])
        reference = sr.get_reference(self.bid, done["reference_id"])
        self.assertEqual((reference["label"], reference["scene_tags"], reference["graphics"]),
                         ("Cafe scene", ["dispute"], "include"))
        self.assertEqual((bibles._folder(self.bid) / reference["file"]).read_bytes(), b"mp4 bytes")
        self.assertEqual(started, [(self.bid, reference["id"])])

    def test_a_failed_download_is_recorded_and_leaves_no_file(self):
        http = FakeFetchHttp(polls=("downloading", "failed"))
        sr.fetch_reference(self.bid, "https://www.youtube.com/watch?v=x", fetcher=("https://fetch.test", ""),
                           http=http, sleep=lambda s: None, wait=True)
        done = sr.list_fetches(self.bid)[0]
        self.assertEqual(done["status"], "failed")
        self.assertIn("blocked", done["error"])
        self.assertEqual(sr.list_references(self.bid), [])
        self.assertEqual([p.name for p in bibles._folder(self.bid).glob("styleref_*")], [])

    def test_a_refused_job_and_bad_input(self):
        http = FakeFetchHttp(fail_status=400)
        sr.fetch_reference(self.bid, "https://youtu.be/abc", fetcher=("https://fetch.test", "t"), http=http,
                           sleep=lambda s: None, wait=True)
        self.assertIn("HTTP 400", sr.list_fetches(self.bid)[0]["error"])
        with self.assertRaises(bibles.BibleError):
            sr.fetch_reference(self.bid, "ftp://nope", fetcher=("https://fetch.test", "t"))
        with self.assertRaises(bibles.BibleError):
            sr.fetch_reference(self.bid, "https://youtu.be/abc", fetcher=None)

    def test_the_route_queues_a_fetch(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        api = FastAPI()
        seen = []
        sr.register_routes(api, get_fal_key=lambda: "k", get_fetcher=lambda: ("https://fetch.test", "t"))
        client = TestClient(api)
        base = f"/api/v1/director/bibles/{self.bid}/style-references"
        with patch.object(sr, "_run_fetch", lambda *a, **kw: seen.append(a[1]["url"])):
            response = client.post(f"{base}/from-url", json={"url": "https://youtu.be/abc", "label": "Café"})
            batch = client.post(f"{base}/from-url", json={
                "urls": "https://youtu.be/one\nhttps://youtu.be/two, https://youtu.be/one\nnot a link",
                "label": "ignored for a batch"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["fetch"]["status"], "queued")
        self.assertEqual(client.get(f"{base}/fetches").json()["fetches"][0]["label"], "Café")
        self.assertEqual(batch.status_code, 200)
        self.assertEqual([f["url"] for f in batch.json()["fetches"]], ["https://youtu.be/one", "https://youtu.be/two"])
        self.assertEqual([f["label"] for f in batch.json()["fetches"]], ["", ""])
        self.assertEqual(seen, ["https://youtu.be/abc", "https://youtu.be/one", "https://youtu.be/two"])
        self.assertEqual(client.post(f"{base}/from-url", json={"url": "nope"}).status_code, 400)
        self.assertEqual(client.post(f"{base}/from-url", json={"urls": ["nope", ""]}).status_code, 400)

    def test_parse_urls(self):
        self.assertEqual(sr.parse_urls(["https://a.test/1", " https://a.test/1 ", "ftp://x", "https://b.test/2"]),
                         ["https://a.test/1", "https://b.test/2"])
        self.assertEqual(sr.parse_urls("https://a.test/1\nhttps://b.test/2,https://c.test/3 words"),
                         ["https://a.test/1", "https://b.test/2", "https://c.test/3"])



class AutoNamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
        self.folder = bibles._folder(self.bid)
        self.folder.mkdir(parents=True, exist_ok=True)

    def add(self, name, **kw):
        (self.folder / name).write_bytes(b"video")
        return sr.add_reference(self.bid, str(self.folder / name), source_filename=name, **kw)

    @staticmethod
    def fake_analyse(path, *, fal_key, model, on_progress, backend=None, graphics="ignore", known_shots=None):
        grammar, summary = sr.parse_grammar(grammar_answer(label="A café confession", scene_tags=["Confession", "romance"]))
        return {"stats": {"asl_sec": 2.5, "shot_count": 2, "cuts": [1.0], "shot_lengths": [1.0, 1.0]},
                "shots": [{"index": 0, "shot_size": "close_up"}, {"index": 1, "shot_size": "medium"}],
                "grammar": grammar, "summary": summary, "cost_usd": 0.0, "model": model, "chunk_errors": []}

    def test_parse_grammar_keeps_the_suggested_name_and_tags(self):
        grammar, _ = sr.parse_grammar(grammar_answer(label="A café confession", scene_tags=["Confession", "romance"]))
        self.assertEqual((grammar["suggested_label"], grammar["suggested_tags"]), ("A café confession", ["confession", "romance"]))
        self.assertNotIn("suggested_label", sr.parse_grammar(grammar_answer())[0])

    def test_the_analysis_names_an_unlabelled_reference(self):
        reference = self.add("IMG_4521.mov")
        self.assertTrue(reference["label_auto"] and reference["tags_auto"])
        with patch.object(sr, "ensure_proxy", lambda bible_id, reference, **kw: dict(reference)):
            sr.start_analysis(self.bid, reference["id"], fal_key="k", analyse=self.fake_analyse, wait=True)
        stored = sr.get_reference(self.bid, reference["id"])
        self.assertEqual((stored["label"], stored["scene_tags"]), ("A café confession", ["confession", "romance"]))
        self.assertTrue(stored["label_auto"] and stored["tags_auto"])

    def test_the_users_own_label_and_tags_are_kept(self):
        reference = self.add("scene.mp4", label="Episode 3 café", scene_tags="dispute")
        self.assertFalse(reference["label_auto"] or reference["tags_auto"])
        with patch.object(sr, "ensure_proxy", lambda bible_id, reference, **kw: dict(reference)):
            sr.start_analysis(self.bid, reference["id"], fal_key="k", analyse=self.fake_analyse, wait=True)
        stored = sr.get_reference(self.bid, reference["id"])
        self.assertEqual((stored["label"], stored["scene_tags"]), ("Episode 3 café", ["dispute"]))
        # Typing tags later turns the automatic ones off for good.
        auto = self.add("other.mp4")
        edited = sr.update_reference(self.bid, auto["id"], {"scene_tags": "chase"})
        self.assertEqual((edited["tags_auto"], edited["label_auto"]), (False, True))



class LiveShotsTests(unittest.TestCase):
    def test_each_excerpt_reports_the_shots_so_far(self):
        ask = FakeAsk([(shots_answer(2), 0.0), (shots_answer(2), 0.0), (grammar_answer(), 0.0)])
        seen = []
        with patch.object(sr, "encode_chunk", lambda src, a, b, dest, **kw: dest), \
                patch.object(sr, "data_uri", lambda path: "data:video/mp4;base64,AAAA"):
            sr.analyse_reference("video.mp4", fal_key="key", ask=ask, http=object(), sleep=lambda s: None,
                                 local=fake_local([30.0, 50.0, 80.0], 100.0),
                                 on_progress=lambda m, p: seen.append((m, sum(1 for s in p.get("shots") or [] if s.get("shot_size")))))
        self.assertEqual([n for m, n in seen if m.startswith("Analysing excerpt") and n], [2, 4])

    def test_the_analysis_thread_saves_live_shots(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles"):
            bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
            folder = bibles._folder(bid)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "styleref_1.mp4").write_bytes(b"video")
            reference = sr.add_reference(bid, str(folder / "styleref_1.mp4"))
            seen = {}

            def analyse(path, *, fal_key, model, on_progress, backend=None, graphics="ignore", known_shots=None):
                on_progress("Analysing excerpt 1 of 2", {"shots": [{"index": 0, "start": 0.0, "end": 1.0,
                                                                    "duration": 1.0, "shot_size": "wide"},
                                                                   {"index": 1, "start": 1.0, "end": 2.0, "duration": 1.0}]})
                seen["live"] = (sr.load_shots(bid, reference), sr.get_reference(bid, reference["id"])["described_shots"])
                raise sr.StyleReferenceError("stop here")

            with patch.object(sr, "ensure_proxy", lambda bible_id, reference, **kw: dict(reference)):
                sr.start_analysis(bid, reference["id"], fal_key="k", analyse=analyse, wait=True)
            self.assertEqual(seen["live"][1], 1)
            self.assertEqual(seen["live"][0][0]["shot_size"], "wide")
            self.assertEqual(sr.get_reference(bid, reference["id"])["status"], "failed")

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg needed")
    def test_the_still_route_serves_a_cached_frame(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles"):
            bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
            folder = bibles._folder(bid)
            folder.mkdir(parents=True, exist_ok=True)
            make_color_video(str(folder / "styleref_1.mp4"), [("red", 1.0), ("blue", 1.0)], audio=False)
            reference = sr.add_reference(bid, str(folder / "styleref_1.mp4"))
            api = FastAPI()
            sr.register_routes(api, get_fal_key=lambda: "k")
            client = TestClient(api)
            base = f"/api/v1/director/bibles/{bid}/style-references/{reference['id']}"
            response = client.get(f"{base}/still?t=1.5")
            self.assertEqual((response.status_code, response.headers["content-type"]), (200, "image/jpeg"))
            self.assertTrue((sr.stills_dir(bid, reference) / "00000015.jpg").is_file())
            self.assertEqual(client.get(f"{base}/still?t=-1").status_code, 400)
            sr.delete_reference(bid, reference["id"])
            self.assertFalse(sr.stills_dir(bid, reference).exists())



class ResumeTests(unittest.TestCase):
    def test_described_excerpts_are_skipped_when_the_cuts_match(self):
        known = sr.shot_spans([30.0, 50.0, 80.0], 100.0)
        for shot in known[:2]:
            shot.update({"shot_size": "close_up", "subject": "a woman", "action": "turns"})
        ask = FakeAsk([(shots_answer(2), 0.0), (grammar_answer(), 0.0)])
        progress = []
        with patch.object(sr, "encode_chunk", lambda src, a, b, dest, **kw: dest), \
                patch.object(sr, "data_uri", lambda path: "data:video/mp4;base64,AAAA"):
            result = sr.analyse_reference("video.mp4", fal_key="key", ask=ask, http=object(), sleep=lambda s: None,
                                          local=fake_local([30.0, 50.0, 80.0], 100.0), known_shots=known,
                                          on_progress=lambda m, p: progress.append(m))
        self.assertEqual([c["videos"] for c in ask.calls], [1, 0])  # one excerpt sent, then the synthesis
        self.assertIn("Shot list of this excerpt (2 shots)", ask.calls[0]["prompt"])
        self.assertIn("Excerpt 2 of 2", ask.calls[0]["prompt"])
        self.assertEqual(result["shots"][0]["subject"], "a woman")
        self.assertTrue(all(s.get("shot_size") for s in result["shots"]))
        self.assertIn("Measured 4 shots, 2 already described", progress[0])

    def test_other_cuts_start_over(self):
        shots = sr.shot_spans([30.0, 50.0], 100.0)
        known = [dict(s, shot_size="wide") for s in sr.shot_spans([31.0, 50.0], 100.0)]
        self.assertEqual(sr.resume_shots(shots, known), 0)
        self.assertFalse(any(s.get("shot_size") for s in shots))
        self.assertEqual(sr.resume_shots(shots, known[:1]), 0)



def frame_answer(n: int) -> str:
    return json.dumps({"prompt": f"Frame {n}: a young woman in a beige coat sits at a café table, soft window light "
                                 "from the left, warm wood tones, shallow depth of field, quiet tension.",
                       "shot_size": "close_up" if n % 2 else "medium", "lighting": "soft window light",
                       "palette": ["beige", "warm wood", "cream"], "mood": "quiet tension",
                       "camera": {"lens_mm": "50mm" if n % 2 else 35, "aperture": "f/2", "format": "Super 35",
                                  "height": "eye level", "angle": "slightly low", "look": "soft roll-off, light grain"},
                       "lighting_setup": "soft key from a window camera left, warm practical back light",
                       "grade": "warm, lifted blacks", "aspect_ratio": "2.39:1"})


class FramesTests(unittest.TestCase):
    def shots(self):
        shots = sr.shot_spans([5.0, 5.2, 12.0], 20.0)
        for shot in shots:
            shot["shot_size"] = "medium"
        shots[3]["credits"] = True
        return shots

    def test_sampling_and_frame_times(self):
        shots = self.shots()
        self.assertEqual([s["index"] for s in sr.sample_shots(shots)], [0, 1, 2])
        self.assertEqual([s["index"] for s in sr.sample_shots(shots, 2)], [0, 1])
        self.assertEqual(sr.frame_times(shots[0]), [("first", 0.12), ("last", 4.88)])
        self.assertEqual(sr.frame_times(shots[1]), [("first", 5.1)])  # 0.2 s: one frame

    def test_describe_frames_asks_one_call_per_frame_and_resumes(self):
        class Backend:
            provider = "openai"
            available = True
            model = "analyst"
            ask = FakeAsk([("nope", 0.0)] + [(frame_answer(n), 0.0) for n in range(1, 6)])

        backend = Backend()
        stills = []
        known = [{"index": 0, "position": "first", "t": 0.12, "prompt": "kept from before", "shot_size": "wide"}]
        progress = []
        result = sr.describe_frames("v.mp4", self.shots(), fal_key="", backend=backend, http=object(), sleep=lambda s: None,
                                    known=known, on_progress=lambda m, f: progress.append((m, len(f))),
                                    still=lambda path, seconds, dest: stills.append(seconds) or dest)
        # shot 0 last, shot 1 (one frame), shot 2 (5.2-12 s) first and last = 4 frames; the first call failed once.
        self.assertEqual(stills, [4.88, 5.1, 5.32, 11.88])
        self.assertEqual(len(backend.ask.calls), 5)
        self.assertEqual(backend.ask.calls[0]["system"], sr.FRAME_SYSTEM_PROMPT)
        self.assertEqual(backend.ask.calls[0]["videos"], 1)
        self.assertEqual([(f["index"], f["position"]) for f in result["frames"]],
                         [(0, "first"), (0, "last"), (1, "first"), (2, "first"), (2, "last")])
        self.assertEqual(result["frames"][0]["prompt"], "kept from before")
        self.assertEqual(result["frames"][1]["palette"], ["beige", "warm wood", "cream"])
        self.assertEqual(result["frames"][1]["camera"], {"lens_mm": 50, "aperture": "f/2", "format": "Super 35",
                                                         "height": "eye level", "angle": "slightly low",
                                                         "look": "soft roll-off, light grain"})
        self.assertEqual((result["frames"][1]["grade"], result["frames"][1]["aspect_ratio"]), ("warm, lifted blacks", "2.39:1"))
        profile = sr.camera_profile(result["frames"])
        self.assertEqual((profile["frames"], profile["lens_mm_median"], sorted(profile["lenses_mm"]), profile["format"],
                          profile["aspect_ratio"]), (5, 50, [35, 50], "super 35", "2.39:1"))
        self.assertIn("lenses 35/50 mm (median 50), f/2, super 35, 2.39:1", sr.camera_profile_text(profile))
        block = sr.prompt_examples_block(result["frames"][1:3], profile=profile)
        self.assertIn("Camera profile of the references: lenses 35/50 mm", block)
        self.assertIn("(camera: 50 mm, f/2, Super 35, eye level, slightly low, 2.39:1; light: soft key", block)
        self.assertEqual((result["cost_usd"], result["errors"]), (0.0, []))
        self.assertEqual(progress[0], ("Describing frame 1 of 4", 1))

    def test_the_frames_thread_saves_the_sidecar_and_the_status(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles"):
            bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
            folder = bibles._folder(bid)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "styleref_1.mp4").write_bytes(b"video")
            reference = sr.add_reference(bid, str(folder / "styleref_1.mp4"))
            with self.assertRaises(bibles.BibleError):
                sr.start_frames(bid, reference["id"], fal_key="k")  # not analysed yet
            sr._set(bid, reference["id"], status="done")
            sr.save_shots(bid, reference, self.shots())

            def describe(path, shots, *, fal_key, model, backend, limit, known, on_progress):
                frames = [{"index": s["index"], "position": "first", "t": s["start"], "prompt": f"shot {s['index']}",
                           "shot_size": "medium"} for s in sr.sample_shots(shots, limit)]
                on_progress("Describing frame 1 of 3", frames[:1])
                return {"frames": frames, "cost_usd": 0.0, "model": model, "errors": []}

            sr.start_frames(bid, reference["id"], fal_key="k", describe=describe, limit=2, wait=True)
            stored = sr.get_reference(bid, reference["id"])
            self.assertEqual((stored["frames_status"], stored["frames_count"]), ("done", 2))
            self.assertEqual([f["prompt"] for f in sr.load_frames(bid, reference)], ["shot 0", "shot 1"])
            examples = sr.prompt_examples(bid, [stored])
            self.assertEqual([e["prompt"] for e in examples], ["shot 0", "shot 1"])
            block = sr.prompt_examples_block(examples)
            self.assertIn("REFERENCE IMAGE PROMPT STYLE", block)
            self.assertIn("- [medium] shot 0", block)
            self.assertEqual(sr.prompt_examples_block([]), "")
            sr.delete_reference(bid, reference["id"])
            self.assertFalse(sr.frames_path(bid, reference).exists())

    def test_examples_spread_over_shot_sizes(self):
        frames = [{"index": i, "position": "first", "prompt": f"p{i}", "shot_size": size}
                  for i, size in enumerate(["close_up", "close_up", "close_up", "wide", "medium", "insert"])]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles"):
            bid = bibles.save_bible({"name": "Seoul Nights"})["id"]
            folder = bibles._folder(bid)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "styleref_1.mp4").write_bytes(b"video")
            reference = sr.add_reference(bid, str(folder / "styleref_1.mp4"))
            sr.save_frames(bid, reference, frames)
            chosen = sr.prompt_examples(bid, [reference], limit=4)
        self.assertEqual([c["shot_size"] for c in chosen], ["close_up", "wide", "medium", "insert"])



class SynthesisBoundsTests(unittest.TestCase):
    def test_a_long_video_is_downsampled_for_the_synthesis(self):
        shots = sr.shot_spans([float(t) for t in range(3, 3000, 3)], 3000.0)
        for shot in shots:
            shot.update({"shot_size": "medium", "subject": "a man", "action": "x" * 200})
        stats = {"duration_sec": 3000.0, "shot_count": len(shots), "pacing": [{"t": t, "asl_sec": 3.0} for t in range(0, 3000, 10)]}
        prompt = sr.synthesis_prompt(stats, shots)
        lines = [l for l in prompt.split("\n") if " | " in l and not l.startswith("Shot breakdown")]
        self.assertEqual(len(lines), sr.MAX_SYNTHESIS_SHOTS)
        self.assertEqual(json.loads(prompt.split("\n")[1])["pacing"].__len__(), sr.MAX_SYNTHESIS_PACING)
        self.assertLess(len(prompt), 60000)  # ~15k tokens: fits one 32k slot with room for the answer


if __name__ == "__main__":
    unittest.main()
