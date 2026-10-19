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
        self.base = f"/api/v1/director/bibles/{self.bible['id']}/style-references"

    def fake_analyse(self, path, *, fal_key, model, on_progress, backend=None):
        self.seen = (fal_key, model)
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


if __name__ == "__main__":
    unittest.main()
