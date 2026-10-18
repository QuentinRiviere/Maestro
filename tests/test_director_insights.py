"""Director insights registry: ingestion, idempotence, summary filters, backfill, CLI."""
from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import insights  # noqa: E402

NOW = time.time()


def pipeline_state(pid, *, created_at, image_model, video_model, skill, drafts=0, skill_version=None):
    return {
        "pipeline_id": pid, "status": "completed", "created_at": created_at, "completed_at": created_at + 600,
        "image_model": image_model, "video_model": video_model, "skill_version": skill_version,
        "_params_snapshot": {"director_skill": skill, "bible": {"id": "noir-bible"},
                             "director_storyboard_drafts": drafts, "resolution": "1280x720"},
        "clips": [
            {"index": 0, "_director_scene_type": "dialogue", "_director_camera_framing": "close_up",
             "_director_camera_movement": "static", "_director_duration_sec": 5, "_director_keep_sec": [0, 4],
             "_director_dialogue_beats": [{"speaker": "Mina", "text": "Hi"}],
             "start_image_filename": "shot_1.png", "video_filename": "shot_1.mp4"},
            {"index": 1, "_director_scene_type": "action", "_director_camera_framing": "wide",
             "_director_camera_movement": "pan", "_director_duration_sec": 8, "_director_keep_sec": None,
             "_director_dialogue_beats": [], "start_image_filename": "shot_2.png", "video_filename": "shot_2.mp4"},
        ],
        "edit_file": None, "output_files": ["shot_1.mp4", "shot_2.mp4"],
    }


def storyboard_report(*, drafts):
    if drafts:
        return {"status": "completed", "mode": "drafts", "model": "judge-v", "provider": "fal", "cost_usd": 0.02,
                "shots": [
                    {"index": 0, "verdict": "pass", "issues": [], "fix_hint": "", "duplicate_of": None,
                     "image": "shot_1.png", "cost_usd": 0.01,
                     "drafts": {"count": 3, "resolution": "640x360",
                                "candidates": [
                                    {"round": 1, "candidate": 1, "seed": 1, "file": "d1.png", "verdict": "flagged",
                                     "issues": ["Extra fingers on the left hand"]},
                                    {"round": 1, "candidate": 2, "seed": 2, "file": "d2.png", "verdict": "pass",
                                     "issues": []},
                                    {"round": 1, "candidate": 3, "seed": 3, "file": "d3.png", "verdict": "flagged",
                                     "issues": ["extra fingers on the left hand", "Wrong wardrobe colour"]},
                                ],
                                "rounds": [{"round": 1, "best": 2, "hint": "", "fix_hint": ""}],
                                "chosen": {"candidate": 2}, "final": "shot_1.png"}},
                    {"index": 1, "verdict": "flagged", "issues": ["Camera too close for a wide shot"],
                     "fix_hint": "", "duplicate_of": None, "image": "shot_2.png", "cost_usd": 0.01,
                     "drafts": {"count": 3, "resolution": "640x360",
                                "candidates": [
                                    {"round": 1, "candidate": 1, "seed": 4, "file": "e1.png", "verdict": "flagged",
                                     "issues": ["Camera too close for a wide shot"]},
                                    {"round": 1, "candidate": 2, "seed": 5, "file": "e2.png", "verdict": "flagged",
                                     "issues": ["Missing the second character"]},
                                ],
                                "rounds": [{"round": 1, "best": 1, "hint": "", "fix_hint": "closer"},
                                           {"round": 2, "best": 1, "hint": "", "fix_hint": ""}],
                                "chosen": {"candidate": 1}, "final": "shot_2.png"}},
                ]}
    return {"status": "completed", "mode": "vision", "model": "judge-v", "provider": "openai", "cost_usd": 0,
            "shots": [
                {"index": 0, "verdict": "regenerated", "issues": ["Extra fingers on the left hand."],
                 "fix_hint": "", "duplicate_of": None, "image": "shot_1.png", "cost_usd": 0},
                {"index": 1, "verdict": "pass", "issues": [], "fix_hint": "", "duplicate_of": None,
                 "image": "shot_2.png", "cost_usd": 0},
            ]}


def review_report(score, verdict):
    return {
        "status": "completed", "model": "judge-clip", "backend": "fal", "cost_usd": 0.1, "score": score,
        "shots": [
            {"index": 0, "score": score, "summary": "ok",
             "issues": [{"at_sec": 1.0, "severity": "major", "category": "lip_sync", "fixable_by": "skill",
                         "description": "Lips do not move while Mina speaks"}]},
            {"index": 1, "score": score, "summary": "ok",
             "issues": [{"at_sec": 2.0, "severity": "minor", "category": "motion", "fixable_by": "planner",
                         "description": "Pan drifts upward"},
                        {"at_sec": 4.0, "severity": "minor", "category": "motion", "fixable_by": "none_model_limit",
                         "description": "Pan drifts upward."}]},
        ],
        "episode": {"status": "completed", "model": "judge-episode", "score": score, "verdict": verdict,
                    "scores": {"continuity": 7, "authenticity": 6},
                    "defects": [{"at_sec": 6.0, "shot": 2, "severity": "major", "criterion": "continuity",
                                 "root_cause": "model_limit", "description": "Jacket changes colour between shots"},
                                {"at_sec": 0.0, "shot": None, "severity": "minor", "criterion": "pacing",
                                 "root_cause": "planner", "description": "Opening lingers too long"}],
                    "skill_fixes": [], "planner_fixes": [], "pipeline_fixes": [], "source": "joined"},
    }


def write_run(directory, pid, *, state, storyboard=None, review=None):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"_director_pipeline_{pid}.json"), "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    if storyboard is not None:
        with open(os.path.join(directory, f"_director_storyboard_{pid}.json"), "w", encoding="utf-8") as handle:
            json.dump(storyboard, handle)
    if review is not None:
        with open(os.path.join(directory, f"_director_review_{pid}.json"), "w", encoding="utf-8") as handle:
            json.dump(review, handle)


class InsightsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        # Run A: a recent draft run, fully reviewed.
        write_run(self.dir, "runA",
                  state=pipeline_state("runA", created_at=NOW - 3600, image_model="qwen", video_model="h3",
                                       skill="noir", drafts=3),
                  storyboard=storyboard_report(drafts=True), review=review_report(7.5, "ship"))
        # Run B: an old judge-only run in a workspace, reviewed.
        write_run(os.path.join(self.dir, "ws1"), "runB",
                  state=pipeline_state("runB", created_at=NOW - 40 * 86400, image_model="klein", video_model="ltx",
                                       skill="vlog"),
                  storyboard=storyboard_report(drafts=False), review=review_report(5.0, "rework"))

    def rows(self, table, pid=None):
        connection = sqlite3.connect(insights.db_path(self.dir))
        try:
            connection.row_factory = sqlite3.Row
            query = f"SELECT * FROM {table}" + (" WHERE pid = ?" if pid else "")
            return [dict(r) for r in connection.execute(query, (pid,) if pid else ())]
        finally:
            connection.close()

    def test_ingest_run_records_runs_shots_and_findings(self):
        counts = insights.ingest_run(self.dir, "runA")
        self.assertEqual((counts["runs"], counts["shots"], counts["findings"]), (1, 2, 10))
        self.assertEqual(counts["by_source"], {"storyboard": 0, "draft": 5, "clip": 3, "episode": 2})
        run = self.rows("runs", "runA")[0]
        self.assertEqual(run["skill"], "noir")
        self.assertEqual(run["bible"], "noir-bible")
        self.assertEqual((run["image_model"], run["video_model"], run["drafts_count"]), ("qwen", "h3", 3))
        self.assertEqual((run["storyboard_judge_model"], run["storyboard_judge_provider"]), ("judge-v", "fal"))
        self.assertEqual((run["review_backend"], run["review_model"]), ("fal", "judge-clip"))
        self.assertEqual((run["episode_score"], run["episode_verdict"]), (7.5, "ship"))
        self.assertEqual(json.loads(run["scores_json"]), {"continuity": 7, "authenticity": 6})

        shots = sorted(self.rows("shots", "runA"), key=lambda s: s["index"])
        self.assertEqual(shots[0]["scene_type"], "dialogue")
        self.assertEqual(shots[0]["framing"], "close_up")
        self.assertEqual(shots[0]["has_dialogue"], 1)
        self.assertEqual(json.loads(shots[0]["keep_json"]), [0, 4])
        self.assertEqual((shots[0]["draft_candidates"], shots[0]["draft_rounds"], shots[0]["draft_flagged"]),
                         (3, 1, 2))
        self.assertEqual(shots[0]["storyboard_verdict"], "pass")
        self.assertEqual(shots[1]["has_dialogue"], 0)
        self.assertIsNone(shots[1]["keep_json"])
        self.assertEqual((shots[1]["draft_candidates"], shots[1]["draft_rounds"], shots[1]["draft_flagged"]),
                         (2, 2, 2))
        self.assertEqual(shots[1]["clip_score"], 7.5)

        findings = self.rows("findings", "runA")
        episode = sorted((f for f in findings if f["source"] == "episode"), key=lambda f: f["description"])
        self.assertEqual(episode[0]["shot_index"], 1)
        self.assertEqual((episode[0]["category"], episode[0]["severity"], episode[0]["root_cause"]),
                         ("continuity", "major", "model_limit"))
        self.assertEqual(episode[0]["scene_type"], "action")
        self.assertEqual(episode[0]["judge_model"], "judge-episode")
        self.assertIsNone(episode[1]["shot_index"])
        self.assertIsNone(episode[1]["scene_type"])
        clip = next(f for f in findings if f["source"] == "clip" and f["shot_index"] == 0)
        self.assertEqual((clip["category"], clip["fixable_by"], clip["stage"]), ("lip_sync", "skill", "video"))
        self.assertEqual((clip["image_model"], clip["video_model"]), ("qwen", "h3"))
        draft = next(f for f in findings if f["source"] == "draft")
        self.assertEqual((draft["category"], draft["stage"]), ("storyboard", "image"))

    def test_judge_mode_report_records_storyboard_source(self):
        counts = insights.ingest_run(self.dir, "runB")
        self.assertEqual(counts["by_source"], {"storyboard": 1, "draft": 0, "clip": 3, "episode": 2})
        self.assertEqual(self.rows("runs", "runB")[0]["storyboard_judge_provider"], "openai")
        self.assertEqual([s["storyboard_verdict"] for s in sorted(self.rows("shots", "runB"),
                                                                 key=lambda s: s["index"])],
                         ["regenerated", "pass"])

    def test_ingest_is_idempotent(self):
        insights.ingest_run(self.dir, "runA")
        insights.ingest_run(self.dir, "runA")
        self.assertEqual(len(self.rows("runs")), 1)
        self.assertEqual(len(self.rows("shots")), 2)
        self.assertEqual(len(self.rows("findings")), 10)

    def test_run_without_review_is_recorded(self):
        write_run(self.dir, "runC",
                  state=pipeline_state("runC", created_at=NOW, image_model="qwen", video_model="h3", skill="noir"),
                  storyboard=storyboard_report(drafts=False))
        counts = insights.ingest_run(self.dir, "runC")
        self.assertEqual((counts["shots"], counts["findings"], counts["has_review"]), (2, 1, False))
        run = self.rows("runs", "runC")[0]
        self.assertIsNone(run["episode_score"])
        self.assertIsNone(run["review_model"])
        self.assertIsNone(self.rows("shots", "runC")[0]["clip_score"])

    def test_missing_run_raises(self):
        with self.assertRaises(FileNotFoundError):
            insights.ingest_run(self.dir, "nope")

    def test_summary_before_any_ingest_is_empty(self):
        result = insights.summary(self.dir)
        self.assertEqual(result["runs"], 0)
        self.assertEqual(result["top_descriptions"], [])
        self.assertFalse(os.path.exists(insights.db_path(self.dir)))

    def test_summary_aggregates_runs(self):
        insights.backfill(self.dir)
        result = insights.summary(self.dir)
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["average_episode_score"], 6.25)
        self.assertEqual([t["pid"] for t in result["trend"]], ["runB", "runA"])
        self.assertEqual(result["trend"][1]["verdict"], "ship")
        self.assertEqual(result["models"], ["h3", "klein", "ltx", "qwen"])

        by_category = {(r["source"], r["category"]): r["count"] for r in result["findings_by_category"]}
        self.assertEqual(by_category[("clip", "motion")], 4)
        self.assertEqual(by_category[("clip", "lip_sync")], 2)
        self.assertEqual(by_category[("episode", "continuity")], 2)
        self.assertEqual(by_category[("draft", "storyboard")], 5)
        self.assertEqual(by_category[("storyboard", "storyboard")], 1)

        by_scene = {(r["scene_type"], r["source"]): r["count"] for r in result["findings_by_scene_type"]}
        self.assertEqual(by_scene[("action", "clip")], 4)
        self.assertEqual(by_scene[("dialogue", "draft")], 3)

        top = result["top_descriptions"]
        self.assertEqual(top[0]["description"], "pan drifts upward")
        self.assertEqual(top[0]["count"], 4)
        self.assertEqual(top[0]["categories"], ["motion"])
        self.assertEqual(sorted(top[0]["pids"]), ["runA", "runB"])
        fingers = next(t for t in top if t["description"] == "extra fingers on the left hand")
        self.assertEqual(fingers["count"], 3)  # two draft candidates of A + the judge issue of B, normalised

        self.assertEqual(result["draft_stats"], [
            {"scene_type": "dialogue", "shots": 1, "candidates": 3, "flagged": 2, "flagged_ratio": 0.667,
             "rounds": 1.0},
            {"scene_type": "action", "shots": 1, "candidates": 2, "flagged": 2, "flagged_ratio": 1.0,
             "rounds": 2.0},
        ])

    def test_summary_filters(self):
        insights.backfill(self.dir)
        recent = insights.summary(self.dir, days=7)
        self.assertEqual([t["pid"] for t in recent["trend"]], ["runA"])
        self.assertEqual(recent["average_episode_score"], 7.5)
        self.assertEqual(recent["models"], ["h3", "qwen"])
        self.assertEqual(sum(r["count"] for r in recent["findings_by_category"]), 10)
        self.assertTrue(all(t["pids"] == ["runA"] for t in recent["top_descriptions"]))

        by_video = insights.summary(self.dir, video_model="ltx")
        self.assertEqual([t["pid"] for t in by_video["trend"]], ["runB"])
        self.assertEqual(by_video["draft_stats"], [])
        by_skill = insights.summary(self.dir, skill="noir", image_model="qwen")
        self.assertEqual(by_skill["runs"], 1)
        self.assertEqual(insights.summary(self.dir, skill="noir", image_model="klein")["runs"], 0)

    def test_backfill_walks_workspaces_and_reports_failures(self):
        with open(os.path.join(self.dir, "_director_pipeline_broken.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        counts = insights.backfill(self.dir)
        self.assertEqual((counts["runs"], counts["shots"], counts["findings"]), (2, 4, 16))
        self.assertEqual([f["pid"] for f in counts["failed"]], ["broken"])
        self.assertEqual(sorted(r["pid"] for r in self.rows("runs")), ["runA", "runB"])

    def test_cli_prints_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            insights.main(["backfill", self.dir])
        self.assertEqual(json.loads(out.getvalue())["runs"], 2)
        out = io.StringIO()
        with redirect_stdout(out):
            insights.main(["summary", self.dir, "--days", "7"])
        self.assertEqual(json.loads(out.getvalue())["runs"], 1)


if __name__ == "__main__":
    unittest.main()


class SkillVersionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        scores = {1: (6.0, 6.0, 7.0), 2: (4.0, 5.0, 4.0)}
        for version, episode_scores in scores.items():
            for n, score in enumerate(episode_scores):
                pid = f"v{version}r{n}"
                write_run(self.dir, pid,
                          state=pipeline_state(pid, created_at=NOW - 3600 * (10 - version * 3 - n), image_model="qwen",
                                               video_model="ltx", skill="custom:k_drama", skill_version=version),
                          storyboard=storyboard_report(drafts=True), review=review_report(score, "needs_polish"))
                insights.ingest_run(self.dir, pid)

    def test_versions_are_compared(self):
        versions = insights.skill_versions(self.dir, "custom:k_drama")
        self.assertEqual([(v["version"], v["reviewed"], v["episode_mean"]) for v in versions],
                         [(1, 3, 6.33), (2, 3, 4.33)])
        self.assertEqual(insights.skill_versions(self.dir, "custom:other"), [])

    def test_a_worse_current_version_is_reverted(self):
        decision = insights.rollback_decision(self.dir, "custom:k_drama")
        self.assertEqual((decision["current"], decision["revert_to"]), (2, 1))
        self.assertIn("version 1 scored 6.33", decision["reason"])

    def test_too_few_runs_or_a_small_gap_hold(self):
        self.assertIsNone(insights.rollback_decision(self.dir, "custom:k_drama", min_runs=4)["revert_to"])
        self.assertIsNone(insights.rollback_decision(self.dir, "custom:k_drama", margin=3.0)["revert_to"])
        self.assertIsNone(insights.rollback_decision(self.dir, "custom:k_drama", current=1)["revert_to"])
