"""Viral Video skill: planner contract and its short-film workflow routing."""
from __future__ import annotations

import os
import sys
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.planners.viral_video import ViralVideoPlanner  # noqa: E402


def shot(goal, duration=4):
    return {"scene_goal": goal, "duration_sec": duration, "action_beats": ["x"],
            "camera_plan": {"framing": "close-up"}, "audio_plan": {"mode": "generated_audio"}}


class FakePlanner(ViralVideoPlanner):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.prompts = []

    def _call_llm_json(self, **kwargs):
        self.prompts.append(kwargs)
        return list(self.answer)


class ViralPlannerTests(unittest.TestCase):
    def test_scene_description_is_the_concept(self):
        planner = FakePlanner([shot("hook", 3), shot("payoff", 5)])
        plan = planner.plan(scene_description="A cat discovers a cucumber", platform="tiktok", style="meme")
        self.assertEqual(plan.skill_type, "viral_video")
        self.assertIn("A cat discovers a cucumber", planner.prompts[0]["user_prompt"])
        self.assertEqual([s.duration_sec for s in plan.shots], [3, 5])
        self.assertTrue(plan.shots[0].metadata["is_hook"])
        self.assertEqual(plan.shots[0].audio_plan.mode, "generated_audio")

    def test_audio_timeline_fixes_shot_count_and_durations(self):
        planner = FakePlanner([shot("hook"), shot("a"), shot("b"), shot("extra")])
        clips = [{"start": 0, "end": 2.5}, {"start": 2.5, "end": 6.0}, {"duration_sec": 4.0}]
        plan = planner.plan(concept="Morning routine", clips=clips, audio_path="/a.wav")
        self.assertEqual([s.duration_sec for s in plan.shots], [2.5, 3.5, 4.0])
        self.assertTrue(all(s.audio_plan.mode == "audio_driven" for s in plan.shots))
        self.assertIn("exactly 3 shot plans", planner.prompts[0]["system_prompt"])

    def test_short_llm_answer_is_padded_to_the_timeline(self):
        plan = FakePlanner([shot("only")]).plan(concept="x", clips=[{"duration_sec": 2}, {"duration_sec": 3}])
        self.assertEqual(len(plan.shots), 2)

    def test_concept_is_required(self):
        with self.assertRaises(ValueError):
            FakePlanner([]).plan(scene_description="  ")


class SkillRoutingTests(unittest.TestCase):
    def test_director_skill_only_overrides_short_film_workflows(self):
        from services.director_pipeline import director_skill

        self.assertEqual(director_skill({"director_skill": "viral_video"}, "short_film_story"), "viral_video")
        self.assertEqual(director_skill({"director_skill": "viral_video"}, "short_film_audio"), "viral_video")
        self.assertIsNone(director_skill({"director_skill": "viral_video"}, "music_video"))
        self.assertIsNone(director_skill({"director_skill": "podcast"}, "short_film_story"))
        self.assertIsNone(director_skill({}, "short_film_story"))


if __name__ == "__main__":
    unittest.main()
