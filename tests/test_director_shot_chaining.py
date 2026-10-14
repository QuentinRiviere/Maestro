"""Same-camera continuations render from the previous clip's last frame."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director import custom_skills  # noqa: E402
from services.director.planners.custom import CustomSkillPlanner  # noqa: E402
from services.director_video_strategy import SHOT_IMAGE_PROMPT_ONLY  # noqa: E402

SPEC = {
    "name": "Chase", "description": "A foot chase.", "icon": "heart", "workflow": "story",
    "default_duration": 20, "shot_duration": [4, 8], "max_shots": 6,
    "rules": {"goal": "Keep the runner moving."},
}


class FakePlanner(CustomSkillPlanner):
    def __init__(self, spec, answer, scenes):
        super().__init__(spec)
        self.answer, self.scenes, self.prompts = answer, scenes, []

    def _call_llm_json(self, **kwargs):
        self.prompts.append(kwargs)
        if kwargs["system_prompt"].startswith("You are the director preparing the scenes"):
            return list(self.scenes)
        return list(self.answer)


class PlannerMappingTests(unittest.TestCase):
    def test_only_explicit_same_scene_continuations_extend_the_previous_shot(self):
        planner = FakePlanner(
            custom_skills.validate_spec(SPEC),
            [
                {"scene_id": "alley", "scene_goal": "wide", "duration_sec": 6, "keep_sec": [1, 4],
                 "continues_previous": True},
                {"scene_id": "alley", "scene_goal": "the run goes on", "duration_sec": 6, "keep_sec": [2, 5],
                 "continues_previous": True},
                {"scene_id": "alley", "scene_goal": "close-up", "duration_sec": 6, "continues_previous": False},
                {"scene_id": "roof", "scene_goal": "new scene", "duration_sec": 6, "continues_previous": "true"},
                {"scene_id": "roof", "scene_goal": "still running", "duration_sec": 6, "continues_previous": "true"},
            ],
            [{"scene_id": "alley", "place": "alley"}, {"scene_id": "roof", "place": "roof"}],
        )
        plan = planner.plan(scene_description="A runner escapes")
        self.assertIn("continues_previous", planner.prompts[-1]["system_prompt"])
        self.assertEqual(
            [shot.continuity_strategy for shot in plan.shots],
            ["independent", "extend_previous", "continuous", "independent", "extend_previous"],
        )
        # The edit plays across the handoff frame.
        self.assertEqual(plan.shots[0].metadata["keep_sec"], (1.0, 6.0))
        self.assertEqual(plan.shots[1].metadata["keep_sec"], (0.0, 5.0))


class VideoStageChainingTests(unittest.TestCase):
    def setUp(self):
        self.original_wgp = pipeline._wgp
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def tearDown(self):
        pipeline._wgp = self.original_wgp

    def _image(self, name):
        path = os.path.join(self.temp_dir.name, name)
        Path(path).write_bytes(b"png")
        return name

    def _install(self, model_type, model_def, frames=(17, 8, 8)):
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda selected: model_def if selected == model_type else None,
            get_model_family=lambda selected, for_ui=False: "test",
            get_base_model_type=lambda selected: model_type,
            get_model_min_frames_and_step=lambda selected: frames,
            get_default_settings=lambda selected: {},
            get_lora_dir=lambda selected: os.path.join(self.temp_dir.name, "loras"),
        )

    @staticmethod
    def _plans():
        return [
            {"video_prompt": "wide", "_director_continuity_group": "alley",
             "_director_continuity_strategy": "independent"},
            {"video_prompt": "reverse", "_director_continuity_group": "alley",
             "_director_continuity_strategy": "continuous"},
            {"video_prompt": "the run goes on", "_director_continuity_group": "alley",
             "_director_continuity_strategy": "extend_previous"},
        ]

    def _run(self, model_type, params, clips, images):
        captured = {}
        with patch.object(pipeline, "_submit_and_wait",
                          side_effect=lambda gen, **kwargs: captured.update(gen) or ["joined.mp4"]):
            pipeline._run_video_generation(
                "chain", {"video_model": model_type, "pipeline_type": "short_film_story", "seamless": False,
                          "video_params": {}, **params},
                self._plans(), clips, images, out_dir=self.temp_dir.name,
            )
        return captured

    def test_rolling_window_model_chains_only_the_continuation(self):
        self._install("ltx", {
            "name": "LTX", "fps": 24, "image_prompt_types_allowed": "TSEV", "sliding_window": True,
            "returns_audio": True, "custom_frames_injection": True,
        })
        images = [self._image(f"shot{i}.png") for i in range(3)]
        captured = self._run("ltx", {}, [{"duration_sec": 4} for _ in range(3)], images)
        starts = captured["image_start"]
        self.assertTrue(starts[0].endswith("shot0.png"))
        self.assertTrue(starts[1].endswith("shot1.png"))  # an ordinary cut keeps its own image
        self.assertEqual(starts[2], "")
        self.assertEqual(captured["per_clip_continue_from_previous"], [False, False, True])

    def test_bounded_model_with_shot_images_chains_instead_of_aiming_at_the_next_image(self):
        self._install("minimax_h3", {
            "name": "H3 FL2VA", "fps": 24, "frames_minimum": 124, "frames_maximum": 345, "frames_steps": 17,
            "latent_size": 17, "image_prompt_types_allowed": "TSE", "returns_audio": True,
            "director_video_strategy": "bounded_start_end", "director_audio_input_mode": "none",
            "director_trim_end_frames": False,
        }, frames=(124, 17, 17))
        images = [self._image(f"shot{i}.png") for i in range(3)]
        captured = self._run("minimax_h3", {}, [{"duration_frames": 124} for _ in range(3)], images)
        self.assertEqual(captured["image_start"][2], "")
        self.assertNotIn("image_end", captured)  # shot 2 ends freely
        self.assertEqual(captured["per_clip_continue_from_previous"], [False, False, True])

    def test_prompt_only_rolling_model_marks_the_continuation(self):
        self._install("ltx", {
            "name": "LTX", "fps": 24, "image_prompt_types_allowed": "TSEV", "sliding_window": True,
            "returns_audio": True, "custom_frames_injection": True, "director_shot_image_support": "optional",
        })
        captured = self._run("ltx", {"_director_shot_image_policy": SHOT_IMAGE_PROMPT_ONLY},
                             [{"duration_sec": 4} for _ in range(3)], ["", "", ""])
        self.assertNotIn("image_start", captured)
        self.assertEqual(captured["per_clip_continue_from_previous"], [False, False, True])

    def test_reference_only_models_never_chain(self):
        plans = self._plans()
        self.assertEqual(
            pipeline._director_chained_shots(plans, [{}] * 3, {"director_video_strategy": "omni_reference"}),
            [False, False, False],
        )
        self.assertEqual(
            pipeline._director_chained_shots(plans, [{}] * 3, {"image_prompt_types_allowed": "T"}),
            [False, False, False],
        )
        self.assertEqual(
            pipeline._director_chained_shots(plans, [{}] * 3, {"image_prompt_types_allowed": "TS"}),
            [False, False, True],
        )


class WorkerContractTests(unittest.TestCase):
    def test_only_bounded_models_hand_off_the_true_final_frame(self):
        launch = Path(_APP_DIR, "launch.py").read_text(encoding="utf-8")
        self.assertIn("if bounded_continuation and _mc_bounded_director:", launch)
        self.assertIn('"_continuation_tail_skip", 8', launch)


if __name__ == "__main__":
    unittest.main()
