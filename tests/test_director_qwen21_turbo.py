"""Director accepts the official Qwen Image 2.1 Turbo checkpoint for multi-reference shots."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from models.qwen21.qwen21_handler import MODEL_TYPE, family_handler  # noqa: E402
from services import director_pipeline as pipeline  # noqa: E402
from services.director_model_compat import assess_director_model  # noqa: E402

TURBO = "qwen_image_21_7B_turbo_8step"
LIGHTNING = "qwen_image_edit_2511_20B_fp8_lightning_8step"


def _turbo_def():
    """The Turbo definition as wgp.init_model_def resolves it."""
    with open(os.path.join(_APP_DIR, "defaults", f"{TURBO}.json"), encoding="utf-8") as handle:
        model = json.load(handle)["model"]
    resolved = family_handler.query_model_def(MODEL_TYPE, model)
    resolved.update(model)
    return resolved


def _lightning_def():
    return {
        "name": "Qwen Image Edit (2511) Lightning 8-Step FP8 20B",
        "image_outputs": True,
        "image_ref_choices": {"choices": [("None", ""), ("Main plus references", "KI")]},
    }


class DirectorQwen21TurboTests(unittest.TestCase):
    def setUp(self):
        self.original_wgp = pipeline._wgp
        definitions = {TURBO: _turbo_def(), LIGHTNING: _lightning_def()}
        pipeline._wgp = SimpleNamespace(
            get_model_def=lambda model_type: definitions.get(model_type),
            get_model_family=lambda model_type, for_ui=False: "qwen",
            get_base_model_type=lambda model_type: MODEL_TYPE if model_type == TURBO else model_type,
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.refs = []
        for index in range(6):
            path = os.path.join(directory.name, f"ref{index}.png")
            with open(path, "wb") as handle:
                handle.write(b"png")
            self.refs.append(path)
        self.layout = os.path.join(directory.name, "layout.png")
        with open(self.layout, "wb") as handle:
            handle.write(b"png")

    def tearDown(self):
        pipeline._wgp = self.original_wgp

    def test_turbo_is_a_compatible_reference_editing_image_model(self):
        assessment = assess_director_model(TURBO, _turbo_def(), architecture=MODEL_TYPE)
        self.assertTrue(assessment["image"]["compatible"], assessment["image"]["reason"])
        self.assertEqual(assessment["image_reference_mode"], "KI")
        self.assertEqual(assessment["max_image_refs"], 10)
        # Raises when the model cannot drive Director's image stage.
        pipeline._validate_director_models({"image_model": TURBO}, stages=("image",))

    def test_turbo_shots_take_up_to_four_references_named_for_qwen_21(self):
        self.assertEqual(pipeline.director_image_ref_budget(TURBO), 4)
        self.assertEqual(pipeline.image_model_reference_limit(TURBO), 4)
        self.assertTrue(pipeline.layout_diagram_fits(TURBO))
        self.assertEqual(pipeline._ref_name(TURBO, 2), "reference image 2")
        refs, prompt = pipeline.assemble_director_image_refs(
            TURBO, "Two friends at the bar.", self.refs[0], self.refs[1:], pid="test")
        self.assertEqual(refs, self.refs[:4])
        refs, prompt = pipeline.assemble_director_image_refs(
            TURBO, "Two friends at the bar.", self.refs[0], self.refs[1:], layout_ref=self.layout, pid="test")
        self.assertEqual(refs, self.refs[:3] + [self.layout])
        self.assertIn("reference image 4 is a top-view layout diagram", prompt)
        self.assertEqual(pipeline.director_shot_prompt_type(TURBO, refs, canvas=True), "KI")
        self.assertEqual(pipeline.director_shot_prompt_type(TURBO, refs, canvas=False), "I")
        self.assertEqual(pipeline.director_shot_prompt_type(TURBO, [], canvas=True), "")

    def test_scene_shot_keeps_three_people_and_the_set_beside_nothing_else(self):
        refs, prompt = pipeline.scene_shot_refs(
            TURBO, {"_director_camera_framing": "wide shot"}, "the three of them sit down",
            master=self.refs[0], identity=self.refs[1:5], empty_set=True)
        self.assertEqual(refs, self.refs[1:4] + [self.refs[0]])
        self.assertIn("reference image 1, reference image 2, reference image 3 show", prompt)

    def test_lightning_2511_profile_is_unchanged(self):
        self.assertEqual(pipeline.director_image_ref_budget(LIGHTNING), 3)
        self.assertEqual(pipeline._ref_name(LIGHTNING, 1), "Picture 1")
        pipeline._validate_director_models({"image_model": LIGHTNING}, stages=("image",))


if __name__ == "__main__":
    unittest.main()
