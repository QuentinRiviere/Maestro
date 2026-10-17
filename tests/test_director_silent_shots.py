"""Shots without a spoken line tell a sound-making video model to keep every mouth closed."""
import os
import sys

_APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


def test_a_shot_without_a_line_gets_the_silent_note():
    plan = {"_director_dialogue_beats": []}
    prompt = pipeline._silent_shot_prompt(plan, "She nods once.")
    assert prompt == f"She nods once. {pipeline.SILENT_SHOT_NOTE}"
    assert pipeline._silent_shot_prompt(plan, prompt) == prompt


def test_a_shot_with_a_line_keeps_its_prompt():
    plan = {"_director_dialogue_beats": [{"speaker_id": "char_1", "spoken_text": "Don't run this time."}]}
    assert pipeline._silent_shot_prompt(plan, "He speaks.") == "He speaks."
    assert pipeline._silent_shot_prompt({"_director_dialogue_beats": [{"spoken_text": "  "}]}, "") == ""
