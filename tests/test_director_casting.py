"""Director casting: canonical cast, reference images and storyboard review."""
from __future__ import annotations

import json
import os
import sys
import tempfile

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import casting  # noqa: E402


def test_parse_cast_bounds_and_cleans_the_llm_answer():
    answer = "Here:\n" + json.dumps({
        "characters": [{"name": f"C{i}", "description": f"face {i}", "wardrobe": "uniform"} for i in range(6)]
        + [{"name": "Nobody", "description": ""}],
        "locations": [{"name": "Hall", "description": "Korean school corridor"}, {"name": "A", "description": "a"},
                      {"name": "B", "description": "b"}],
    })
    cast = casting.parse_cast(answer)
    assert [c["name"] for c in cast["characters"]] == ["C0", "C1", "C2", "C3"]
    assert len(cast["locations"]) == casting.MAX_LOCATIONS
    assert cast["characters"][0]["image"] is None


def test_derive_cast_retries_once_on_bad_json():
    answers = iter(["no json", json.dumps({"characters": [{"name": "Ha-eun", "description": "17, Korean"}]})])
    cast = casting.derive_cast("A new student meets a mysterious boy", lambda **_: next(answers))
    assert cast["characters"][0]["name"] == "Ha-eun"


def test_user_characters_become_the_cast_and_planner_characters():
    cast = casting.cast_from_characters([{"name": "Min", "description": "tall, short black hair", "wardrobe": "navy blazer"}])
    assert casting.planner_characters(cast) == [
        {"name": "Min", "description": "tall, short black hair", "wardrobe": "navy blazer"}]
    prompt = casting.character_sheet_prompt(cast["characters"][0], "live-action Korean drama")
    assert "navy blazer" in prompt and "live-action Korean drama" in prompt


def test_style_phrase_skips_unspecified_contracts():
    assert casting.style_phrase({"medium": "unspecified"}) == ""
    assert casting.style_phrase({"medium": "3D animation", "visual_language": "soft toon shading"}) == \
        "3D animation; soft toon shading"


def test_casting_is_automatic_only_for_story_projects_without_identity():
    from services.director_pipeline import _director_casting_enabled

    assert _director_casting_enabled({}, "short_film_story")
    assert not _director_casting_enabled({}, "music_video")
    assert _director_casting_enabled({"director_casting": "on"}, "music_video")
    assert not _director_casting_enabled({"director_casting": "off"}, "short_film_story")
    assert not _director_casting_enabled({"character_ref_paths": ["/a.png"]}, "short_film_story")


def test_cast_images_replace_previous_cast_references_only():
    from services import director_pipeline as dp

    with tempfile.TemporaryDirectory() as out_dir:
        for name in ("user.png", "sheet.png", "hall.png"):
            open(os.path.join(out_dir, name), "wb").close()
        pid = "casttest"
        dp._pipelines[pid] = {"out_dir": out_dir, "status": "running", "params": {}}
        try:
            params = {
                "shot_image_guidance": "prompt_only",
                "character_ref_paths": [os.path.join(out_dir, "user.png"), os.path.join(out_dir, "old.png")],
                "character_ref_labels": ["Me", "Old"],
                "_director_cast_ref_paths": [os.path.join(out_dir, "old.png")],
            }
            cast = {"characters": [{"name": "Ha-eun", "description": "d", "image": "sheet.png"}],
                    "locations": [{"name": "Hall", "description": "d", "image": "hall.png"}]}
            dp._apply_cast_references(pid, params, cast)
            assert params["character_ref_paths"] == [os.path.join(out_dir, "user.png"), os.path.join(out_dir, "sheet.png")]
            assert params["character_ref_labels"] == ["Me", "Ha-eun"]
            assert params["location_ref_labels"] == ["Hall"]
            assert dp._pipelines[pid]["cast"]["characters"][0]["image"] == "sheet.png"
        finally:
            dp._pipelines.pop(pid, None)


def test_planner_llm_calls_retry_a_dropped_stream_only(monkeypatch):
    from services.director.planners import base

    monkeypatch.setattr(base.time, "sleep", lambda _s: None)
    answers = iter([RuntimeError("LLM request failed: Response ended prematurely"), "[]"])

    def flaky(**_kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert base._retry_transient(flaky)(prompt="p") == "[]"

    def broken(**_kwargs):
        raise ValueError("bad schema")

    try:
        base._retry_transient(broken)(prompt="p")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-transient error must not be retried")
