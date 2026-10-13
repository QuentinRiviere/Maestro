"""Casting: fix the recurring people and places before shots are planned.

Without a cast every shot re-invents its actors from a vague phrase ("a teen
girl in light blue"), so faces, age and wardrobe drift between shots. Casting
writes one canonical description per recurring character and main location,
then the Director renders one reference image of each with its image model.
The descriptions feed the planners as ``characters``; the images become the
project's character and location references, used by shot images, Omni
references and the planner's visual identity rules alike.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

MAX_CHARACTERS = 4
MAX_LOCATIONS = 2

CAST_SYSTEM = f"""You are the casting director and production designer of a short video.
From the brief (and the skill rules and visual style when given), list the RECURRING on-screen people
and the main locations, as concrete visual facts a painter could reproduce identically in every shot.
Return ONLY JSON:
{{"characters": [{{"name": "short name or role", "role": "story role in a few words",
  "descriptor": "how a shot names this person without the name, at most 8 words, e.g. 'the teen girl with
  see-through bangs'",
  "description": "apparent age, origin/ethnicity when it matters to the story or setting, face shape, eyes,
  skin, hair (cut, color, length), build, height, one or two distinctive features",
  "wardrobe": "the exact outfit worn in this production (garments, colors, materials, accessories)"}}],
 "locations": [{{"name": "short name", "description": "architecture, materials, props, signage, period,
  culture-specific details, time of day and light"}}]}}
Rules:
- At most {MAX_CHARACTERS} characters (only those seen in several shots; crowds and extras are not cast)
  and at most {MAX_LOCATIONS} locations.
- Match the culture, country and era the brief implies (e.g. a Korean drama set in a high school gets Korean
  students in Korean school uniforms and a Korean school building), unless the brief says otherwise.
- Describe looks only: no personality, no actions, no camera language.
- Write every field in English, whatever the language of the brief or the skill: they become image prompts.
  Only names keep their original form.
- Stay consistent with the visual style; never name a real person."""


class CastingError(RuntimeError):
    pass


def _text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def parse_cast(text: str) -> dict:
    """Extract and bound the cast JSON an LLM returned."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise CastingError("The LLM returned no cast")
    try:
        raw = json.loads(match.group(0))
    except ValueError as error:
        raise CastingError(f"The LLM returned an invalid cast: {error}") from error
    return normalize_cast(raw)


def normalize_cast(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    characters = []
    for item in raw.get("characters") or []:
        if not isinstance(item, dict):
            continue
        description = _text(item.get("description"), 700)
        name = _text(item.get("name"), 60) or f"Character {len(characters) + 1}"
        if not description:
            continue
        characters.append({
            "name": name,
            "role": _text(item.get("role"), 120),
            "descriptor": _text(item.get("descriptor"), 80),
            "description": description,
            "wardrobe": _text(item.get("wardrobe"), 400),
            "image": _text(item.get("image"), 260) or None,
        })
    locations = []
    for item in raw.get("locations") or []:
        if not isinstance(item, dict):
            continue
        description = _text(item.get("description"), 700)
        if not description:
            continue
        locations.append({
            "name": _text(item.get("name"), 60) or f"Location {len(locations) + 1}",
            "description": description,
            "image": _text(item.get("image"), 260) or None,
        })
    return {"characters": characters[:MAX_CHARACTERS], "locations": locations[:MAX_LOCATIONS]}


def cast_from_characters(characters: list[dict]) -> dict:
    """A cast built from the user's own character cards (no LLM)."""
    return normalize_cast({"characters": [
        {"name": c.get("name") or c.get("display_name"), "role": c.get("role"),
         "description": c.get("description") or c.get("physical_description") or c.get("visual_description"),
         "wardrobe": c.get("wardrobe")}
        for c in characters or [] if isinstance(c, dict)
    ]})


def derive_cast(
    brief: str,
    generate: Callable[..., str],
    *,
    style: str = "",
    skill_rules: str = "",
) -> dict:
    """One LLM call (and one retry on bad JSON) returning the normalized cast."""
    prompt = "\n\n".join(filter(None, [
        f"Brief:\n{brief.strip()}",
        f"Skill rules:\n{skill_rules.strip()}" if skill_rules.strip() else "",
        f"Visual style:\n{style.strip()}" if style.strip() else "",
    ]))
    last_error: Exception | None = None
    for _ in range(2):
        text = generate(prompt=prompt, system_prompt=CAST_SYSTEM, max_new_tokens=1600, temperature=0.4,
                        enable_thinking=False)
        try:
            return parse_cast(text)
        except CastingError as error:
            last_error = error
    raise CastingError(str(last_error or "No cast"))


def planner_characters(cast: dict) -> list[dict]:
    """The cast in the ``characters`` shape every Director planner reads."""
    return [
        {"name": c["name"], "description": c["description"], "wardrobe": c.get("wardrobe", ""),
         **({"role": c["role"]} if c.get("role") else {}),
         **({"descriptor": c["descriptor"]} if c.get("descriptor") else {})}
        for c in cast.get("characters") or []
    ]


def character_sheet_prompt(character: dict, style: str = "") -> str:
    look = f" Rendered as: {style}." if style else ""
    return (
        "Character reference sheet on a plain light-grey studio background: the same person shown twice side by "
        "side, a front head-and-shoulders portrait on the left and a full-body front view on the right, neutral "
        f"expression, even soft lighting, no text, no logo.{look} Person: {character['description']}"
        + (f" Wearing: {character['wardrobe']}." if character.get("wardrobe") else "")
    )


def location_plate_prompt(location: dict, style: str = "") -> str:
    look = f" Rendered as: {style}." if style else ""
    return (
        f"Empty establishing view of a film location, no people, no text overlay.{look} "
        f"Location: {location['description']}"
    )


def style_phrase(contract: dict | None) -> str:
    """A short description of the medium for image prompts."""
    contract = contract if isinstance(contract, dict) else {}
    parts = [contract.get("medium"), contract.get("visual_language")]
    return _text("; ".join(str(p) for p in parts if p and str(p).lower() != "unspecified"), 300)
