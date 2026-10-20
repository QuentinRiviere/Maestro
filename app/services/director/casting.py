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


def normalize_cast(raw: Any, *, max_characters: int = MAX_CHARACTERS, max_locations: int = MAX_LOCATIONS) -> dict:
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
    return {"characters": characters[:max_characters], "locations": locations[:max_locations]}


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


# "23 years old", "23-year-old", "aged 23", "about 17", or a description opening on "17, ...".
_AGE = re.compile(
    r"\b(\d{1,2})\s*(?:-\s*)?(?:years?[\s-]*old|yo\b|ans\b)"
    r"|\b(?:aged?|about|around|roughly)\s+(\d{1,2})\b"
    r"|^\s*(\d{1,2})\s*[,;]",
    re.IGNORECASE,
)


def _subject_lead(description: str) -> str:
    """Who the sheet shows, first: image models read the start of a prompt hardest.

    A long description that states the age late lets a model draw a
    23-year-old as a child or a 49-year-old as a teenager.
    """
    match = _AGE.search(description or "")
    if not match:
        return "a person"
    age = int(next(group for group in match.groups() if group))
    return f"an adult aged {age}" if age >= 18 else f"a young person aged {age}"


# One person per sheet, in a tall frame: asked for two panels, text-only models
# draw two different people, and a full figure in a wide frame leaves the face
# a few dozen pixels tall, where the eyes come out garbled.
SHEET_RESOLUTION = "1088x1920"

# Soft directional light rather than flat, even light: Director shots take
# the people from these sheets, and an evenly lit sheet pulls the shot's own
# (often low-key) lighting back toward a flat studio look.
SHEET_LIGHT = ("a soft key light from the front-left with a gentle fill, the face fully readable with a soft "
               "natural shadow on the far cheek")


def character_sheet_prompt(character: dict, style: str = "") -> str:
    look = f" Rendered as: {style}." if style else ""
    return (
        f"Character reference photograph of {_subject_lead(character['description'])}, alone, standing in a "
        "full-body front view from head to shoes, centered in a tall portrait frame on a plain mid-grey studio "
        "background, looking straight at the lens. The face is sharp and readable: both eyes symmetrical and "
        f"matching, round irises with natural catchlights. Neutral expression, {SHEET_LIGHT}, sharp focus, "
        f"no text, no logo.{look} Person: {character['description']}"
        + (f" Wearing: {character['wardrobe']}." if character.get("wardrobe") else "")
    )


def location_plate_prompt(location: dict, style: str = "") -> str:
    look = f" Rendered as: {style}." if style else ""
    return (
        "Empty establishing view of a film location: the place is deserted and every seat and desk is unoccupied, "
        f"no text overlay.{look} Location: {location['description']}"
        + (f" Layout: {location['notes']}." if location.get("notes") else "")
    )


def prop_shot_prompt(prop: dict, style: str = "") -> str:
    look = f" Rendered as: {style}." if style else ""
    return (
        "Prop reference photograph: the object alone on a plain mid-grey studio table, soft even light, true "
        f"proportions and contact shadow, no people, no hands, no text.{look} Object: {prop['description']}"
        + (f" Details: {prop['notes']}." if prop.get("notes") else "")
    )


_VIEW_SHOTS = {
    "face_front": "a close-up portrait: the head and the top of the shoulders fill a square frame, facing the camera",
    "face_three_quarter": "a close-up portrait: the head and the top of the shoulders fill a square frame, the head "
                          "turned three-quarters toward the left of the frame",
    "face_profile": "a close-up portrait in strict 90-degree side profile: the camera sees the side of the head, "
                    "the nose points to the left edge of the frame, only one eye and one ear are visible; the head "
                    "and the top of the shoulders fill a square frame",
    "back": "a full-body view from directly behind (the camera rotated 180 degrees around the person), head to "
            "shoes, in a tall portrait frame: the person faces away "
            "from the camera with the head straight, the face is not visible, only the back of the head, hair, "
            "garments and shoes; the hair keeps the same length and is worn loose the same way, and front-only "
            "details (pendants, badges, buttons, prints) stay hidden on the front of the body",
}
# Asking for two matching eyes turns a profile back toward the camera.
_VIEW_EYES = {
    "face_front": "both eyes sharp and matching with natural catchlights",
    "face_three_quarter": "both eyes sharp and matching with natural catchlights",
    "face_profile": "the visible eye sharp with a natural catchlight",
    "back": "",
}
VIEW_RESOLUTIONS = {"face_front": "1024x1024", "face_three_quarter": "1024x1024", "face_profile": "1024x1024",
                    "back": SHEET_RESOLUTION}


# Wardrobe items worn on the front of the body. Named in a back-view prompt,
# the image model paints them on the back (a pendant between the shoulder blades).
_FRONT_ONLY_RE = re.compile(
    r"\b(pendants?|necklaces?|lockets?|chains?|broo?ch(?:es)?|broach(?:es)?|badges?|name ?tags?|neckties?|"
    r"bow ?ties?|lanyards?)\b", re.IGNORECASE)


def back_view_wardrobe(wardrobe: str) -> str:
    """The wardrobe without the sentences or items only visible from the front."""
    kept = []
    for sentence in re.split(r"(?<=[.;])\s+", wardrobe or ""):
        if not _FRONT_ONLY_RE.search(sentence):
            kept.append(sentence)
            continue
        items = [part for part in re.split(r",\s*|\s+and\s+", sentence) if not _FRONT_ONLY_RE.search(part)]
        if items:
            kept.append(", ".join(items).rstrip(".;,") + ".")
    return " ".join(part.strip() for part in kept if part.strip())


def view_prompt(view: str, element: dict, variant: dict, style: str = "", *, ref_label: str = "") -> str:
    """One angle of a character's look, edited from its approved image (reference 1)."""
    label = ref_label or "reference image"
    wardrobe = variant.get("description") or element.get("wardrobe") or ""
    if view == "back":
        wardrobe = back_view_wardrobe(wardrobe)
    return (
        f"The exact person of {label} 1, now in {_VIEW_SHOTS[view]}. Keep the same face, eyes, age, skin, "
        f"hairstyle and hair color, build and wardrobe; plain mid-grey studio background, {SHEET_LIGHT}, "
        f"neutral expression, sharp focus, {_VIEW_EYES[view] + ', ' if _VIEW_EYES[view] else ''}"
        f"no text, no logo. Person: {element.get('description', '')}"
        + (f" Wearing: {wardrobe}." if wardrobe else "")
        + realism_clause("character", style)
    )


_LOCATION_VIEW_SHOTS = {
    "reverse": "the camera now stands against the wall that {label} 1 looks at and looks back toward where the "
               "camera of {label} 1 stood: the walls, windows, doors and furniture that were behind that camera "
               "fill the frame, and the wall {label} 1 shows is now behind the camera, out of frame",
    "left": "the camera now stands against the right wall of {label} 1 (the wall on the right of its frame) and "
            "looks straight at the left wall (the wall on the left of its frame), which fills the frame: the far "
            "wall {label} 1 looks at is now on the right edge of the frame and the wall that was behind the camera "
            "of {label} 1 is on the left edge",
    "right": "the camera now stands against the left wall of {label} 1 (the wall on the left of its frame) and "
             "looks straight at the right wall (the wall on the right of its frame), which fills the frame: the far "
             "wall {label} 1 looks at is now on the left edge of the frame and the wall that was behind the camera "
             "of {label} 1 is on the right edge",
}


def location_view_prompt(view: str, element: dict, variant: dict, style: str = "", *, ref_label: str = "") -> str:
    """One angle of a place's look, edited from its approved image (reference 1): the same empty room."""
    label = ref_label or "reference image"
    change = variant.get("description") or ""
    description = element.get("description", "") + (f" State: {change}" if change else "")
    where = "from the opposite side of the same room" if view == "reverse" else "from another side of the same room"
    return (
        f"The exact place of {label} 1, empty, seen {where}: "
        f"{_LOCATION_VIEW_SHOTS[view].format(label=label)}. Keep the same architecture, floor, wall and ceiling "
        "materials, colours, period and light: the same time of day and the same light sources, their direction "
        f"now seen from the other side. The furniture of {label} 1 keeps its size, style and place in the room, "
        "now seen from its other side. Nobody in the place, every seat unoccupied, no text overlay. "
        f"Location: {description}"
        + (f" Layout: {element['notes']}." if element.get("notes") else "")
        + realism_clause("location", style)
    )


def bible_image_prompt(kind: str, element: dict, variant: dict, style: str = "", *,
                       ref_label: str = "", from_default: bool = False, chained: int = 0) -> str:
    """The prompt of one bible image: the element's sheet, plate or prop shot, in one of its looks.

    ``from_default``: reference 1 is the approved default look, which the
    render must keep except for what the variant changes. ``chained``: how
    many approved elements of the same world follow as references.
    """
    entry = dict(element)
    change = variant.get("description") or ""
    if change and kind == "character":
        entry["wardrobe"] = change
    elif change:
        entry["description"] = f"{element['description']} State: {change}"
    body = (character_sheet_prompt(entry, style) if kind == "character"
            else location_plate_prompt(entry, style) if kind == "location"
            else prop_shot_prompt(entry, style)) + realism_clause(kind, style)
    label = ref_label or "reference image"
    subject = {"character": "person", "location": "place", "prop": "object"}[kind]
    leads = []
    if from_default:
        leads.append(f"The same {subject} as {label} 1: keep identity, proportions, materials and every detail "
                     f"exactly; change only this: {change or 'nothing'}.")
    if chained:
        first = 2 if from_default else 1
        names = ", ".join(f"{label} {first + i}" for i in range(chained))
        leads.append(f"Same world as {names}: match their architecture, materials, palette, period and light.")
    return " ".join(leads + [body])


_PHOTOGRAPHIC = re.compile(
    r"\b(?:photo\w*|live[- ]action|cinema\w*|film stock|shot on|camera|lens\w*|arri|alexa|kodak|fuji|"
    r"35\s?mm|anamorphic|35mm)\b", re.IGNORECASE)


def asset_medium(contract: dict | None, look: str = "") -> str:
    """The medium a bible image is rendered in: the style contract's, else the look block's first sentence.

    Reference sheets leave the look's atmosphere out (haze and depth of
    field blur them) but never the medium: without one a model picks its
    own, and a friendly round-faced student comes out as a 3D chibi doll.
    """
    medium = style_phrase(contract)
    if not medium and look:
        medium = _text(re.split(r"(?<=[.!?])\s", look.strip(), maxsplit=1)[0], 220)
    return medium


def is_photographic_medium(medium: str) -> bool:
    """Whether a medium names live action or photography."""
    return bool(_PHOTOGRAPHIC.search(medium or ""))


def realism_clause(kind: str, medium: str) -> str:
    """For a photographic medium, say the subject is real, with real proportions."""
    if not is_photographic_medium(medium):
        return ""
    return {"character": " A real person photographed in live action, natural adult human proportions.",
            "prop": " A real object photographed.",
            "location": " A real place photographed."}[kind]


def style_phrase(contract: dict | None) -> str:
    """A short description of the medium for image prompts."""
    contract = contract if isinstance(contract, dict) else {}
    parts = [contract.get("medium"), contract.get("visual_language")]
    return _text("; ".join(str(p) for p in parts if p and str(p).lower() != "unspecified"), 300)
