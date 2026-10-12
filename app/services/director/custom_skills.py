"""Declarative Director skills.

A custom skill is data, never code: a validated JSON definition that the
generic ``CustomSkillPlanner`` turns into shot plans. It runs on the
short-film workflows (written story, or uploaded audio cut into segments), so
timeline, images, video, assembly and review are reused unchanged. The Skill
Creator writes these definitions; the built-in Python skills stay as they are.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

SKILLS_DIR = Path(__file__).resolve().parents[2] / "settings" / "director_skills"
CUSTOM_PREFIX = "custom:"
ICONS = ("sparkles", "film", "music", "mic", "camera", "clapperboard", "megaphone", "book", "gamepad", "palette",
         "heart", "zap")
WORKFLOWS = ("story", "audio", "both")
RULE_FIELDS = ("goal", "structure", "pacing", "camera", "dialogue", "sound")
_LOCK = threading.Lock()
_SLUG = re.compile(r"[^a-z0-9]+")

# Read-only summaries of the Python skills, offered as inspiration.
BUILTIN_SKILLS: dict[str, dict[str, Any]] = {
    "music_video": {
        "name": "Music Video", "icon": "music", "workflow": "audio",
        "description": "Automated music video cut on the analysed song: sections, beats and lyrics drive the shots.",
        "rules": {
            "goal": "Visualise a song so that picture and music feel inseparable.",
            "structure": "Follow the song sections (intro, verse, chorus, bridge, outro); escalate visual energy into choruses.",
            "pacing": "Cuts land on beats and phrase boundaries; shot length follows tempo and section energy.",
            "camera": "Performance coverage mixed with narrative or abstract b-roll; movement follows rhythm.",
            "dialogue": "Lyrics are sung by performers when on screen; no added speech.",
            "sound": "The song is the soundtrack; no generated music.",
        },
    },
    "short_film": {
        "name": "Short Film", "icon": "film", "workflow": "both",
        "description": "Dialogue-driven narrative scenes, written from a story or cut on recorded dialogue.",
        "rules": {
            "goal": "Tell a coherent story with distinct characters, causality and an ending.",
            "structure": "Setup, rising tension, turn, resolution; each scene has a goal and an ending beat.",
            "pacing": "Scene length follows dialogue and action; breathing room for reactions.",
            "camera": "Classic coverage: establishing, medium, close-ups; camera motivated by story and performance.",
            "dialogue": "Natural, attributed lines with delivery cues; exact lines are preserved.",
            "sound": "Dialogue first, then ambience and effects; generated or recorded.",
        },
    },
    "viral_video": {
        "name": "Viral Video", "icon": "sparkles", "workflow": "both",
        "description": "Hook-first short-form video for TikTok, Reels and Shorts.",
        "rules": {
            "goal": "Stop the scroll in the first second and keep attention to a payoff.",
            "structure": "Hook, setup, escalation, payoff or callback to the hook.",
            "pacing": "Short shots (3-8 s); every shot earns its screen time.",
            "camera": "Tight framing, snap zooms, handheld energy or polished cinematic, per style.",
            "dialogue": "Punchy, direct-address lines; reactions over exposition.",
            "sound": "Generated audio or the uploaded voice-over or trending sound.",
        },
        "options": [
            {"key": "platform", "label": "Platform", "values": ["general", "tiktok", "reels", "shorts"], "default": "general"},
            {"key": "style", "label": "Style", "values": ["cinematic", "meme", "ugc", "direct_address", "reaction", "tutorial"],
             "default": "cinematic"},
        ],
    },
}


class SkillError(ValueError):
    pass


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _texts(value: Any, limit: int, count: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (_text(item, limit) for item in value) if text][:count]


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def slugify(name: str) -> str:
    return _SLUG.sub("_", str(name or "").lower()).strip("_")[:40] or "skill"


def validate_spec(raw: Any, *, skill_id: str | None = None) -> dict:
    """Normalise a definition: known fields only, bounded sizes, no code."""
    if not isinstance(raw, dict):
        raise SkillError("A skill definition must be an object")
    name = _text(raw.get("name"), 60)
    if not name:
        raise SkillError("A skill needs a name")
    rules_raw = raw.get("rules") if isinstance(raw.get("rules"), dict) else {}
    rules = {field: _text(rules_raw.get(field), 600) for field in RULE_FIELDS}
    rules["do"] = _texts(rules_raw.get("do"), 200, 12)
    rules["dont"] = _texts(rules_raw.get("dont"), 200, 12)
    if not any(rules[field] for field in RULE_FIELDS) and not rules["do"]:
        raise SkillError("A skill needs at least one writing rule")
    shot = raw.get("shot_duration") if isinstance(raw.get("shot_duration"), list) else [3, 8]
    low = _int(shot[0] if shot else 3, 3, 1, 30)
    high = _int(shot[1] if len(shot) > 1 else 8, 8, low, 30)
    options = []
    for option in raw.get("options") or []:
        if not isinstance(option, dict):
            continue
        values = _texts(option.get("values"), 40, 12)
        key = slugify(option.get("key") or option.get("label") or "")
        if not values or not key or any(o["key"] == key for o in options):
            continue
        default = _text(option.get("default"), 40)
        options.append({"key": key, "label": _text(option.get("label") or key, 40), "values": values,
                        "default": default if default in values else values[0]})
    references = []
    for ref in raw.get("references") or []:
        if isinstance(ref, dict) and _text(ref.get("notes"), 2000):
            references.append({"kind": ref.get("kind") if ref.get("kind") in ("video", "image", "audio") else "video",
                               "name": _text(ref.get("name"), 120), "notes": _text(ref.get("notes"), 2000)})
    contract = raw.get("style_hint")
    return {
        "id": skill_id or slugify(raw.get("id") or name),
        "name": name,
        "description": _text(raw.get("description"), 300),
        "icon": raw.get("icon") if raw.get("icon") in ICONS else "sparkles",
        "workflow": raw.get("workflow") if raw.get("workflow") in WORKFLOWS else "both",
        "default_duration": _int(raw.get("default_duration"), 30, 5, 600),
        "shot_duration": [low, high],
        "max_shots": _int(raw.get("max_shots"), 12, 1, 60),
        "style_hint": _text(contract, 600),
        "rules": rules,
        "options": options[:6],
        "references": references[:12],
        "based_on": _text(raw.get("based_on"), 80) or None,
    }


def _path(skill_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]{1,40}", skill_id or ""):
        raise SkillError("Invalid skill id")
    return SKILLS_DIR / f"{skill_id}.json"


def list_custom() -> list[dict]:
    if not SKILLS_DIR.is_dir():
        return []
    skills = []
    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            skills.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return skills


def get_custom(skill_id: str) -> dict | None:
    if skill_id.startswith(CUSTOM_PREFIX):
        skill_id = skill_id[len(CUSTOM_PREFIX):]
    try:
        return json.loads(_path(skill_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, SkillError):
        return None


def save_custom(raw: dict, *, skill_id: str | None = None) -> dict:
    """Create (new id) or overwrite (given id) a definition."""
    spec = validate_spec(raw, skill_id=skill_id)
    with _LOCK:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        if skill_id is None:
            base, suffix = spec["id"], 2
            while _path(spec["id"]).exists() or spec["id"] in BUILTIN_SKILLS:
                spec["id"] = f"{base[:36]}_{suffix}"
                suffix += 1
        previous = get_custom(spec["id"]) or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        spec["version"] = int(previous.get("version") or 0) + 1
        spec["created_at"] = previous.get("created_at") or now
        spec["updated_at"] = now
        target = _path(spec["id"])
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
    return spec


def delete_custom(skill_id: str) -> bool:
    with _LOCK:
        path = _path(skill_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def catalog() -> list[dict]:
    """Every skill the Director selector lists (built-in first)."""
    entries = [
        {"id": skill_id, "name": info["name"], "description": info["description"], "icon": info["icon"],
         "workflow": info["workflow"], "builtin": True, "options": info.get("options", [])}
        for skill_id, info in BUILTIN_SKILLS.items()
    ]
    entries += [
        {"id": CUSTOM_PREFIX + spec["id"], "name": spec["name"], "description": spec.get("description", ""),
         "icon": spec.get("icon", "sparkles"), "workflow": spec.get("workflow", "both"), "builtin": False,
         "options": spec.get("options", []), "default_duration": spec.get("default_duration"),
         "version": spec.get("version"), "updated_at": spec.get("updated_at")}
        for spec in list_custom()
    ]
    return entries


def inspiration(skill_id: str) -> dict | None:
    """A definition to start from: a custom spec, or a built-in summary."""
    if skill_id in BUILTIN_SKILLS:
        info = BUILTIN_SKILLS[skill_id]
        return {"id": skill_id, "name": info["name"], "description": info["description"], "icon": info["icon"],
                "workflow": info["workflow"], "rules": {**info["rules"], "do": [], "dont": []},
                "options": info.get("options", []), "builtin": True}
    return get_custom(skill_id)
