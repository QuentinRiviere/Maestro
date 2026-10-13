"""A chat that writes and revises a series bible with the user.

The assistant sees the bible's text (never its images) and the conversation,
and answers with a short reply, a full proposed bible and the list of its
changes. Nothing is saved here: the user applies the proposal to the draft
and saves it. Existing elements and looks keep their ids, so their images
stay attached.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.director import bibles

MAX_MESSAGES = 16
MAX_MESSAGE_CHARS = 4000

SYSTEM = """You are the showrunner's assistant who writes the BIBLE of an AI-generated series: the fixed
people, places, props, looks and world rules every episode is generated from. You edit it with the user.

The bible is JSON:
{"name", "description",
 "world": {"logline", "tone",
           "look": "one verbatim style block pasted into every image prompt: camera, lenses, film stock or
                    rendering, palette (hex colors welcome), light and texture, written as positive facts",
           "rules": "world rules in positive form", "forbidden": "what never happens"},
 "characters": [{"id", "name", "role", "descriptor": "how a shot names them without the name, at most 8 words",
                 "description": "apparent age, origin, face, eyes, skin, hair, build, height, distinctive features",
                 "wardrobe": "the default outfit, exact garments, colors, materials",
                 "variants": [{"id", "label", "description": "what changes from the default look"}]}],
 "locations": [{"id", "name", "description": "architecture, materials, set dressing, period, light",
                "notes": "geometry and physics: door hinge side and swing, scale in cm, windows, light sources",
                "variants": [...]}],
 "props": [{"id", "name", "descriptor", "description": "material, color, shape", "notes": "scale in cm, details",
            "variants": [...]}]}

How to work:
- Answer the user's latest message. Change only what it asks for (or what it clearly implies); keep every
  other field exactly as it is.
- Keep the "id" of every existing element and look. New elements and looks have no "id". Every element has
  a default look {"id": "default", "label": "Default", "description": ""} first.
- Write every bible field in English (they become image prompts), whatever the user's language. Names keep
  their original form. Write "reply" in the user's language.
- Be concrete and visual, as a painter or a set designer would: no personality in descriptions, numbers for
  scale, materials and colors named. Rules in positive form ("the window stays open"), never lists of "no".
- Never claim to render images: say the user can render or upload them on each element.
- At most 24 characters, 24 locations and 24 props.

Return ONLY JSON: {"reply": "a short answer", "changes": ["one line per change"], "bible": {the whole bible}}"""


class BibleAssistantError(RuntimeError):
    pass


def _text_view(bible: dict) -> dict:
    """The bible as the assistant sees it: text and ids, whether each look has an image."""
    view = {
        "name": bible.get("name", ""),
        "description": bible.get("description", ""),
        "world": bible.get("world") or {},
    }
    for kind, plural in bibles.KINDS.items():
        view[plural] = []
        for element in bible.get(plural) or []:
            entry = {"id": element.get("id"), "name": element.get("name")}
            entry.update({field: element.get(field, "") for field in bibles.FIELDS[kind]})
            entry["variants"] = [
                {"id": v.get("id"), "label": v.get("label"), "description": v.get("description", ""),
                 "has_image": any(i.get("status") == "approved" for i in v.get("images") or [])}
                for v in element.get("variants") or []
            ]
            view[plural].append(entry)
    return view


def _messages(raw: Any) -> list[dict]:
    messages = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and str(item.get("content") or "").strip():
            messages.append({"role": item["role"], "content": str(item["content"])[:MAX_MESSAGE_CHARS]})
    messages = messages[-MAX_MESSAGES:]
    if not messages or messages[-1]["role"] != "user":
        raise BibleAssistantError("Write a message for the assistant")
    return messages


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise BibleAssistantError("The assistant returned no JSON")
    try:
        value = json.loads(match.group(0))
    except ValueError as error:
        raise BibleAssistantError(f"The assistant returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise BibleAssistantError("The assistant returned no object")
    return value


def _merge(current: dict, proposed: dict, folder: Path) -> dict:
    """The proposal, bounded, with the current images on the elements and looks it kept."""
    raw = {**proposed, "name": proposed.get("name") or current.get("name"), "style": current.get("style")}
    merged = bibles.normalize_bible(raw, folder)
    stored = {e["id"]: e for plural in bibles.KINDS.values() for e in current.get(plural) or []}
    for plural in bibles.KINDS.values():
        for element in merged[plural]:
            before = {v["id"]: v for v in (stored.get(element["id"]) or {}).get("variants") or []}
            for variant in element["variants"]:
                variant["images"] = list((before.get(variant["id"]) or {}).get("images") or [])
            element["image"] = bibles.approved_image(element)
    merged["id"] = current.get("id", "")
    merged["revision"] = current.get("revision")
    return merged


def assist(bible: dict, messages: Any, *, generate: Callable[..., str], folder: Path) -> dict:
    """One assistant turn: {"reply", "changes", "bible"} (the proposed bible, not saved)."""
    conversation = _messages(messages)
    transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = ("Current bible:\n" + json.dumps(_text_view(bible), ensure_ascii=False, indent=1)
              + "\n\nConversation:\n" + transcript)
    last_error: Exception | None = None
    for _ in range(2):
        text = generate(prompt=prompt, system_prompt=SYSTEM, max_new_tokens=6000, temperature=0.4,
                        enable_thinking=False)
        try:
            answer = _json_object(text)
            proposed = answer.get("bible") if isinstance(answer.get("bible"), dict) else {}
            merged = _merge(bible, proposed, folder) if proposed else bible
            changes = [str(c).strip()[:300] for c in answer.get("changes") or [] if str(c).strip()][:40]
            reply = str(answer.get("reply") or "").strip()[:2000] or ("Done." if changes else "No change.")
            return {"reply": reply, "changes": changes if proposed else [], "bible": merged}
        except (BibleAssistantError, bibles.BibleError) as error:
            last_error = error
    raise BibleAssistantError(str(last_error or "The assistant did not answer"))
