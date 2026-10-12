"""Style contract: the medium and craft rules a Director project aims for.

The goal of quality control is not realism. A cartoon need not be realistic,
but no style may look like a failed generation. The contract states what is
intended (medium, look, motion language, allowed stylisation) so the planner
writes for it, the judge separates intended stylisation from accidental
artifacts, and finishing matches the medium.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

FIELDS = ("medium", "visual_language", "motion_language", "allowed_stylization", "must_avoid", "finishing")
DEFAULT_CONTRACT = {
    "medium": "unspecified",
    "visual_language": "",
    "motion_language": "",
    "allowed_stylization": [],
    "must_avoid": [],
    "finishing": {"film_grain": False, "upscale": False},
}

SYSTEM_PROMPT = """You define the STYLE CONTRACT of a video project from its brief. The brief may ask for
any style: live action, documentary, commercial, 3D animation, 2D anime, cartoon, claymation, stop-motion,
VHS, pixel art, painterly... Never default to realism; infer the medium the brief implies, or the most fitting
one when it is silent.
Return ONLY JSON:
{"medium": "short name of the production medium",
 "visual_language": "look, palette, lighting and lens or rendering style of a professional production in it",
 "motion_language": "how things move in this medium (e.g. real-camera inertia, squash and stretch, animated on twos, stop-motion stepping)",
 "allowed_stylization": ["deliberate departures from realism that are CORRECT here"],
 "must_avoid": ["what would look like a failed AI generation in THIS medium"],
 "finishing": {"film_grain": true|false, "upscale": true|false}}
Keep each text field under 40 words."""


def _strings(value: Any) -> list[str]:
    return [str(item).strip()[:160] for item in value if str(item).strip()][:8] if isinstance(value, list) else []


def parse_contract(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON in style contract")
    raw = json.loads(match.group(0))
    finishing = raw.get("finishing") if isinstance(raw.get("finishing"), dict) else {}
    return {
        "medium": str(raw.get("medium") or DEFAULT_CONTRACT["medium"]).strip()[:80],
        "visual_language": str(raw.get("visual_language") or "").strip()[:400],
        "motion_language": str(raw.get("motion_language") or "").strip()[:400],
        "allowed_stylization": _strings(raw.get("allowed_stylization")),
        "must_avoid": _strings(raw.get("must_avoid")),
        "finishing": {"film_grain": bool(finishing.get("film_grain")), "upscale": bool(finishing.get("upscale"))},
    }


def derive_style_contract(brief: str, generate: Callable[..., str]) -> dict:
    """Ask the planning LLM for the contract; fall back to an open contract."""
    if not str(brief or "").strip():
        return dict(DEFAULT_CONTRACT)
    try:
        text = generate(prompt=f"Brief:\n{brief.strip()[:4000]}", system_prompt=SYSTEM_PROMPT,
                        max_new_tokens=700, temperature=0.3, enable_thinking=False)
        return parse_contract(text)
    except Exception as error:  # noqa: BLE001 - the contract is guidance, never a blocker
        print(f"[Director] Style contract unavailable, continuing without it: {error}")
        return dict(DEFAULT_CONTRACT)


def contract_block(contract: dict | None) -> str:
    """Planner instruction block; empty when no style was established."""
    if not contract or contract.get("medium") in (None, "", "unspecified"):
        return ""
    lines = [
        "STYLE CONTRACT — write every shot for this medium and keep it consistent across shots.",
        f"Medium: {contract['medium']}",
    ]
    if contract.get("visual_language"):
        lines.append(f"Visual language: {contract['visual_language']}")
    if contract.get("motion_language"):
        lines.append(f"Motion language: {contract['motion_language']}")
    if contract.get("allowed_stylization"):
        lines.append("Intended stylisation: " + "; ".join(contract["allowed_stylization"]))
    if contract.get("must_avoid"):
        lines.append("Avoid (reads as a failed generation here): " + "; ".join(contract["must_avoid"]))
    lines.append("Favour what video models render cleanly in this medium: one main action per short shot, "
                 "stable framing during fine manipulation, and no element that must morph unless the style asks for it.")
    return "\n".join(lines)
