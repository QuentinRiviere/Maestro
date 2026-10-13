"""
LTX Text-to-Video Renderer — the most descriptive mode.

Used when the model must invent the whole shot from scratch.
Pass 1: assembles all ShotPlan fields into a tagged draft
Pass 2: LLM rewrites into a flowing cinematic paragraph
"""

from __future__ import annotations
from typing import Optional

from ..schema import ShotPlan, ProductionPlan
from .base import BaseRenderer


class LtxT2VRenderer(BaseRenderer):
    mode = "t2v"

    def _refinement_system_prompt(self, shot: ShotPlan, **context) -> str:
        return "Rewrite into one flowing paragraph, present tense, 4-8 sentences. Include shot type, setting, characters by appearance, action, camera movement, ending beat. No names outside dialogue. Output ONLY the prompt."

    def render(
        self,
        shot: ShotPlan,
        plan: Optional[ProductionPlan] = None,
        **context,
    ) -> str:
        """Pass 1: assemble tagged draft for T2V."""
        parts = []

        # Style and shot type
        style_shot = []
        if shot.visual_style:
            style_shot.append(shot.visual_style)
        style_shot.append(shot.camera_plan.framing)
        if shot.camera_plan.angle:
            style_shot.append(shot.camera_plan.angle)
        parts.append(f"SHOT: {', '.join(style_shot)}")

        # Setting
        setting = self._setting_text(shot)
        if setting:
            parts.append(f"SETTING: {setting}")

        scene_map = self._scene_map_text(shot)
        if scene_map:
            parts.append(f"SCENE MAP: {scene_map}")

        # Characters
        subjects = self._subjects_text(shot, plan)
        if subjects:
            spatial = f", {shot.spatial_setup}" if shot.spatial_setup else ""
            parts.append(f"SUBJECTS: {subjects}{spatial}")

        # Action beats
        action = self._action_text(shot, with_performance=False)
        if action:
            parts.append(f"ACTION: {action}")

        # Dialogue
        dialogue = self._dialogue_text(shot, plan)
        if dialogue:
            parts.append(f"DIALOGUE: {dialogue}")

        if shot.performance_beats:
            parts.append(f"PERFORMANCE: {'. '.join(shot.performance_beats)}")

        # Camera movement
        if shot.camera_plan.movement:
            parts.append(f"CAMERA: {shot.camera_plan.movement} ({shot.camera_plan.movement_intensity})")

        light = self._light_lock_text(shot)
        if light:
            parts.append(f"LIGHT: {light}")

        # Audio/ambient
        audio = self._audio_text(shot)
        if audio:
            parts.append(f"AUDIO: {audio}")

        locks = self._locks_text(shot)
        if locks:
            parts.append(f"LOCKS: {locks}")

        # Ending beat
        if shot.ending_beat:
            parts.append(f"ENDING: {shot.ending_beat}")

        return " | ".join(parts)
