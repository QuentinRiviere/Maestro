"""H3 sequences: several same-scene shots in one render, split back per shot."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as dp  # noqa: E402
from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    compile_h3_clip_plans,
    validate_h3_prompt_contract,
)
from services.director.h3_sequences import (  # noqa: E402
    MAX_SEQUENCE_SHOTS,
    SEQUENCE_KEY,
    annotate_h3_sequences,
    compile_h3_sequence_prompt,
    h3_sequences_enabled,
    plan_h3_sequences,
    saved_sequence_members,
    sequence_total_frames,
    sequence_units,
    shift_local_times,
    split_sequence_video,
)
from services.director_video_strategy import BOUNDED_START_END, OMNI_REFERENCE  # noqa: E402

H3_DEF = {"director_video_strategy": BOUNDED_START_END, "frames_minimum": 124, "frames_steps": 17,
          "frames_maximum": 345}
LATTICE = {"min_frames": 124, "frame_step": 17}


def _plan(group: str, strategy: str = "continuous", **extra) -> dict:
    return {"_director_continuity_group": group, "_director_continuity_strategy": strategy, **extra}


class EnabledTests(unittest.TestCase):
    PARAMS = {"video_model": "minimax_h3", "pipeline_type": "short_film_story", "director_skill": "custom:noir"}

    def test_auto_is_on_for_custom_story_skills_on_base_h3(self):
        self.assertTrue(h3_sequences_enabled(self.PARAMS, H3_DEF))
        self.assertTrue(h3_sequences_enabled({**self.PARAMS, "video_model": "minimax_h3_full"}, H3_DEF))

    def test_flag_turns_it_off_or_on(self):
        self.assertFalse(h3_sequences_enabled({**self.PARAMS, "director_h3_sequences": "off"}, H3_DEF))
        self.assertFalse(h3_sequences_enabled({**self.PARAMS, "director_h3_sequences": False}, H3_DEF))
        builtin = {**self.PARAMS, "director_skill": ""}
        self.assertFalse(h3_sequences_enabled(builtin, H3_DEF))
        self.assertTrue(h3_sequences_enabled({**builtin, "director_h3_sequences": "on"}, H3_DEF))

    def test_ref2va_omni_audio_and_other_models_keep_per_shot_renders(self):
        on = {**self.PARAMS, "director_h3_sequences": "on"}
        self.assertFalse(h3_sequences_enabled({**on, "video_model": "minimax_h3_ref2va"}, H3_DEF))
        self.assertFalse(h3_sequences_enabled(on, {**H3_DEF, "director_video_strategy": OMNI_REFERENCE}))
        self.assertFalse(h3_sequences_enabled({**on, "audio_path": "song.wav"}, H3_DEF))
        self.assertFalse(h3_sequences_enabled({**on, "pipeline_type": "music_video"}, H3_DEF))
        self.assertFalse(h3_sequences_enabled({**on, "video_model": "ltx2_22B"}, H3_DEF))


class GroupingTests(unittest.TestCase):
    def test_total_frames_snap_up_to_the_h3_lattice(self):
        self.assertEqual(sequence_total_frames([124, 124], **LATTICE), 260)  # 248 -> 260 (17n+5)
        self.assertEqual(sequence_total_frames([60], **LATTICE), 124)

    def test_groups_consecutive_same_scene_shots_within_one_render(self):
        plans = [_plan("a"), _plan("a"), _plan("a"), _plan("b"), _plan("b"), _plan("")]
        frames = [124, 124, 124, 141, 141, 124]
        groups = plan_h3_sequences(plans, frames, max_frames=345, **LATTICE)
        # Three 124-frame shots need 372 frames: the third shot renders alone.
        self.assertEqual(groups, [[0, 1], [3, 4]])

    def test_max_four_shots_and_short_shots(self):
        plans = [_plan("a") for _ in range(6)]
        groups = plan_h3_sequences(plans, [72] * 6, max_frames=345, max_shots=4, **LATTICE)
        self.assertEqual(groups, [[0, 1, 2, 3], [4, 5]])

    def test_default_cap_is_three_shots(self):
        self.assertEqual(MAX_SEQUENCE_SHOTS, 3)
        plans = [_plan("a") for _ in range(7)]
        groups = plan_h3_sequences(plans, [72] * 7, max_frames=345, **LATTICE)
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5]])

    def test_never_across_an_extend_previous_link(self):
        plans = [_plan("a"), _plan("a"), _plan("a", "extend_previous"), _plan("a"), _plan("a")]
        groups = plan_h3_sequences(plans, [100] * 5, max_frames=345, **LATTICE)
        # Shots 2 and 3 keep their last-frame chain; 4 and 5 still group.
        self.assertEqual(groups, [[3, 4]])

    def test_duration_split_segments_stay_alone(self):
        plans = [_plan("a") for _ in range(4)]
        groups = plan_h3_sequences(plans, [100] * 4, max_frames=345, linked=[False, True, False, False], **LATTICE)
        self.assertEqual(groups, [[2, 3]])

    def test_annotation_records_offsets_and_parks_keep_ranges(self):
        plans = [_plan("a", _director_keep_sec=[1.0, 3.0]), _plan("a"), _plan("b", _director_keep_sec=[0, 2.5])]
        annotate_h3_sequences(plans, [[0, 1]], [124, 130, 124], fps=24, **LATTICE)
        first, second = plans[0][SEQUENCE_KEY], plans[1][SEQUENCE_KEY]
        self.assertEqual((first["start_frame"], second["start_frame"]), (0, 124))
        self.assertEqual(second["offset_sec"], round(124 / 24, 3))
        self.assertEqual(first["total_frames"], 260)
        self.assertNotIn("_director_keep_sec", plans[0])  # members render full length
        self.assertEqual(plans[2]["_director_keep_sec"], [0, 2.5])
        self.assertNotIn(SEQUENCE_KEY, plans[2])
        # Sequences off on the next run: the keep range comes back.
        annotate_h3_sequences(plans, [], [124, 130, 124], fps=24, **LATTICE)
        self.assertEqual(plans[0]["_director_keep_sec"], [1.0, 3.0])
        self.assertNotIn(SEQUENCE_KEY, plans[0])

    def test_units_and_saved_members(self):
        self.assertEqual(sequence_units(5, [[1, 2, 3]]), [[0], [1, 2, 3], [4]])
        plans = [_plan("a"), _plan("a"), _plan("a")]
        annotate_h3_sequences(plans, [[1, 2]], [124, 124, 124], fps=24, **LATTICE)
        self.assertEqual(saved_sequence_members(plans, 2), [1, 2])
        self.assertEqual(saved_sequence_members(plans, 0), [])
        plans[2][SEQUENCE_KEY]["id"] = "other"
        self.assertEqual(saved_sequence_members(plans, 1), [])


def _compiled_members() -> list[dict]:
    look = "Grainy 16mm, warm tungsten practicals."
    plans = [
        {
            "video_prompt": (
                "Kitchen at night. Action: 0.0-2.0s: Monica sets down a mug, then looks up. "
                "overall_soundscape: Fridge hum. non_diegetic_music: N/A."
            ),
            "_director_subjects_on_screen": [{"character_id": "monica", "speaker_name": "Monica"}],
            "_director_dialogue_beats": [{"speaker_id": "monica", "spoken_text": "You're late."}],
            "_director_look": look,
        },
        {
            "video_prompt": (
                "Kitchen at night, reverse angle. Action: Ross freezes in the doorway. "
                "overall_soundscape: Fridge hum and rain on the window. non_diegetic_music: N/A."
            ),
            "_director_subjects_on_screen": [{"character_id": "ross", "speaker_name": "Ross"}],
            "_director_dialogue_beats": [{"speaker_id": "ross", "spoken_text": "The train stopped."}],
            "_director_look": look,
        },
    ]
    compile_h3_clip_plans(plans, prompt_modes=["i2va", "i2va"], durations=[124 / 24, 136 / 24])
    return plans


class SequencePromptTests(unittest.TestCase):
    def test_one_prompt_with_timed_hard_cuts_passes_the_h3_contract(self):
        plans = _compiled_members()
        offset = round(124 / 24, 3)
        prompt = compile_h3_sequence_prompt(plans, mode="i2va", offsets_sec=[0.0, offset],
                                            total_duration_sec=260 / 24)
        self.assertTrue(prompt.startswith("For the target video, at 0.00 seconds"))
        self.assertIn("integrated_multimodal_description: [Shot 1] ", prompt)
        self.assertIn("[Shot 2] At 00:05.167, hard cut.", prompt)
        self.assertEqual(prompt.count("[Shot 1]"), 2)  # header + description
        self.assertLess(prompt.index("You're late."), prompt.index("The train stopped."))
        self.assertEqual(prompt.count("<d>"), 2)
        self.assertEqual(prompt.count("Visual style:"), 1)
        self.assertIn("Fridge hum; Fridge hum and rain on the window.", prompt)
        beats = plans[0]["_director_dialogue_beats"] + plans[1]["_director_dialogue_beats"]
        self.assertEqual(validate_h3_prompt_contract(prompt, beats, mode="i2va"), [])

    def test_fl2va_header_names_the_last_shot(self):
        plans = _compiled_members()
        prompt = compile_h3_sequence_prompt(plans, mode="fl2va", offsets_sec=[0.0, 5.167],
                                            total_duration_sec=260 / 24)
        self.assertIn("Picture 2 (from Shot 2)", prompt)
        self.assertIn("10.83-second mark", prompt)

    def test_rejects_ref2va_and_dialogue_drift(self):
        plans = _compiled_members()
        with self.assertRaises(H3DialogueContractError):
            compile_h3_sequence_prompt(plans, mode="ref2va", offsets_sec=[0, 5], total_duration_sec=10)
        plans[1]["_director_dialogue_beats"] = [{"speaker_id": "ross", "spoken_text": "Something else."}]
        with self.assertRaises(H3DialogueContractError):
            compile_h3_sequence_prompt(plans, mode="t2va", offsets_sec=[0, 5], total_duration_sec=10)

    def test_local_times_move_onto_the_sequence_timeline(self):
        body = ("Action: 0.0-2.0s: he turns; 2.5s: the door slams. <d>[English] At 1.5s sharp.</d> "
                "Dialogue timing: mouths stay closed with no human voice from 0.00 to 0.40 seconds; "
                "everyone remains silent through 5.17 seconds.")
        shifted = shift_local_times(body, 5.0)
        self.assertIn("5.0-7.0s: he turns; 7.5s: the door slams.", shifted)
        self.assertIn("<d>[English] At 1.5s sharp.</d>", shifted)  # spoken words untouched
        self.assertIn("from 5.00 to 5.40 seconds", shifted)
        self.assertIn("through 10.17 seconds.", shifted)


class RenderUnitTests(unittest.TestCase):
    SEP = "\n---CLIP_BOUNDARY---\n"

    def test_collapse_builds_one_task_per_sequence(self):
        gen = {
            "prompt": self.SEP.join(["p0", "p1", "p2", "p3"]),
            "per_clip_frames": [124, 124, 130, 141],
            "image_start": ["s0", "s1", "s2", "s3"],
            "image_end": ["", "", "e2", ""],
            "per_clip_prompt_modes": [0, 0, 0, 0],
            "video_length": 519, "sliding_window_size": 141,
        }
        dp._collapse_h3_sequence_params(gen, [[0], [1, 2], [3]], {1: "SEQ"}, [0, 260, 0], self.SEP)
        self.assertEqual(gen["prompt"].split(self.SEP), ["p0", "SEQ", "p3"])
        self.assertEqual(gen["per_clip_frames"], [124, 260, 141])
        self.assertEqual(gen["image_start"], ["s0", "s1", "s3"])
        self.assertEqual(gen["image_end"], ["", "e2", ""])
        self.assertEqual(gen["per_clip_prompt_modes"], [0, 0, 0])
        self.assertEqual((gen["video_length"], gen["sliding_window_size"]), (525, 260))

    def test_outputs_map_back_to_one_file_per_shot(self):
        plans = [_plan("a"), _plan("b"), _plan("b"), _plan("b"), _plan("c")]
        annotate_h3_sequences(plans, [[1, 2, 3]], [124, 80, 80, 80, 124], fps=24, **LATTICE)
        calls = []

        def fake_split(source, ranges, fps, outputs):
            calls.append((os.path.basename(source), list(ranges), fps))
            return outputs

        original = dp.split_sequence_video
        dp.split_sequence_video = fake_split
        try:
            outputs = dp._DirectorOutputs(["u0.mp4", "u1.mp4", "u2.mp4", "x_multiclip.mp4"],
                                          clip_output_files={0: "u0.mp4", 1: "u1.mp4", 2: "u2.mp4"})
            mapped = dp._split_h3_sequence_outputs(outputs, [[0], [1, 2, 3], [4]], plans, "/tmp/out")
        finally:
            dp.split_sequence_video = original
        self.assertEqual(calls, [("u1.mp4", [(0, 80), (80, 160), (160, 240)], 24.0)])
        self.assertEqual(mapped.clip_output_files,
                         {0: "u0.mp4", 1: "u1_shot1.mp4", 2: "u1_shot2.mp4", 3: "u1_shot3.mp4", 4: "u2.mp4"})
        self.assertEqual(list(mapped)[-1], "x_multiclip.mp4")
        self.assertEqual(dp._clip_video_slots(mapped, 5),
                         ["u0.mp4", "u1_shot1.mp4", "u1_shot2.mp4", "u1_shot3.mp4", "u2.mp4"])
        self.assertEqual(plans[2][SEQUENCE_KEY]["render_file"], "u1.mp4")

    def test_partial_outputs_keep_shot_indices(self):
        plans = [_plan("a"), _plan("a"), _plan("b")]
        annotate_h3_sequences(plans, [[0, 1]], [124, 124, 124], fps=24, **LATTICE)
        original = dp.split_sequence_video
        dp.split_sequence_video = lambda *args: args[-1]
        try:
            mapped = dp._split_h3_sequence_outputs(
                dp._DirectorOutputs(["u1.mp4"], clip_output_files={1: "u1.mp4"}), [[0, 1], [2]], plans, "/o",
            )
        finally:
            dp.split_sequence_video = original
        self.assertEqual(dp._clip_video_slots(mapped, 3), [None, None, "u1.mp4"])

    def test_rerun_of_a_member_reruns_the_sequence(self):
        plans = [_plan("a"), _plan("a"), _plan("b")]
        annotate_h3_sequences(plans, [[0, 1]], [124, 124, 124], fps=24, **LATTICE)
        state = {"clips": [{**plan, "video_prompt": "p"} for plan in plans]}
        seen = {}
        originals = (dp.load_pipeline_state, dp._rerun_h3_sequence_video)
        dp.load_pipeline_state = lambda out_dir, pid: state
        dp._rerun_h3_sequence_video = lambda out_dir, pid, st, members, index, override: seen.update(
            members=members, index=index) or {"sequence_rerun": True}
        try:
            result = dp._rerun_clip_video_impl("/o", "pid", 1)
        finally:
            dp.load_pipeline_state, dp._rerun_h3_sequence_video = originals
        self.assertEqual(result, {"sequence_rerun": True})
        self.assertEqual(seen, {"members": [0, 1], "index": 1})


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class SplitVideoTests(unittest.TestCase):
    def test_split_is_frame_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "seq.mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=3",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source],
                check=True, capture_output=True,
            )
            outputs = [os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")]
            split_sequence_video(source, [(0, 29), (29, 72)], 24.0, outputs)
            counts = []
            for path in outputs:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                    check=True, capture_output=True, text=True,
                )
                counts.append(int(probe.stdout.strip()))
            self.assertEqual(counts, [29, 43])

    def test_scene_detection_finds_hard_cuts(self):
        from services.director.h3_sequences import detect_scene_cuts

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "cuts.mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24:duration=2",
                 "-f", "lavfi", "-i", "smptebars=size=160x120:rate=24:duration=1.5",
                 "-f", "lavfi", "-i", "mandelbrot=size=160x120:rate=24",
                 "-filter_complex", "[2]trim=duration=1,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1:a=0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", source],
                check=True, capture_output=True,
            )
            self.assertEqual(detect_scene_cuts(source), [2.0, 3.5])
            with self.assertRaises(RuntimeError):
                detect_scene_cuts(os.path.join(tmp, "missing.mp4"))


if __name__ == "__main__":
    unittest.main()


class SequenceLengthTests(unittest.TestCase):
    def test_members_last_their_planned_length_and_lone_shots_keep_only_theirs(self):
        from services import director_pipeline as pipeline

        plans = [{}, {}, {}]
        planned = [{}, {}, {}]
        # Shots 1-2 form a sequence; shot 3 is alone and renders the 124-frame minimum.
        pipeline._fit_h3_sequence_lengths(plans, planned, [[0, 1]], [48, 72, 60], [124, 124, 124], 24.0)
        self.assertEqual([p["duration_sec"] for p in planned], [2.0, 3.0, 124 / 24])
        self.assertEqual(planned[2]["start"], 5.0)
        self.assertNotIn("_director_keep_sec", plans[0])
        self.assertEqual(plans[2]["_director_keep_sec"], [0.0, 2.5])

    def test_planned_length_comes_from_the_planner_even_on_a_fitted_timeline(self):
        fitted = {"duration_frames": 124, "duration_sec": 124 / 24, "_director_duration_sec": 4.0}
        self.assertEqual(dp._planned_shot_frames(fitted, 124, 24.0), 96)
        self.assertEqual(dp._planned_shot_frames({"duration_frames": 72}, 72, 24.0), 72)

    def test_a_sequence_run_keeps_short_planned_shots_unmerged(self):
        from unittest.mock import patch

        from services import director_pipeline as pipeline

        clips = [{"start": 0, "end": 2, "duration_sec": 2}, {"start": 2, "end": 4.5, "duration_sec": 2.5}]
        params = {"video_model": "minimax_h3", "pipeline_type": "short_film_story", "director_skill": "custom:k"}
        model_def = {"fps": 24, "frames_minimum": 124, "frames_maximum": 345, "frames_steps": 17}
        from types import SimpleNamespace

        with patch.object(pipeline, "_wgp", SimpleNamespace(get_model_def=lambda name: model_def)), \
                patch.object(pipeline, "video_strategy", return_value=pipeline.BOUNDED_START_END), \
                patch.object(pipeline, "h3_sequences_enabled", return_value=True):
            plans, planned = pipeline.prepare_director_timeline(params, [{}, {}], clips)
        self.assertEqual(len(planned), 2)


class CutSnappingTests(unittest.TestCase):
    """Planned cuts snap to the cuts H3 actually made (measured runs, 24 fps)."""

    FPS = 24.0

    def snap(self, planned_sec, detected_sec, total_sec=12.0, **extra):
        from services.director.h3_sequences import snap_sequence_cuts

        cuts, ignored = snap_sequence_cuts(
            [round(t * self.FPS) for t in planned_sec], detected_sec,
            fps=self.FPS, total_frames=round(total_sec * self.FPS), **extra,
        )
        return [round(c / self.FPS, 2) for c in cuts], ignored

    def test_drift_snaps_to_the_detected_cuts(self):
        cuts, ignored = self.snap([4.0, 7.0, 10.5], [3.9, 6.19, 9.52])
        self.assertEqual(cuts, [3.92, 6.21, 9.5])
        self.assertEqual(ignored, [])

    def test_extra_detected_cuts_are_ignored(self):
        cuts, ignored = self.snap([4.5, 7.5, 10.0], [2.1, 2.4, 5.6, 7.85, 10.56])
        self.assertEqual(cuts, [5.58, 7.83, 10.54])
        self.assertEqual(ignored, [2.1, 2.4])
        cuts, ignored = self.snap([3.5, 6.5, 8.5], [3.4, 4.65, 6.56, 8.94, 10.06])
        self.assertEqual(cuts, [3.42, 6.54, 8.96])
        self.assertEqual(ignored, [4.65, 10.06])

    def test_missing_cut_keeps_its_planned_time(self):
        cuts, ignored = self.snap([3.0, 6.0], [2.9])
        self.assertEqual(cuts, [2.92, 6.0])
        cuts, _ = self.snap([3.0, 6.0], [])
        self.assertEqual(cuts, [3.0, 6.0])

    def test_cut_beyond_the_tolerance_is_not_used(self):
        cuts, ignored = self.snap([3.0], [4.5])
        self.assertEqual((cuts, ignored), ([3.0], [4.5]))
        cuts, _ = self.snap([3.0], [4.5], tolerance_sec=2.0)
        self.assertEqual(cuts, [4.5])

    def test_each_shot_keeps_the_minimum_length(self):
        # Snapping both cuts would leave a 0.25 s middle shot: only one snaps.
        cuts, ignored = self.snap([4.0, 5.0], [4.4, 4.65])
        self.assertGreaterEqual(cuts[1] - cuts[0], 0.5)
        self.assertEqual(len(ignored), 1)
        # A detected cut too close to the end of the render is not used.
        cuts, _ = self.snap([3.0], [3.0], total_sec=3.2)
        self.assertEqual(cuts, [3.0])
        cuts, ignored = self.snap([2.5], [2.9], total_sec=3.2)
        self.assertEqual((cuts, ignored), ([2.5], [2.9]))

    def test_cuts_stay_strictly_increasing(self):
        # Each planned cut's nearest detected cut is the other's: no crossing.
        cuts, _ = self.snap([4.0, 5.0], [5.1, 3.9])
        self.assertEqual(cuts, [3.92, 5.08])
        cuts, _ = self.snap([4.0, 4.8], [5.2])
        self.assertEqual(sorted(cuts), cuts)
        self.assertEqual(len(set(cuts)), len(cuts))

    def test_ranges_follow_snapped_cuts_and_keep_the_last_shot_length(self):
        from services.director.h3_sequences import sequence_frame_ranges

        plans = [_plan("a"), _plan("a"), _plan("a")]
        annotate_h3_sequences(plans, [[0, 1, 2]], [72, 72, 72], fps=24, **LATTICE)
        records = [plan[SEQUENCE_KEY] for plan in plans]
        ranges, ignored = sequence_frame_ranges(records, None)
        self.assertEqual((ranges, ignored), ([(0, 72), (72, 144), (144, 216)], []))
        ranges, ignored = sequence_frame_ranges(records, [2.9, 5.5, 8.0])
        self.assertEqual(ranges, [(0, 70), (70, 132), (132, 204)])
        self.assertEqual(ignored, [8.0])
        # A late last cut is clamped to the rendered length (226 frames).
        ranges, _ = sequence_frame_ranges(records, [3.0, 7.0])
        self.assertEqual(ranges[-1], (168, 226))


class SnappedSplitTests(unittest.TestCase):
    def _plans(self):
        plans = [_plan("a"), _plan("a"), _plan("a")]
        annotate_h3_sequences(plans, [[0, 1, 2]], [96, 72, 72], fps=24, **LATTICE)
        return plans

    def _split(self, plans, detector):
        from unittest.mock import patch

        calls = []

        def fake_split(source, ranges, fps, outputs):
            calls.append(list(ranges))
            return outputs

        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "seq.mp4"), "wb").close()
            with patch.object(dp, "detect_scene_cuts", detector), \
                    patch.object(dp, "split_sequence_video", fake_split):
                names = dp._split_h3_sequence_file("seq.mp4", plans, tmp)
        return names, calls

    def test_split_uses_snapped_frames_and_records_actual_ranges(self):
        plans = self._plans()
        names, calls = self._split(plans, lambda source: [1.2, 3.9, 6.6])
        self.assertEqual(names, ["seq_shot1.mp4", "seq_shot2.mp4", "seq_shot3.mp4"])
        self.assertEqual(calls, [[(0, 94), (94, 158), (158, 230)]])
        records = [plan[SEQUENCE_KEY] for plan in plans]
        self.assertEqual([r["actual_start_frame"] for r in records], [0, 94, 158])
        self.assertEqual([r["actual_frames"] for r in records], [94, 64, 72])
        self.assertEqual([r["start_frame"] for r in records], [0, 96, 168])
        self.assertEqual(records[1]["detected_cuts"], [1.2, 3.9, 6.6])
        self.assertEqual(records[0]["cuts_source"], "scene_detect")

    def test_failed_detection_splits_at_planned_frames(self):
        def broken(source):
            raise RuntimeError("no ffmpeg")

        plans = self._plans()
        _, calls = self._split(plans, broken)
        self.assertEqual(calls, [[(0, 96), (96, 168), (168, 240)]])
        records = [plan[SEQUENCE_KEY] for plan in plans]
        self.assertEqual([r["actual_frames"] for r in records], [96, 72, 72])
        self.assertEqual(records[0]["cuts_source"], "planned")
        self.assertIsNone(records[0]["detected_cuts"])

    def test_timeline_uses_actual_lengths(self):
        plans = self._plans() + [_plan("b", _director_duration_sec=5.0)]
        self._split(plans[:3], lambda source: [3.9, 6.6])
        planned = [
            {"start": 0.0, "end": 4.0, "duration_sec": 4.0},
            {"start": 4.0, "end": 7.0, "duration_sec": 3.0},
            {"start": 7.0, "end": 10.0, "duration_sec": 3.0},
            {"start": 10.0, "end": 15.0, "duration_sec": 5.0},
        ]
        self.assertTrue(dp._apply_h3_sequence_actual_lengths(plans, planned))
        self.assertEqual([round(p["duration_sec"], 3) for p in planned], [3.917, 2.667, 3.0, 5.0])
        self.assertEqual([round(p["start"], 3) for p in planned], [0.0, 3.917, 6.583, 9.583])
        self.assertEqual(round(planned[-1]["end"], 3), 14.583)
        self.assertEqual(planned[1]["duration_frames"], 64)
        self.assertEqual(plans[1]["_director_duration_sec"], 64 / 24)
        self.assertFalse(dp._apply_h3_sequence_actual_lengths(plans, planned))
