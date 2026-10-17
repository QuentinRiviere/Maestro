"""Keep windows are capped by scene type and the episode judge watches the tight edit."""
import os
import sys
import tempfile
import unittest

_APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import edit_cut, video_judge  # noqa: E402


class KeepCapTests(unittest.TestCase):
    def test_a_long_reaction_window_is_shortened_around_its_centre(self):
        self.assertEqual(edit_cut.cap_keep((0.5, 6.0), "reaction", 6.5), (1.75, 4.75))

    def test_a_window_inside_the_cap_is_kept(self):
        self.assertEqual(edit_cut.cap_keep((1.0, 3.5), "insert", 6.0), (1.0, 3.5))
        self.assertEqual(edit_cut.cap_keep((0.5, 4.5), "exchange", 6.0), (0.5, 4.5))

    def test_a_whole_shot_over_its_cap_gets_a_centred_window(self):
        self.assertEqual(edit_cut.cap_keep(None, "insert", 6.0), (1.5, 4.5))
        self.assertIsNone(edit_cut.cap_keep(None, "establishing", 5.0))
        self.assertIsNone(edit_cut.cap_keep(None, "unknown", 5.0))

    def test_the_window_never_leaves_the_shot(self):
        self.assertEqual(edit_cut.cap_keep((2.0, 7.0), "reaction", 6.0), (3.0, 6.0))

    def test_plan_keeps_reads_the_plans(self):
        plans = [{"_director_keep_sec": [0.5, 5.5], "_director_duration_sec": 6.0, "_director_scene_type": "reaction"},
                 {"_director_keep_sec": [0.0, 6.0], "_director_duration_sec": 6.0, "_director_scene_type": "establishing"}]
        self.assertEqual(edit_cut.plan_keeps(plans), [(1.5, 4.5), (0.5, 5.5)])


class EpisodeEditTests(unittest.TestCase):
    def test_the_tight_edit_is_the_episode_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("x_multiclip.mp4", "x_edit.mp4"):
                open(os.path.join(tmp, name), "wb").close()
            paths, kind = video_judge.episode_source([], ["x_multiclip.mp4"], tmp, edit_file="x_edit.mp4")
            self.assertEqual((os.path.basename(paths[0]), kind), ("x_edit.mp4", "edit"))
            paths, kind = video_judge.episode_source([], ["x_multiclip.mp4"], tmp, edit_file="missing.mp4")
            self.assertEqual(kind, "multiclip")

    def test_the_timeline_follows_the_kept_windows(self):
        clips = [{"planned_clip": {"duration_sec": 6.0}}, {"planned_clip": {"duration_sec": 6.0}}]
        timeline = video_judge.episode_timeline(clips, "/nowhere", probe=lambda p: None, keeps=[(1.5, 4.5), None])
        self.assertEqual([(t["start_sec"], t["end_sec"]) for t in timeline], [(0.0, 3.0), (3.0, 9.0)])
