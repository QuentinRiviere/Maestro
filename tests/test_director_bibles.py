"""Bibles keep a series' cast, style and world rules for every project made from them."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director import bibles, casting  # noqa: E402

GIRL = {"name": "Eun-seo", "role": "lead", "descriptor": "the teen girl with see-through bangs",
        "description": "About 17, Korean, long black hair with see-through bangs.",
        "wardrobe": "navy blazer, grey pleated skirt"}
HALL = {"name": "Hallway", "description": "School corridor, lockers on the right, windows on the left."}


class BibleStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = os.path.join(self.tmp.name, "project")
        os.makedirs(self.project)

    def _image(self, name):
        with open(os.path.join(self.project, name), "wb") as handle:
            handle.write(b"jpeg")
        return name

    def test_a_project_cast_becomes_a_bible_with_its_own_images(self):
        cast = {"characters": [{**GIRL, "image": self._image("girl.jpg")}],
                "locations": [{**HALL, "image": self._image("hall.jpg")}]}
        bible = bibles.add_cast(cast, [self.project], name="Spring Rain", style={"medium": "live action"})

        self.assertEqual(bible["id"], "spring_rain")
        self.assertEqual(bible["revision"], 2)
        self.assertEqual(bible["style"], {"medium": "live action"})
        image = bible["characters"][0]["image"]
        self.assertTrue(image.startswith("char_eun_seo_"))
        self.assertIsNotNone(bibles.image_path("spring_rain", image))
        # The bible no longer depends on the project's folder.
        os.remove(os.path.join(self.project, "girl.jpg"))
        self.assertIsNotNone(bibles.image_path("spring_rain", image))
        self.assertEqual([b["id"] for b in bibles.list_bibles()], ["spring_rain"])

    def test_saving_a_later_cast_replaces_entries_by_name_and_keeps_the_others(self):
        bibles.add_cast({"characters": [GIRL], "locations": [HALL]}, [self.project], name="Spring Rain")
        boy = {"name": "Ji-ho", "description": "About 18, Korean, short black hair."}
        older_girl = {**GIRL, "wardrobe": "winter coat"}
        bible = bibles.add_cast({"characters": [older_girl, boy]}, [self.project], bible_id="spring_rain")

        self.assertEqual([c["name"] for c in bible["characters"]], ["Eun-seo", "Ji-ho"])
        self.assertEqual(bible["characters"][0]["wardrobe"], "winter coat")
        self.assertEqual([l["name"] for l in bible["locations"]], ["Hallway"])

    def test_an_episode_picks_entries_and_gets_copies_in_its_folder(self):
        cast = {"characters": [{**GIRL, "image": self._image("girl.jpg")}, {"name": "Ji-ho", "description": "boy"}],
                "locations": [{**HALL, "image": self._image("hall.jpg")}]}
        bible = bibles.add_cast(cast, [self.project], name="Spring Rain")
        episode = os.path.join(self.tmp.name, "episode2")

        picked = bibles.project_cast(bible, episode, characters=["eun-seo"])

        self.assertEqual([c["name"] for c in picked["characters"]], ["Eun-seo"])
        self.assertTrue(os.path.isfile(os.path.join(episode, picked["characters"][0]["image"])))
        self.assertEqual(len(picked["locations"]), 1)

    def test_ids_and_image_names_cannot_leave_the_bible_folder(self):
        with self.assertRaises(bibles.BibleError):
            bibles.get_bible("../secrets") or bibles._folder("../secrets")
        bibles.save_bible({"name": "Spring Rain"})
        self.assertIsNone(bibles.image_path("spring_rain", "../../bible.json"))
        self.assertIsNone(bibles.image_path("spring_rain", "bible.json"))

    def test_world_rules_reach_the_planner(self):
        bible = bibles.save_bible({"name": "Spring Rain", "world": {"logline": "First love in a Busan high school",
                                                                    "forbidden": "phones before episode 3"}})
        block = bibles.world_block(bible)
        self.assertIn("Series premise: First love in a Busan high school", block)
        self.assertIn("Never: phones before episode 3", block)
        self.assertEqual(bibles.world_block(bibles.save_bible({"name": "Empty"})), "")


class BibleLookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.upload = os.path.join(self.tmp.name, "sheet.png")
        with open(self.upload, "wb") as handle:
            handle.write(b"png")
        bible = bibles.save_bible({
            "name": "Spring Rain",
            "characters": [{**GIRL, "variants": [{"id": "default"},
                                                 {"id": "winter", "label": "Winter", "description": "long camel coat"}]}],
            "locations": [HALL, {"name": "Rooftop", "description": "School rooftop, green fence."}],
            "props": [{"name": "Umbrella", "description": "clear plastic umbrella", "notes": "90 cm"}],
        })
        self.girl = bible["characters"][0]["id"]
        self.hall = bible["locations"][0]["id"]
        self.roof = bible["locations"][1]["id"]

    def _bible(self):
        return bibles.get_bible("spring_rain")

    def test_the_first_image_of_a_look_is_approved_and_later_ones_are_candidates(self):
        bibles.add_image("spring_rain", self.girl, "default", self.upload)
        bible = bibles.add_image("spring_rain", self.girl, "default", self.upload)
        images = bible["characters"][0]["variants"][0]["images"]
        self.assertEqual([i["status"] for i in images], ["approved", "candidate"])
        bible = bibles.approve_image("spring_rain", self.girl, "default", images[1]["id"])
        self.assertEqual(bible["characters"][0]["image"], images[1]["file"])
        bible = bibles.delete_image("spring_rain", self.girl, "default", images[0]["id"])
        self.assertEqual(len(bible["characters"][0]["variants"][0]["images"]), 1)

    def test_a_render_slot_becomes_a_candidate_or_a_failure(self):
        _, slot = bibles.start_render("spring_rain", self.girl, "default")
        self.assertEqual(self._bible()["characters"][0]["variants"][0]["images"][0]["status"], "rendering")
        bible = bibles.finish_render("spring_rain", self.girl, "default", slot, source=self.upload)
        self.assertEqual(bible["characters"][0]["variants"][0]["images"][0]["status"], "approved")
        _, slot = bibles.start_render("spring_rain", self.girl, "winter")
        bible = bibles.finish_render("spring_rain", self.girl, "winter", slot, error="worker down")
        self.assertEqual(bible["characters"][0]["variants"][1]["images"][0]["error"], "worker down")

    def test_a_render_cut_by_a_restart_reads_as_failed(self):
        bibles.start_render("spring_rain", self.girl, "default")
        with patch.object(bibles, "RENDER_STALE_SECONDS", -1):
            image = self._bible()["characters"][0]["variants"][0]["images"][0]
        self.assertEqual((image["status"], image["error"]), ("failed", "Interrupted"))

    def test_editing_the_text_keeps_the_images(self):
        bibles.add_image("spring_rain", self.girl, "default", self.upload)
        bible = self._bible()
        bible["characters"][0]["description"] = "About 18 now."
        bible["characters"][0]["variants"][0]["images"] = []  # a stale client copy
        saved = bibles.save_bible(bible, bible_id="spring_rain")
        self.assertEqual(saved["characters"][0]["description"], "About 18 now.")
        self.assertEqual(len(saved["characters"][0]["variants"][0]["images"]), 1)

    def test_new_looks_and_places_are_edited_from_approved_images(self):
        bibles.add_image("spring_rain", self.girl, "default", self.upload)
        bibles.add_image("spring_rain", self.hall, "default", self.upload)
        bible = self._bible()
        _, girl = bibles.find_element(bible, self.girl)
        _, roof = bibles.find_element(bible, self.roof)
        winter = bibles.find_variant(girl, "winter")
        self.assertEqual([p.name for p in bibles.reference_paths(bible, girl, winter)], [girl["image"]])
        hall_image = bibles.find_element(bible, self.hall)[1]["image"]
        self.assertEqual([p.name for p in bibles.reference_paths(bible, roof, roof["variants"][0], [self.hall])],
                         [hall_image])
        prompt = casting.bible_image_prompt("character", girl, winter, ref_label="Picture", from_default=True)
        self.assertTrue(prompt.startswith("The same person as Picture 1"))
        self.assertIn("Wearing: long camel coat", prompt)

    def test_an_episode_uses_a_look_and_the_props_reach_the_planner(self):
        bibles.add_image("spring_rain", self.girl, "default", self.upload)
        episode = os.path.join(self.tmp.name, "ep3")
        cast = bibles.project_cast(self._bible(), episode, variants={"Eun-seo": "winter"})
        girl = cast["characters"][0]
        self.assertEqual((girl["wardrobe"], girl["bible_variant"], girl["tag"]), ("long camel coat", "winter", "@eun_seo"))
        # The winter look has no image yet: the approved default sheet stands in.
        self.assertTrue(os.path.isfile(os.path.join(episode, girl["image"])))
        self.assertIn("- Umbrella: clear plastic umbrella (90 cm)", bibles.world_block(self._bible()))
        self.assertNotIn("Umbrella", bibles.world_block(self._bible(), props=["nothing"]))


class BibleRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = pipeline._wgp
        self.addCleanup(lambda: setattr(pipeline, "_wgp", saved))
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)

    def test_a_render_lands_in_the_bible(self):
        bible = bibles.save_bible({"name": "Spring Rain", "props": [{"name": "Umbrella", "description": "clear"}]})
        prop = bible["props"][0]["id"]
        requests = []

        def submit(gen_params, **kwargs):
            requests.append(gen_params)
            name = "render.png"
            with open(os.path.join(kwargs["out_dir"], name), "wb") as handle:
                handle.write(b"png")
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline.render_bible_image("spring_rain", prop, "default", image_model="cloud_fal_seedream_4")
            for _ in range(100):
                image = bibles.get_bible("spring_rain")["props"][0]["variants"][0]["images"][0]
                if image["status"] != "rendering":
                    break
                time.sleep(0.02)
        self.assertEqual(image["status"], "approved")
        self.assertTrue(requests[0]["prompt"].startswith("Prop reference photograph"))
        self.assertEqual(requests[0]["image_refs"], [])


class ShotSubjectRefTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = {}
        for name in ("girl.jpg", "boy.jpg", "teacher.jpg", "umbrella.jpg"):
            path = os.path.join(self.tmp.name, name)
            with open(path, "wb") as handle:
                handle.write(b"jpeg")
            self.paths[name] = path
        self.params = {
            "character_ref_labels": ["Eun-seo", "Ji-ho", "Ms. Park"],
            "character_ref_paths": [self.paths["girl.jpg"], self.paths["boy.jpg"], self.paths["teacher.jpg"]],
            "_director_cast": {
                "characters": [{"name": "Eun-seo", "descriptor": "the teen girl with see-through bangs", "tag": "@eun_seo"},
                               {"name": "Ji-ho", "descriptor": "the tall boy in a loose tie"},
                               {"name": "Ms. Park", "descriptor": "the stern homeroom teacher"}],
                "props": [{"name": "Umbrella", "descriptor": "clear plastic umbrella", "image": "umbrella.jpg"}],
            },
        }
        self.characters, self.props = pipeline.cast_ref_needles(self.params, self.tmp.name)
        self.all = [self.paths["girl.jpg"], self.paths["boy.jpg"], self.paths["teacher.jpg"]]

    def test_a_shot_gets_the_sheets_of_the_people_it_names(self):
        plan = {"_director_subjects_on_screen": [{"visual_description": "the tall boy in a loose tie, leaning"}]}
        refs = pipeline.shot_subject_refs(plan, "Close-up: the teen girl with see-through bangs looks up", self.characters,
                                          self.props, self.all)
        self.assertEqual(refs, [self.paths["girl.jpg"], self.paths["boy.jpg"]])

    def test_named_props_follow_and_an_unnamed_shot_keeps_every_sheet(self):
        refs = pipeline.shot_subject_refs({}, "Rain on the window; a clear plastic umbrella leans by the door",
                                          self.characters, self.props, self.all)
        self.assertEqual(refs, [*self.all, self.paths["umbrella.jpg"]])


class BiblePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = (pipeline._pipelines, pipeline._wgp)
        self.addCleanup(lambda: setattr(pipeline, "_pipelines", saved[0]) or setattr(pipeline, "_wgp", saved[1]))
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)
        self.out_dir = os.path.join(self.tmp.name, "out")
        pipeline._pipelines["p1"] = {"id": "p1", "status": "running", "progress": {}, "created_at": time.time(),
                                     "params": {}, "out_dir": self.out_dir}

    def test_a_project_starts_from_its_bible(self):
        bibles.save_bible({"name": "Spring Rain", "style": {"medium": "live action"},
                           "characters": [GIRL], "locations": [HALL],
                           "world": {"tone": "tender, rainy", "look": "Soft pastel highlights, luminous skin."}})
        params = {"bible_id": "spring_rain", "bible_locations": ["Hallway"]}

        block = pipeline._apply_bible("p1", params)

        self.assertEqual([c["name"] for c in params["_director_cast"]["characters"]], ["Eun-seo"])
        self.assertEqual(params["_director_style_contract"], {"medium": "live action"})
        self.assertEqual(params["director_casting"], "on")
        self.assertEqual(params["_director_bible_revision"], 1)
        self.assertIn("Tone: tender, rainy", block)
        self.assertEqual(params["_director_bible_look"], "Soft pastel highlights, luminous skin.")
        self.assertEqual(pipeline._pipelines["p1"]["bible"]["id"], "spring_rain")

    def test_a_revised_project_keeps_its_own_cast(self):
        bibles.save_bible({"name": "Spring Rain", "characters": [GIRL]})
        own = {"characters": [], "locations": []}
        params = {"bible_id": "spring_rain", "_director_cast": own}
        pipeline._apply_bible("p1", params)
        self.assertIs(params["_director_cast"], own)

    def test_an_unknown_bible_stops_the_project(self):
        with self.assertRaises(ValueError):
            pipeline._apply_bible("p1", {"bible_id": "missing"})


if __name__ == "__main__":
    unittest.main()


class BibleAssistantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        upload = os.path.join(self.tmp.name, "sheet.png")
        with open(upload, "wb") as handle:
            handle.write(b"png")
        bible = bibles.save_bible({"name": "Spring Rain", "characters": [GIRL]})
        self.girl = bible["characters"][0]["id"]
        self.bible = bibles.add_image("spring_rain", self.girl, "default", upload)

    def _assist(self, *answers):
        from services.director import bible_assistant
        replies = iter(answers)
        prompts = []

        def generate(**call):
            prompts.append(call["prompt"])
            return next(replies)

        result = bible_assistant.assist(self.bible, [{"role": "user", "content": "Ajoute un lieu : le toit"}],
                                        generate=generate, folder=bibles._folder("spring_rain"))
        return result, prompts

    def test_a_proposal_keeps_ids_and_images_and_adds_new_elements(self):
        proposal = {"reply": "J'ai ajouté le toit.", "changes": ["Added location Rooftop"], "bible": {
            "name": "Spring Rain", "world": {"look": "Soft pastel highlights."},
            "characters": [{"id": self.girl, "name": "Eun-seo", "description": GIRL["description"],
                            "variants": [{"id": "default", "label": "Default"}]}],
            "locations": [{"name": "Rooftop", "description": "Flat concrete roof, green fence 120 cm high."}],
        }}
        result, prompts = self._assist(json.dumps(proposal))
        bible = result["bible"]
        self.assertEqual(result["changes"], ["Added location Rooftop"])
        self.assertEqual(bible["characters"][0]["id"], self.girl)
        self.assertEqual(bible["characters"][0]["image"], self.bible["characters"][0]["image"])
        self.assertEqual(bible["locations"][0]["name"], "Rooftop")
        self.assertEqual(bible["world"]["look"], "Soft pastel highlights.")
        # The assistant reads text and ids, never image files.
        self.assertIn('"has_image": true', prompts[0])
        self.assertNotIn(self.bible["characters"][0]["image"], prompts[0])
        # Nothing is saved until the user applies and saves.
        self.assertEqual(bibles.get_bible("spring_rain")["locations"], [])

    def test_bad_json_is_retried_once(self):
        result, prompts = self._assist("not json", json.dumps({"reply": "ok", "changes": [], "bible": {}}))
        self.assertEqual((result["reply"], len(prompts)), ("ok", 2))
        self.assertEqual(result["bible"]["characters"][0]["id"], self.girl)

    def test_the_last_message_must_be_the_users(self):
        from services.director import bible_assistant
        with self.assertRaises(bible_assistant.BibleAssistantError):
            bible_assistant.assist(self.bible, [{"role": "assistant", "content": "hi"}],
                                   generate=lambda **_: "{}", folder=bibles._folder("spring_rain"))
