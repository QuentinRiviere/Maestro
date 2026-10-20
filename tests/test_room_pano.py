"""Panorama crops, prompts and the dataset LoRA trainings."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from PIL import Image  # noqa: E402

from services.director import bibles  # noqa: E402
from services.director import lora_training as lt  # noqa: E402
from services.director import room_pano as rp  # noqa: E402
from services.director import style_lora as sl  # noqa: E402


def _banded_panorama(width: int = 720, height: int = 360) -> Image.Image:
    """An equirectangular test image: red straight ahead, green to the right, blue behind, yellow to the left."""
    image = Image.new("RGB", (width, height), (0, 0, 0))
    pixels = image.load()
    colours = {0: (255, 0, 0), 90: (0, 255, 0), 180: (0, 0, 255), -90: (255, 255, 0)}
    for x in range(width):
        lon = (x / width - 0.5) * 360.0
        for centre, colour in colours.items():
            if abs(((lon - centre + 180) % 360) - 180) < 20:
                for y in range(height):
                    pixels[x, y] = colour
    return image


class CropTests(unittest.TestCase):
    def test_each_yaw_looks_at_its_band(self):
        panorama = _banded_panorama()
        for yaw, colour in ((0, (255, 0, 0)), (90, (0, 255, 0)), (180, (0, 0, 255)), (-90, (255, 255, 0))):
            view = rp.crop_view(panorama, yaw, size=(160, 90))
            centre = view.getpixel((80, 45))
            self.assertEqual(centre, colour, f"yaw {yaw}")
        # Straight ahead with a 70 degree field of view, the green band (centred at 90) is out of frame.
        front = rp.crop_view(panorama, 0, size=(160, 90))
        self.assertNotIn((0, 255, 0), [front.getpixel((x, 45)) for x in range(0, 160, 8)])
        self.assertEqual(rp.VIEW_YAWS["reverse"], 180.0)

    def test_panorama_prompt_from_the_sheet(self):
        spec = {"walls": {"front": "a window", "left": "tables", "right": "the bar", "back": "shelves"},
                "materials": "brick", "light": "warm", "palette": ["oak"], "summary": "a cafe"}
        prompt = rp.panorama_prompt({"name": "Café"}, spec, look="Film look.")
        self.assertTrue(prompt.startswith("maestro_pano, equirectangular 360 degree panorama of Café"))
        self.assertIn("Centre of the panorama (straight ahead): a window", prompt)
        self.assertIn("At both edges of the panorama (behind the camera): shelves", prompt)
        self.assertTrue(prompt.endswith("Film look."))
        bare = rp.panorama_prompt({"name": "Café", "description": "A room"}, None)
        self.assertIn("Place: A room", bare)


class DatasetTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.loras = os.path.join(self.tmp, "loras")
        os.makedirs(self.loras)
        lt._settings_dir = lambda: os.path.join(self.tmp, "settings")
        self.addCleanup(setattr, lt, "_settings_dir", None)
        registry = patch.object(sl, "_registry_dir", lambda: self.loras)
        registry.start()
        self.addCleanup(registry.stop)

    def _zip(self, count: int = 15) -> str:
        folder = Path(self.tmp, "set")
        (folder / "images").mkdir(parents=True)
        items = []
        for i in range(count):
            Image.new("RGB", (64, 32), (i * 10, 0, 0)).save(folder / "images" / f"{i}.jpg")
            items.append({"image": f"images/{i}.jpg", "caption": f"maestro_pano, test {i}"})
        (folder / "manifest.json").write_text(json.dumps({"trigger": "maestro_pano", "items": items}))
        archive = str(lt.datasets_dir() / "set.zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(folder / "manifest.json", "manifest.json")
            for item in items:
                zipped.write(folder / item["image"], item["image"])
        return archive

    def test_inspect_create_train_and_install(self):
        archive = self._zip()
        facts = lt.inspect_dataset(archive)
        self.assertEqual(facts["images"], 15)
        self.assertEqual(facts["trigger"], "maestro_pano")
        record = lt.create_training("Maestro pano", archive, {"steps": 80, "rank": 8})
        self.assertEqual(record["trigger"], "maestro_pano")
        self.assertEqual(record["config"]["steps"], 80)
        submitted = []

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            name = params["lora_training"]["name"] + ".safetensors"
            from safetensors.torch import save_file
            import torch
            save_file({"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 4)},
                      os.path.join(out_dir, name))
            Path(out_dir, params["lora_training"]["name"] + ".json").write_text(json.dumps({"final_loss": 0.2}))
            return [name]

        done = lt.start_training(record["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(done["file"], "maestro_pano.safetensors")
        self.assertTrue(os.path.isfile(os.path.join(self.loras, "maestro_pano.safetensors")))
        sidecar = json.loads(Path(self.loras, "maestro_pano.civitai.json").read_text())
        self.assertEqual(sidecar["trainedWords"], ["maestro_pano"])
        self.assertEqual(submitted[0]["lora_training"]["trigger"], "maestro_pano")
        self.assertEqual(lt.get_training(record["id"])["training"]["final_loss"], 0.2)
        # The registry file name is what the panorama render loads.
        with patch.object(rp, "_lora_name", lambda: "maestro_pano.safetensors"):
            self.assertTrue(rp.lora_available())
        lt.delete_training(record["id"])
        self.assertEqual(lt.list_trainings(), [])

    def test_bad_dataset_is_refused(self):
        bad = str(lt.datasets_dir() / "bad.zip")
        with zipfile.ZipFile(bad, "w") as zipped:
            zipped.writestr("readme.txt", "no manifest")
        with self.assertRaises(bibles.BibleError):
            lt.inspect_dataset(bad)
        # A record cut by a restart reads as failed.
        archive = self._zip()
        record = lt.create_training("x", archive)
        lt._set(record["id"], status="training")
        self.assertEqual(lt.get_training(record["id"])["status"], "failed")


if __name__ == "__main__":
    unittest.main()
