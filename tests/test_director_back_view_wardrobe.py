"""A back-view sheet must not name front-only accessories, or the model paints them on the back."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.casting import back_view_wardrobe, view_prompt  # noqa: E402


class BackViewWardrobeTests(unittest.TestCase):
    def test_drops_a_pendant_sentence(self):
        wardrobe = ("Oversized knitted cream-colored wool sweater over a crisp white cotton collared shirt, "
                    "charcoal pleated wide-leg trousers, and white canvas lace-up sneakers. "
                    "Wears a slim silver chain with a plain round disc pendant.")
        result = back_view_wardrobe(wardrobe)
        self.assertNotIn("pendant", result)
        self.assertNotIn("chain", result)
        self.assertIn("white canvas lace-up sneakers", result)

    def test_drops_only_the_accessory_item_of_a_list(self):
        result = back_view_wardrobe("Plum wool cape dress, accented by an antique baroque pearl broach and black "
                                    "kidskin leather gloves. Black velvet pumps.")
        self.assertNotIn("broach", result)
        self.assertIn("black kidskin leather gloves", result)
        self.assertIn("Black velvet pumps.", result)

    def test_keeps_garments_with_front_details(self):
        wardrobe = "Forest green corduroy overshirt over a beige graphic t-shirt, relaxed olive cargo pants."
        self.assertEqual(back_view_wardrobe(wardrobe), wardrobe)

    def test_only_the_back_view_is_filtered(self):
        element = {"description": "A woman.", "wardrobe": "A cream sweater. Wears a silver pendant."}
        self.assertNotIn("pendant", view_prompt("back", element, {}).split("Wearing:")[1])
        self.assertIn("pendant", view_prompt("face_front", element, {}).split("Wearing:")[1])


if __name__ == "__main__":
    unittest.main()
