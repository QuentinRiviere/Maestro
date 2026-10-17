"""Name parts ("Jin-woo" for "Song Jin-woo") map to the character's descriptor."""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.prompt_polish import expand_name_aliases  # noqa: E402


class NameAliasTests(unittest.TestCase):
    def test_each_unambiguous_part_becomes_an_alias(self):
        aliases = expand_name_aliases({"Song Jin-woo": "the man in green", "Cha Hae-won": "the woman in cream"})
        self.assertEqual(aliases["Jin-woo"], "the man in green")
        self.assertEqual(aliases["Song"], "the man in green")
        self.assertEqual(aliases["Hae-won"], "the woman in cream")
        self.assertEqual(aliases["Song Jin-woo"], "the man in green")

    def test_shared_parts_titles_and_short_parts_are_skipped(self):
        aliases = expand_name_aliases({"Dr. Kim Min-ji": "the doctor", "Kim Tae-ho": "the student", "Al": "the dog"})
        self.assertNotIn("Kim", aliases)
        self.assertNotIn("Dr", aliases)
        self.assertNotIn("Dr.", aliases)
        self.assertEqual(aliases["Min-ji"], "the doctor")
        self.assertEqual(aliases["Tae-ho"], "the student")
        self.assertEqual(aliases["Al"], "the dog")

    def test_existing_keys_and_empty_descriptors_are_kept(self):
        aliases = expand_name_aliases({"Blaine Carter": "the tall man", "Carter": "the butler", "No Desc": ""})
        self.assertEqual(aliases["Carter"], "the butler")
        self.assertEqual(aliases["Blaine"], "the tall man")
        self.assertNotIn("Desc", aliases)
        self.assertEqual(aliases["No Desc"], "")


if __name__ == "__main__":
    unittest.main()
