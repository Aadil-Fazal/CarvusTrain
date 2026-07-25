"""Unit tests for CarvusTrain parser module."""

import os
import tempfile
import unittest

from carvustrain.parser import CarvusTrainParser, DataParser


class TestParser(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_custom_carvustrain_section_parser(self):
        content = """[CarvusTrain]

[Model]
Name=CarvusTest

[Training]
Method=normal
Duration=5

[Knowledge]
Carvus is a test model.
CarvusTrain is an AI framework.
"""
        parsed = CarvusTrainParser.parse_text(content)
        self.assertEqual(parsed.model_config.get("Name"), "CarvusTest")
        self.assertEqual(parsed.training_config.get("Method"), "normal")
        self.assertEqual(len(parsed.knowledge), 1)
        self.assertIn("Carvus is a test model.", parsed.knowledge[0])

    def test_parse_txt_file(self):
        txt_path = os.path.join(self.temp_dir.name, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Line 1 text.\n\nLine 2 paragraph.")

        records = DataParser.parse(txt_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "Line 1 text.")

    def test_parse_json_file(self):
        json_path = os.path.join(self.temp_dir.name, "sample.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write('[{"text": "Sample A"}, {"text": "Sample B"}]')

        records = DataParser.parse(json_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "Sample A")


if __name__ == "__main__":
    unittest.main()
