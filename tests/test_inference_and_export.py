"""Unit tests for inference engines and multi-format exporters."""

import os
import tempfile
import unittest

import carvustrain


class TestInferenceAndExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_formats(self):
        model = carvustrain.Model(name="ExportTest")
        model.learn("Sample knowledge statement.")

        formats = ["ct", "json", "bin", "onnx", "gguf"]
        for fmt in formats:
            out_file = os.path.join(self.temp_dir.name, f"model.{fmt}")
            model.export(out_file, format=fmt)
            self.assertTrue(os.path.exists(out_file), f"Export failed for format {fmt}")


if __name__ == "__main__":
    unittest.main()
