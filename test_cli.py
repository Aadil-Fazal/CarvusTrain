"""Unit tests for CarvusTrain CLI commands."""

import os
import tempfile
import unittest

from carvustrain.cli import main


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_version(self):
        # Should execute version command without error
        try:
            main(["version"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_cli_doctor(self):
        try:
            main(["doctor"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_cli_init(self):
        proj_dir = os.path.join(self.temp_dir.name, "my_project")
        main(["init", proj_dir])
        self.assertTrue(os.path.exists(os.path.join(proj_dir, "config.toml")))
        self.assertTrue(os.path.exists(os.path.join(proj_dir, "train.ct")))


if __name__ == "__main__":
    unittest.main()
