"""Unit tests for Model and Trainer classes."""

import os
import tempfile
import unittest

import carvustrain


class TestModelAndTrainer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_training_and_ask(self):
        model = carvustrain.Model(name="TestCarvus")
        history = model.train(data=["Carvus is an AI created with CarvusTrain."], epochs=2, batch_size=2)
        self.assertIn("loss", history)

        answer = model.ask("Who is Carvus?")
        self.assertIsNotNone(answer)
        self.assertEqual(model.answer, answer)

    def test_save_and_load(self):
        model = carvustrain.Model(name="SaveLoadCarvus", auto_load=False)
        model.learn("Knowledge fact sample.")

        save_path = os.path.join(self.temp_dir.name, "model.ct")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_model = carvustrain.Model(auto_load=False)
        loaded_model.load(save_path)
        self.assertEqual(len(loaded_model.knowledge_base), 1)


if __name__ == "__main__":
    unittest.main()
