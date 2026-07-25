"""Unit tests for neural layers, activations, and losses."""

import unittest

from carvustrain.activation import GELU, ReLU, Softmax
from carvustrain.layers import Dense, Embedding, LayerNorm, Sequential
from carvustrain.losses import CrossEntropyLoss, MSELoss, Perplexity


class TestLayersAndLosses(unittest.TestCase):

    def test_activations(self):
        relu = ReLU()
        self.assertEqual(relu(-5.0), 0.0)
        self.assertEqual(relu(3.0), 3.0)

        gelu = GELU()
        self.assertAlmostEqual(gelu(0.0), 0.0, places=4)

        softmax = Softmax()
        sm = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(sm), 1.0, places=4)

    def test_layers(self):
        dense = Dense(10, 5)
        out = dense([1.0] * 10)
        self.assertEqual(len(out), 5)

        emb = Embedding(100, 16)
        vec = emb(5)
        self.assertEqual(len(vec), 16)

    def test_losses(self):
        mse = MSELoss()
        loss_val = mse(1.0, 1.0)
        self.assertIsNotNone(loss_val)

        ppl = Perplexity()
        self.assertTrue(ppl(0.5, 0.5) > 0)


if __name__ == "__main__":
    unittest.main()
