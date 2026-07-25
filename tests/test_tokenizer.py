"""Unit tests for CarvusTrain tokenizer module."""

import unittest

from carvustrain.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SentenceTokenizer,
    Tokenizer,
    Vocabulary,
    WordTokenizer,
)


class TestTokenizer(unittest.TestCase):

    def test_vocabulary(self):
        vocab = Vocabulary()
        id1 = vocab.add_token("hello")
        id2 = vocab.add_token("world")
        self.assertNotEqual(id1, id2)
        self.assertEqual(vocab.get_token(id1), "hello")

    def test_word_tokenizer(self):
        tok = Tokenizer.create("word")
        tokens = tok.tokenize("Hello, CarvusTrain world!")
        self.assertEqual(tokens, ["Hello", ",", "CarvusTrain", "world", "!"])

    def test_char_tokenizer(self):
        tok = Tokenizer.create("char")
        tokens = tok.tokenize("abc")
        self.assertEqual(tokens, ["a", "b", "c"])

    def test_sentence_tokenizer(self):
        tok = Tokenizer.create("sentence")
        tokens = tok.tokenize("First sentence. Second sentence!")
        self.assertEqual(len(tokens), 2)

    def test_bpe_tokenizer(self):
        tok = BPETokenizer()
        tok.train_on_texts(["low lower lowest"], num_merges=5)
        encoded = tok.encode("low", add_special_tokens=False)
        self.assertTrue(len(encoded) > 0)


if __name__ == "__main__":
    unittest.main()
