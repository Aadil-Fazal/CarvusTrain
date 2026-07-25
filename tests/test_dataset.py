"""Unit tests for dataset and data loader modules."""

import unittest

from carvustrain.dataset import DataLoader, Dataset, TextDataset


class TestDataset(unittest.TestCase):

    def test_dataset_operations(self):
        samples = [{"text": f"Sample {i}"} for i in range(100)]
        ds = Dataset(samples)
        self.assertEqual(len(ds), 100)
        self.assertEqual(ds[0]["text"], "Sample 0")

        train_ds, val_ds = ds.split(val_ratio=0.2, seed=42)
        self.assertEqual(len(train_ds), 80)
        self.assertEqual(len(val_ds), 20)

    def test_data_loader(self):
        samples = [{"text": f"Sample {i}"} for i in range(25)]
        ds = Dataset(samples)
        loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=False)
        batches = list(loader)
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 10)
        self.assertEqual(len(batches[2]), 5)


if __name__ == "__main__":
    unittest.main()
