"""Dataset management, batching, shuffling, streaming, and loading utilities."""

import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .exceptions import DatasetNotFoundError
from .parser import DataParser


class Dataset:
    """Base dataset container holding items with indexing, iteration, and splitting."""

    def __init__(self, data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.data: List[Dict[str, Any]] = data or []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.data)

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        """Shuffle dataset samples in-place or with a fixed seed."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
        return self

    def split(self, val_ratio: float = 0.1, seed: int = 42) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into train and validation subsets.

        Args:
            val_ratio: Proportion of dataset allocated to validation (0.0 to 1.0).
            seed: Random seed for deterministic splitting.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        data_copy = list(self.data)
        random.seed(seed)
        random.shuffle(data_copy)

        val_size = int(len(data_copy) * val_ratio)
        val_data = data_copy[:val_size]
        train_data = data_copy[val_size:]

        return Dataset(train_data), Dataset(val_data)

    @classmethod
    def from_file(cls, file_path: str, encoding: Optional[str] = None) -> "Dataset":
        """Construct dataset by parsing file path."""
        records = DataParser.parse(file_path, encoding=encoding)
        return cls(records)

    @classmethod
    def from_directory(cls, dir_path: str, recursive: bool = True) -> "Dataset":
        """Construct dataset from scanning a directory."""
        records = DataParser.parse_directory(dir_path, recursive=recursive)
        return cls(records)

    @classmethod
    def from_texts(cls, texts: List[str]) -> "Dataset":
        """Construct dataset from a list of raw text strings."""
        return cls([{"text": t} for t in texts])


class TextDataset(Dataset):
    """Specialized dataset for text generation, classification, and tokenized NLP tasks."""

    def __init__(self, data: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(data)

    def get_all_texts(self) -> List[str]:
        """Extract all text samples as a list of strings."""
        texts = []
        for sample in self.data:
            if isinstance(sample, dict) and "text" in sample:
                texts.append(str(sample["text"]))
            elif isinstance(sample, str):
                texts.append(sample)
        return texts


class TabularDataset(Dataset):
    """Dataset for tabular/structured row data."""

    def __init__(self, data: Optional[List[Dict[str, Any]]] = None, target_column: Optional[str] = None) -> None:
        super().__init__(data)
        self.target_column = target_column

    def get_features_and_targets(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """Separate features and target labels."""
        if not self.target_column:
            return self.data, []

        features, targets = [], []
        for row in self.data:
            feat = {k: v for k, v in row.items() if k != self.target_column}
            targ = row.get(self.target_column)
            features.append(feat)
            targets.append(targ)

        return features, targets


class DataLoader:
    """Batch generator and data loader with shuffling, drop_last, and batching capabilities."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        num_samples = len(self.dataset)
        if self.drop_last:
            return num_samples // self.batch_size
        return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch
