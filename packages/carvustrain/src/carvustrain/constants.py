"""Constants and configuration defaults for CarvusTrain."""

import os
from typing import Final, List, Tuple

# Special Tokens
PAD_TOKEN: Final[str] = "<pad>"
UNK_TOKEN: Final[str] = "<unk>"
BOS_TOKEN: Final[str] = "<bos>"
EOS_TOKEN: Final[str] = "<eos>"
MASK_TOKEN: Final[str] = "<mask>"

SPECIAL_TOKENS: Final[List[str]] = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
]

# File Extensions & Formats
DEFAULT_MODEL_EXTENSION: Final[str] = ".ct"
SUPPORTED_EXPORT_FORMATS: Final[Tuple[str, ...]] = (
    "ct",
    "bin",
    "onnx",
    "json",
    "gguf",
)

SUPPORTED_DATASET_FORMATS: Final[Tuple[str, ...]] = (
    "txt",
    "csv",
    "json",
    "jsonl",
    "xml",
    "yaml",
    "yml",
    "md",
    "markdown",
    "ct",
    "cl",
)

SUPPORTED_ENCODINGS: Final[Tuple[str, ...]] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "ascii",
    "latin-1",
    "cp1252",
)

# Custom CarvusTrain File Sections
SECTION_HEADER_CARVUSTRAIN: Final[str] = "[CarvusTrain]"
SECTION_HEADER_MODEL: Final[str] = "[Model]"
SECTION_HEADER_TRAINING: Final[str] = "[Training]"
SECTION_HEADER_KNOWLEDGE: Final[str] = "[Knowledge]"
SECTION_HEADER_EXAMPLES: Final[str] = "[Examples]"

CARVUSTRAIN_SECTIONS: Final[Tuple[str, ...]] = (
    SECTION_HEADER_CARVUSTRAIN,
    SECTION_HEADER_MODEL,
    SECTION_HEADER_TRAINING,
    SECTION_HEADER_KNOWLEDGE,
    SECTION_HEADER_EXAMPLES,
)

# Training Methods
TRAINING_METHOD_NORMAL: Final[str] = "normal"
TRAINING_METHOD_INCREMENTAL: Final[str] = "incremental"
TRAINING_METHOD_CONTINUOUS: Final[str] = "continuous"
TRAINING_METHOD_FOREVER: Final[str] = "forever"
TRAINING_METHOD_STREAMING: Final[str] = "streaming"
TRAINING_METHOD_FINETUNE: Final[str] = "finetune"
TRAINING_METHOD_SUPERVISED: Final[str] = "supervised"
TRAINING_METHOD_UNSUPERVISED: Final[str] = "unsupervised"
TRAINING_METHOD_SELF_SUPERVISED: Final[str] = "self_supervised"
TRAINING_METHOD_REINFORCEMENT: Final[str] = "reinforcement"

SUPPORTED_TRAINING_METHODS: Final[Tuple[str, ...]] = (
    TRAINING_METHOD_NORMAL,
    TRAINING_METHOD_INCREMENTAL,
    TRAINING_METHOD_CONTINUOUS,
    TRAINING_METHOD_FOREVER,
    TRAINING_METHOD_STREAMING,
    TRAINING_METHOD_FINETUNE,
    TRAINING_METHOD_SUPERVISED,
    TRAINING_METHOD_UNSUPERVISED,
    TRAINING_METHOD_SELF_SUPERVISED,
    TRAINING_METHOD_REINFORCEMENT,
)

# Devices
DEVICE_CPU: Final[str] = "cpu"
DEVICE_CUDA: Final[str] = "cuda"
DEVICE_GPU: Final[str] = "gpu"
DEVICE_MPS: Final[str] = "mps"

# Default Hyperparameters
DEFAULT_VOCAB_SIZE: Final[int] = 10000
DEFAULT_EMBEDDING_DIM: Final[int] = 128
DEFAULT_HIDDEN_DIM: Final[int] = 256
DEFAULT_NUM_LAYERS: Final[int] = 4
DEFAULT_NUM_HEADS: Final[int] = 4
DEFAULT_MAX_SEQ_LEN: Final[int] = 512
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_EPOCHS: Final[int] = 10
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01
DEFAULT_GRADIENT_CLIPPING: Final[float] = 1.0

# Directory Paths
DEFAULT_CACHE_DIR: Final[str] = os.path.join(os.path.expanduser("~"), ".cache", "carvustrain")
DEFAULT_CHECKPOINT_DIR: Final[str] = "checkpoints"
DEFAULT_OUTPUT_DIR: Final[str] = "outputs"
