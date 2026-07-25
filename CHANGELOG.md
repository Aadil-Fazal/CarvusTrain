# Changelog

All notable changes to the `carvustrain` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-25

### Fixed
- **Import resolution**: Changed all internal `from carvustrain.xxx` absolute imports to relative `from .xxx` imports across 13 package files. This fixes `ModuleNotFoundError` on case-sensitive filesystems and makes the package robust regardless of the installed module name.
- **CLI entry point**: Corrected the console script reference in `pyproject.toml` to match the actual package module name.

## [1.0.0] - 2026-07-25

### Added
- **Core Framework**: Initial release of CarvusTrain AI framework built from scratch in Python.
- **High-Level Model API**: `Model`, `ChatModel`, `TextModel`, `LanguageModel`, and `CustomModel` classes.
- **Expressive Methods**: `train()`, `ask()`, `predict()`, `generate()`, `chat()`, `learn()`, `finetune()`, `evaluate()`, `export()`, `load()`, `save()`, `summary()`, `statistics()`, and `memory()`.
- **Dataset System**: Multi-format data parser supporting TXT, CSV, JSON, JSONL, XML, YAML, Markdown, folder scanning, and auto-encoding detection (UTF-8, UTF-16, ASCII).
- **Custom Section Parser**: Parser for native `[CarvusTrain]` config and knowledge files (`[CarvusTrain]`, `[Model]`, `[Training]`, `[Knowledge]`).
- **Tokenization Suite**: `WordTokenizer`, `CharTokenizer`, `SentenceTokenizer`, `BPETokenizer`, `WordPieceTokenizer`, `SentencePieceTokenizer`, and `CustomTokenizer` with vocabulary management.
- **Neural Layer Architecture**: Dense/Linear, Embedding, LayerNorm, RMSNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock, FeedForward, Dropout, Residual, and Sequential layers.
- **Optimizers & Schedulers**: Adam, AdamW, SGD, RMSprop, AdaGrad with weight decay, gradient clipping, step/cosine/warmup/plateau schedulers.
- **Trainer Engine**: Support for normal, incremental, continuous, forever, streaming, fine-tuning, self-supervised, supervised, unsupervised, and reinforcement learning flows.
- **Exporting Suite**: Support for saving and exporting models to `.ct`, `.bin`, `.onnx`, `.json`, and `.gguf` formats.
- **Command-Line Interface**: Full CLI `carvustrain` with commands: `init`, `train`, `predict`, `chat`, `export`, `evaluate`, `benchmark`, `convert`, `version`, `doctor`, `config`, `install`, `update`.
- **Documentation & Examples**: Extensive Markdown documentation under `docs/` and runnable Python scripts under `examples/`.
- **Test Suite**: Unit and integration tests for modules under `tests/`.
