# CarvusTrain Framework Architecture

CarvusTrain uses a multi-tiered, modular, object-oriented architecture designed for scalability, maintainability, and extensibility.

## System Layers

1. **Base Layer**
   - `constants.py`: Default hyperparameters, section headers, special tokens.
   - `exceptions.py`: Custom exception hierarchy (`CarvusTrainError`, `TrainingError`, etc.).
   - `logger.py`: Colorized logging and telemetry.
   - `utils.py`: Device detection, formatting, hardware diagnostics.
   - `configuration.py`: Dataclass config manager (`CarvusConfig`).

2. **Data & Tokenization Layer**
   - `parser.py`: Multi-format file reader & native `.ct` section parser.
   - `preprocessing.py`: Text cleaning, normalization, padding, and sequence chunking.
   - `tokenizer.py`: Word, Char, BPE, WordPiece, SentencePiece tokenizers.
   - `dataset.py`: `Dataset`, `TextDataset`, `DataLoader` batching engine.

3. **Neural Computation Layer**
   - `activation.py`: ReLU, GELU, SiLU, Softmax, Sigmoid, Tanh, LeakyReLU.
   - `layers.py`: Dense, Embedding, LayerNorm, RMSNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock, Sequential.
   - `losses.py`: CrossEntropyLoss, MSELoss, BCELoss, SmoothL1Loss, KLDivLoss, Perplexity, FocalLoss.
   - `metrics.py`: Accuracy, Precision, Recall, F1, BLEU, ROUGE.
   - `optimizer.py`: Adam, AdamW, SGD, RMSprop, AdaGrad with gradient clipping and weight decay.
   - `scheduler.py`: StepLR, CosineAnnealingLR, LinearWarmupLR, ReduceLROnPlateau.

4. **Memory & Context Layer**
   - `memory.py`: `KnowledgeBase`, `MemoryCache` (KV cache), `ContextWindow`.
   - `postprocessing.py`: Temperature, Top-K, Top-P, repetition penalty.

5. **Engine & Interface Layer**
   - `trainer.py`: Epoch training engine.
   - `inference.py`: Question answering & text generation pipelines.
   - `evaluation.py`: Evaluator & Benchmarker.
   - `exporter.py`: Model exporter (.ct, .bin, .onnx, .json, .gguf).
   - `model.py`: High-level `Model`, `ChatModel`, `TextModel`, `LanguageModel`, `CustomModel`.
   - `cli.py`: Command Line Interface (`carvustrain`).
