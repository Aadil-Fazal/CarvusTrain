# CarvusTrain API Reference

## Model Classes

### `carvustrain.Model(name="Carvus", config=None, **kwargs)`
Primary high-level model interface.

#### Methods:
- `train(data=None, method="normal", duration=10, epochs=10, batch_size=32, learning_rate=1e-3, optimizer="adamw", device="auto", ...)`: Train model on dataset.
- `ask(question: str) -> str`: Query model with a question string. Sets `model.answer`.
- `predict(input_text: str) -> str`: Generate prediction for input text.
- `generate(prompt: str, max_new_tokens=100) -> str`: Autoregressively continue text.
- `chat(message: str) -> str`: Interactive multi-turn chat response.
- `learn(text: Union[str, List[str]])`: Ingest text facts directly into memory.
- `finetune(data, epochs=5, learning_rate=1e-4)`: Fine-tune model.
- `evaluate(dataset=None) -> Dict[str, float]`: Evaluate accuracy and loss.
- `benchmark(prompt="Who are you?", num_runs=20)`: Measure latency and tokens/sec.
- `save(filepath=None)`: Save model to `.ct` file format.
- `export(output_path, format=None)`: Export to `.ct`, `.bin`, `.onnx`, `.json`, `.gguf`.
- `load(filepath)`: Restore model from disk.
- `summary()`: Print model architecture summary.
- `statistics()`: Get system and model statistics.
- `memory()`: Inspect memory footprint.

### Specialized Model Subclasses
- `ChatModel`: Specialized for dialogue and chat sessions.
- `TextModel`: Specialized for document completion and generation.
- `LanguageModel`: Specialized for token prediction.
- `CustomModel`: User-customizable base model.

## Data & Tokenization
- `Dataset`: Base dataset container.
- `TextDataset`: Specialized text dataset.
- `DataLoader`: Batching, shuffling, and iteration loader.
- `Tokenizer`: Tokenization factory creating `WordTokenizer`, `CharTokenizer`, `BPETokenizer`, `WordPieceTokenizer`, `SentencePieceTokenizer`.
- `DataParser`: Multi-format file parser (TXT, CSV, JSON, XML, YAML, Markdown, Folders).
- `CarvusTrainParser`: Parser for native `[CarvusTrain]` sectioned files.
