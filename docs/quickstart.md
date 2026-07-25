# CarvusTrain Quick Start Guide

Get started with **CarvusTrain** in less than 5 minutes.

## 1. Minimal Working Example

```python
import carvustrain

# Initialize model
model = carvustrain.Model(name="Carvus")

# Train model on a text dataset or file
model.train(
    data="train.txt",
    method="normal",
    duration="forever"
)

# Save trained model weights and knowledge base
model.save("carvus.ct")

# Query model via Q&A API
model.ask("Who are you?")
print(model.answer)
```

## 2. Training with Custom Data Formats

CarvusTrain supports automatic ingestion of TXT, CSV, JSON, JSONL, XML, YAML, Markdown, and custom `.ct` sectioned files.

```python
import carvustrain

# Train on JSONL dataset
model = carvustrain.Model()
model.train(data="dataset.jsonl", epochs=10, batch_size=16, learning_rate=1e-3)

# Fine-tune model on new domain data
model.finetune(data=["New domain text fact 1.", "New domain text fact 2."])

# Export model to GGUF format
model.export("carvus.gguf", format="gguf")
```

## 3. Interactive Multi-Turn Chat

```python
from carvustrain import ChatModel

chat_bot = ChatModel()
response = chat_bot.chat("Hello Carvus!")
print(response)
```
