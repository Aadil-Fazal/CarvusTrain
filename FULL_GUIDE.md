# CarvusTrain — The Complete Guide

> **Building AI That Actually Learns, Understands Grammar, Generates Real Code, and Deploys Anywhere**
>
> *"The PyTorch + Hugging Face + LangChain + Ollama of the Carvus ecosystem."*

**Version:** 1.0.1 · **Last updated:** July 2026

---

## Table of Contents

1. [Philosophy & Vision](#1-philosophy--vision)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Architecture Overview](#4-architecture-overview)
5. [Core Components](#5-core-components)
6. [Training Configuration (train.ct)](#6-training-configuration-trainct)
7. [Training & Validation](#7-training--validation)
8. [Code Generation](#8-code-generation)
9. [English Grammar & Natural Language Understanding](#9-english-grammar--natural-language-understanding)
10. [Semantic Search](#10-semantic-search)
11. [Deployment](#11-deployment)
12. [Agent System](#12-agent-system)
13. [Plugin System](#13-plugin-system)
14. [CarvusTrain Language (CTL)](#14-carvustrain-language-ctl)
15. [CLI Reference](#15-cli-reference)
16. [Performance & Optimization](#16-performance--optimization)
17. [Best Practices](#17-best-practices)
18. [Extending CarvusTrain](#18-extending-carvustrain)
19. [Troubleshooting](#19-troubleshooting)
20. [Full API Reference](#20-full-api-reference)
21. [Changelog](#21-changelog)

---

## 1. Philosophy & Vision

### What Makes CarvusTrain Different?

Most AI frameworks focus on one thing: training bigger models. CarvusTrain takes a different approach — it's a **complete AI development ecosystem** that focuses on:

1. **Knowledge-first learning** — instead of only training parameters, the model accumulates structured knowledge in a semantic knowledge base.
2. **Real code generation** — 50+ algorithm templates across 14+ programming languages, all syntactically valid and working (no placeholders).
3. **Grammar-aware understanding** — built-in English grammar knowledge for natural language comprehension.
4. **Validation-driven training** — every learning step is validated for accuracy, retention, and understanding.
5. **Zero to production** — from model creation to a REST API server in one command.

### The Ecosystem

```
CarvusTrain
├── Core Library     — Model, KnowledgeBase, Trainer, Inference
├── RAG Pipeline     — Document ingestion, embedding, retrieval
├── Model Server     — REST API deployment
├── Agent System     — Goal-oriented AI agents
├── Plugin System    — Extensible capabilities
├── CLI              — 15+ commands for the full lifecycle
└── CTL              — CarvusTrain Language (DSL)
```

### Key Design Principles

- **Knowledge-first**: the model learns by accumulating facts in a semantic knowledge base, not solely by adjusting billions of parameters.
- **Real code generation**: algorithm templates generate syntactically valid, working code with no placeholders.
- **Grammar-aware**: built-in English grammar knowledge helps the model understand and generate natural language.
- **Validation-driven**: learning accuracy, knowledge retention, and language understanding are tracked and validated continuously.
- **Zero external dependencies required**: core functionality works with pure Python — optional upgrades via `scikit-learn`, `sentence-transformers`, or `torch`.

---

## 2. Installation

### Requirements

- Python 3.10 or higher
- Supported OS: Windows, Linux, macOS

### Standard Installation

```bash
pip install carvustrain
```

### Installing from Source

```bash
git clone https://github.com/Aadil-Fazal/CarvusTrain.git
cd carvustrain
pip install -e .
```

### Optional Dependencies

Install extras to unlock more power as your knowledge base and hardware needs grow:

```bash
# Faster TF-IDF semantic search
pip install scikit-learn

# Deep semantic embeddings
pip install sentence-transformers

# Fast vector search
pip install faiss-cpu

# PyTorch hardware acceleration (CUDA / MPS) + YAML parsing
pip install torch pyyaml
```

None of these are required to get started — CarvusTrain's pure-Python fallback path always works, and each optional package simply upgrades speed or search depth.

---

## 3. Quick Start

### Minimal Working Example

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

# Query the model via the Q&A API
model.ask("Who are you?")
print(model.answer)
```

### Create Your First AI (Full Walkthrough)

```python
from carvustrain import Model

# Create with architecture selection
model = Model(
    name="MyAI",
    architecture="transformer",  # Also: neural_network, cnn, rnn, lstm
    parameters="base"
)

# Load train.ct knowledge (auto-detected in the working directory)
model = Model(auto_load=True)

# Or manually load grammar knowledge
model.knowledge_base.load_grammar_knowledge()
print(f"Loaded {len(model.knowledge_base)} facts")
```

### Train and Chat

```python
# Train with examples
model.learn("Python is a dynamically-typed programming language.")
model.learn("JavaScript runs in web browsers.")
model.learn("Rust provides memory safety without a garbage collector.")

# Or train a full dataset
model.auto_train("train.ct")  # Auto AI Trainer

# Chat with your AI
model.mode = "teacher"  # Switch to teacher mode
answer = model.chat("Explain what a variable is")
print(answer)
```

### Generate Code

```python
code = model.inference_engine.generate_code(
    "Write a Dijkstra shortest path algorithm in Python"
)
print(code)

code = model.inference_engine.generate_code(
    "Write a BFS traversal in JavaScript"
)
print(code)

code = model.inference_engine.generate_code(
    "Implement merge sort in Go with generics"
)
print(code)
```

### Training with Custom Data Formats

CarvusTrain automatically ingests TXT, CSV, JSON, JSONL, XML, YAML, Markdown, and custom `.ct` sectioned files.

```python
import carvustrain

# Train on a JSONL dataset
model = carvustrain.Model()
model.train(data="dataset.jsonl", epochs=10, batch_size=16, learning_rate=1e-3)

# Fine-tune on new domain data
model.finetune(data=["New domain text fact 1.", "New domain text fact 2."])

# Export to GGUF
model.export("carvus.gguf", format="gguf")
```

### Interactive Multi-Turn Chat

```python
from carvustrain import ChatModel

chat_bot = ChatModel()
response = chat_bot.chat("Hello Carvus!")
print(response)
```

### Deploy to Production

```python
# Start a REST API server
model.serve(port=8000)

# From another terminal:
# curl -X POST http://localhost:8000 -H "Content-Type: application/json" -d '{"prompt":"What is Python?"}'
# curl http://localhost:8000  # Get model info
```

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     User                             │
└──────────┬──────────────────────────┬────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│   Model          │    │   train.ct               │
│  (ChatModel,     │◄───│  (Config + Knowledge)    │
│   TextModel,     │    └──────────────────────────┘
│   LanguageModel) │
└──────┬───────────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐  ┌──────────────────┐
│ KnowledgeBase │  │ InferenceEngine  │
│              │  │ ┌──────────────┐ │
│ • Facts      │  │ │CodeGenerator │ │
│ • TF-IDF     │  │ │QA Engine     │ │
│ • Grammar    │  │ │TextGenerator │ │
│ • Patterns   │  │ │ChatSession   │ │
└──────┬───────┘  └──────┬───────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌──────────────────┐
│LearningValid │  │  Trainer         │
│ ator         │  │ • Optimizer      │
│ • Accuracy   │  │ • Loss           │
│ • Retention  │  │ • Metrics        │
│ • Language   │  │ • Checkpoints    │
└──────────────┘  └──────────────────┘
```

### System Layers

CarvusTrain is organized as a multi-tiered, modular, object-oriented framework for scalability, maintainability, and extensibility.

**1. Base Layer**
- `constants.py` — default hyperparameters, section headers, special tokens
- `exceptions.py` — custom exception hierarchy (`CarvusTrainError`, `TrainingError`, etc.)
- `logger.py` — colorized logging and telemetry
- `utils.py` — device detection, formatting, hardware diagnostics
- `configuration.py` — dataclass config manager (`CarvusConfig`)

**2. Data & Tokenization Layer**
- `parser.py` — multi-format file reader and native `.ct` section parser
- `preprocessing.py` — text cleaning, normalization, padding, sequence chunking
- `tokenizer.py` — Word, Char, BPE, WordPiece, SentencePiece tokenizers
- `dataset.py` — `Dataset`, `TextDataset`, `DataLoader` batching engine

**3. Neural Computation Layer**
- `activation.py` — ReLU, GELU, SiLU, Softmax, Sigmoid, Tanh, LeakyReLU
- `layers.py` — Dense, Embedding, LayerNorm, RMSNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock, Sequential
- `losses.py` — CrossEntropyLoss, MSELoss, BCELoss, SmoothL1Loss, KLDivLoss, Perplexity, FocalLoss
- `metrics.py` — Accuracy, Precision, Recall, F1, BLEU, ROUGE
- `optimizer.py` — Adam, AdamW, SGD, RMSprop, AdaGrad with gradient clipping and weight decay
- `scheduler.py` — StepLR, CosineAnnealingLR, LinearWarmupLR, ReduceLROnPlateau

**4. Memory & Context Layer**
- `memory.py` — `KnowledgeBase`, `MemoryCache` (KV cache), `ContextWindow`
- `postprocessing.py` — Temperature, Top-K, Top-P, repetition penalty

**5. Engine & Interface Layer**
- `trainer.py` — epoch training engine
- `inference.py` — question answering & text generation pipelines
- `evaluation.py` — Evaluator & Benchmarker
- `exporter.py` — model exporter (`.ct`, `.bin`, `.onnx`, `.json`, `.gguf`)
- `model.py` — high-level `Model`, `ChatModel`, `TextModel`, `LanguageModel`, `CustomModel`
- `cli.py` — command line interface (`carvustrain`)

---

## 5. Core Components

### 5.1 Model — The AI Brain

The `Model` class is the central orchestrator and primary high-level interface. Every AI you build starts here.

**Constructors:**

```python
from carvustrain import Model

# Basic
model = Model(name="Carvus")

# With architecture
model = Model(name="Carvus", architecture="transformer", parameters="1B")

# From config
model = Model(config={"model": {"name": "Carvus", "vocab_size": 10000}})

# Specialized subclasses
from carvustrain import ChatModel, TextModel, LanguageModel, AgentModel, CustomModel
```

| Class | Purpose |
|---|---|
| `Model` | Base model with the full feature set |
| `ChatModel` | Optimized for interactive dialogue and chat sessions |
| `TextModel` | Optimized for document completion and text generation |
| `LanguageModel` | Optimized for token prediction / language modeling |
| `AgentModel` | Goal-oriented agent, extends `Model` |
| `CustomModel` | User-defined / customizable architecture |

**AI Coding & Response Modes:**

```python
model.mode = "programmer"      # Code-focused responses
model.mode = "teacher"         # Educational explanations
model.mode = "researcher"      # Research-oriented
model.mode = "creative"        # Creative writing
model.mode = "security"        # Security-focused
model.mode = "data_scientist"  # Data science
model.mode = "general"         # Default balanced mode
```

**Memory Engine:**

```python
model.enable_memory(True)  # Enable long-term conversation memory
model.ask("What is Python?")
model.ask("What did I just ask?")  # Recalls conversation history (if RAG is available)
```

**Personality System:**

```python
model.set_personality(
    style="professional",  # Also: friendly, technical, academic
    humor=30,               # 0-100 scale
    creativity=75           # 0-100 scale
)
```

### 5.2 KnowledgeBase — The Memory

The `KnowledgeBase` is the model's memory. It stores facts, detects programming languages, extracts code patterns, and provides semantic search.

**Core features:**

- **Fact storage** — add, search, and retrieve text facts
- **Programming language detection** — auto-detects 20+ languages
- **Code pattern extraction** — extracts function signatures and class definitions
- **Semantic search** — TF-IDF (pure Python), optional scikit-learn, optional sentence-transformers
- **Grammar knowledge** — built-in English grammar with 62 entries across 6 categories
- **Learning validation** — tracks accuracy, learning scores, and metrics

```python
from carvustrain import KnowledgeBase

kb = KnowledgeBase()

# Add knowledge
kb.add_fact("Python is dynamically typed.")
kb.add_facts(["Fact 1", "Fact 2", "Fact 3"])

# Load built-in grammar (62 entries across 6 categories)
kb.load_grammar_knowledge()

# Semantic search (auto-rebuilds index)
results = kb.search("What is a noun?")

# Search by language
python_facts = kb.search_by_language("Write a function", "python")

# Search grammar specifically
grammar_facts = kb.search_grammar("tenses", topic="tenses")

# Query
langs = kb.get_programming_languages()
facts = kb.get_facts_by_language("python")
grammar = kb.get_facts_by_category("grammar")

# Get stats
print(kb.get_learning_status())
```

**How search works:**

```
Query → Index Dirty? → Rebuild → Semantic → Fallback
                                ├── sentence-transformers (deep)
                                ├── scikit-learn TF-IDF (sparse)
                                └── Pure-Python TF-IDF (always works)
```

### 5.3 LearningValidator — The Quality Check

The `LearningValidator` ensures the model actually learns, tracking four key metrics.

| Metric | Description |
|---|---|
| `accuracy` | Overall learning accuracy (0–1), a weighted combination of retention, understanding, and comprehension |
| `comprehension_score` | Understanding of learned material |
| `knowledge_retention` | Ability to recall key terms from training |
| `language_understanding` | Correct language identification/usage in responses |

```python
from carvustrain import LearningValidator

validator = LearningValidator(kb)

# Validate comprehension
metrics = validator.validate_learning(
    training_texts=["Python is a language"],
    generated_responses=["Python is a programming language"]
)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Retention: {metrics['knowledge_retention']}")

# Check grammar understanding
grammar_score = validator.validate_grammar_understanding(
    "Explain what an adverb is.",
    "An adverb modifies a verb, adjective, or other adverb."
)

# Check convergence
if validator.check_learning_convergence(threshold=0.85):
    print("Learning converged!")

# Review full history
summary = validator.get_validation_summary()
```

### 5.4 RAG Pipeline — Retrieval Augmented Generation

The `RAGPipeline` lets you ingest documents and enhance model responses with relevant retrieved context.

```python
from carvustrain.rag import RAGPipeline

rag = RAGPipeline()

# Ingest from text
rag.ingest_text("Python is a high-level programming language.")

# Ingest from files
rag.ingest_file("documentation.md")
rag.ingest_file("knowledge.json")

# Ingest an entire code repository
rag.ingest_code("/path/to/project")

# Retrieve relevant context
results = rag.retrieve("What is Python?", top_k=3)
for doc, score in zip(results.documents, results.scores):
    print(f"Score: {score:.3f} | {doc.content[:100]}...")

# Generate with RAG context
answer = rag.generate("Explain Python classes", model)
print(answer)
```

### 5.5 CodeGenerator

Synthesizes real, working code for 14+ programming languages via a multi-stage pipeline. See [Section 8](#8-code-generation) for full detail.

### 5.6 InferenceEngine

Unified engine bundling code generation, QA, text generation, and chat.

```python
engine = InferenceEngine(knowledge_base)

# Ask a question (auto-detects code vs general)
answer = engine.ask("What is recursion?")

# Explicit code generation
code = engine.generate_code("Write a BFS in JavaScript", language="javascript")

# Text generation
text = engine.generate("Tell me about algorithms")

# Chat (maintains context)
response = engine.chat("Hello!")
response = engine.chat("What was my last question?")
```

### 5.7 Trainer

Core training engine with integrated learning validation.

```python
trainer = Trainer(
    model=model,
    config=TrainingConfig(
        method="normal",
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
    )
)

history = trainer.train(
    dataset=train_dataset,
    val_dataset=val_dataset,
)
```

The trainer automatically:
- Tracks loss, accuracy, and validation metrics
- Validates learning comprehension
- Checks for learning convergence (early stopping)
- Extracts code patterns from code-rich text
- Logs detailed progress per epoch

### 5.8 Dataset & Tokenization

- `Dataset` — base dataset container
- `TextDataset` — specialized text dataset
- `DataLoader` — batching, shuffling, and iteration loader
- `Tokenizer` — factory creating `WordTokenizer`, `CharTokenizer`, `SentenceTokenizer`, `BPETokenizer`, `WordPieceTokenizer`, `SentencePieceTokenizer`, `CustomTokenizer`
- `DataParser` — multi-format file parser (TXT, CSV, JSON, XML, YAML, Markdown, folder scanning), with auto-encoding detection (UTF-8, UTF-16, ASCII)
- `CarvusTrainParser` — parser for native `[CarvusTrain]` sectioned config/knowledge files

### 5.9 Neural Layer Building Blocks

- **Layers**: Dense/Linear, Embedding, LayerNorm, RMSNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock, FeedForward, Dropout, Residual, Sequential
- **Activations**: ReLU, GELU, SiLU, Softmax, Sigmoid, Tanh, LeakyReLU
- **Losses**: CrossEntropyLoss, MSELoss, BCELoss, SmoothL1Loss, KLDivLoss, Perplexity, FocalLoss
- **Metrics**: Accuracy, Precision, Recall, F1, BLEU, ROUGE
- **Optimizers**: Adam, AdamW, SGD, RMSprop, AdaGrad — with weight decay and gradient clipping
- **Schedulers**: StepLR, CosineAnnealingLR, LinearWarmupLR, ReduceLROnPlateau

---

## 6. Training Configuration (train.ct)

The `train.ct` file is the primary configuration and knowledge source. It uses an INI-like format with sections.

### Section Format

```ini
[CarvusTrain]
Version=1.0
KnowledgeValidation=True
LearnCheck=True
LearningAccuracy=0.98
CodeTrainingMode=structured

[Model]
Name=Carvus
ModelType=transformer
VocabSize=10000
EmbeddingDim=128
HiddenDim=256
NumLayers=4
NumHeads=4
MaxSeqLen=512
Dropout=0.1

[Training]
Method=continuous
Duration=forever
LearningRate=0.001
BatchSize=32
Mode=code_generation
ProgrammingLanguages=all
CodeValidation=True
PatternExtraction=True

[Knowledge]

## Core Identity
Your identity facts here...

## Algorithm Patterns
Your code templates here...

## English Grammar
Your grammar facts here...
```

### Section Reference

| Section | Key Settings | Description |
|---|---|---|
| `[CarvusTrain]` | `Version`, `LearnCheck`, `KnowledgeValidation`, `CodeTrainingMode` | Global configuration |
| `[Model]` | `Name`, `ModelType`, `VocabSize`, `EmbeddingDim`, `HiddenDim`, `NumLayers`, `NumHeads`, `MaxSeqLen`, `Dropout` | Model architecture |
| `[Training]` | `Method`, `Duration`, `LearningRate`, `BatchSize`, `Epochs`, `Mode`, `ProgrammingLanguages`, `EarlyStopping`, `ValidationSplit`, `WeightDecay`, `GradientClipping` | Training parameters |
| `[Knowledge]` | Free-form text with `##` headings | Knowledge facts loaded into the `KnowledgeBase` |

### Knowledge Organization

The `[Knowledge]` section organizes facts by topic using Markdown-style headings:

```ini
[Knowledge]

## Core Identity
Carvus is an advanced AI assistant...

## Python Code Generation Patterns
def binary_search(arr, target):
    ...

## English Grammar & Natural Language Understanding
### Parts of Speech
Nouns represent people, places, things, or ideas...

### Sentence Structure
English follows Subject-Verb-Object (SVO) order...
```

Recommended structure:

```
[Knowledge]

## Core Identity
Your AI's identity and purpose

## Domain Knowledge
Facts organized by topic

## Code Patterns
Working code examples by language

## Grammar & Language
Natural language understanding rules
```

---

## 7. Training & Validation

### Auto AI Trainer

`auto_train()` intelligently configures training for you:

```python
model.auto_train("dataset.txt", auto_optimize=True)
# Automatically determines:
# - Batch size (based on dataset size)
# - Learning rate (based on dataset complexity)
# - Device (auto-detects GPU/CUDA)
```

### Training Configurations

```python
# Standard training
model.train(data="train.ct", epochs=10, method="normal")

# Forever training (streaming)
model.train(data="stream.txt", method="forever", duration="forever")

# Fine-tuning
model.finetune(data="new_data.txt", epochs=5, learning_rate=1e-4)

# Auto-optimized training
history = model.auto_train("large_dataset.txt")
print(f"Final loss: {history['loss'][-1]:.4f}")
```

Supported training flows include normal, incremental, continuous, forever, streaming, fine-tuning, self-supervised, supervised, unsupervised, and reinforcement learning.

### Learning Validation During Training

During training, the validator tracks accuracy, retention, and language understanding after every epoch, and can trigger early stopping on convergence:

```
Epoch 1/10
  LearnCheck → Accuracy: 0.6234 | Retention: 0.5812 | Language: 0.7150
Epoch 2/10
  LearnCheck → Accuracy: 0.7812 | Retention: 0.7431 | Language: 0.8322
Epoch 3/10
  LearnCheck → Accuracy: 0.8921 | Retention: 0.8610 | Language: 0.9145
  ✓ Learning converged at epoch 3 (accuracy > 0.85)
```

---

## 8. Code Generation

### Overview

The `CodeGenerator` synthesizes real, working code — not placeholders — for **14+ languages**: Python, JavaScript, TypeScript, Java, C++, Rust, Go, Kotlin, Swift, Bash, Ruby, PHP, R, Scala.

### The Pipeline

```
Prompt → Language Detection → Algorithm Match → Snippet Match
    → Function Generation → Class Generation → Data Structure
    → Control Flow → KB Fallback
```

1. **Algorithm templates** — matched against 50+ pre-built implementations
2. **Common snippets** — file I/O, HTTP requests, SQL queries, etc.
3. **Function generation** — infers name, params, and body from natural language
4. **Class generation** — builds class skeletons with methods
5. **Data structure examples** — list, dict, map, vector, etc.
6. **Control flow** — loops, conditionals
7. **KB fallback** — retrieves the closest relevant knowledge fact

### Algorithm Templates

| Category | Algorithms Available |
|---|---|
| **Search** | Binary search, linear search |
| **Sort** | Quick sort, merge sort |
| **Graph** | DFS (recursive + iterative), BFS, Dijkstra |
| **Linked List** | Reverse, detect cycle, find middle, merge sorted |
| **Tree** | Inorder, preorder, postorder, level-order |
| **DP** | Fibonacci, LCS, coin change, knapsack, edit distance, max subarray |
| **String** | Reverse string, palindrome check, two sum |
| **Math** | Fibonacci, fizzbuzz |

### Coverage by Language

| Algorithm | Python | JS | Rust | Go |
|---|---|---|---|---|
| Binary Search | ✅ | ✅ | ✅ | ✅ |
| Quick Sort | ✅ | ✅ | ✅ | ✅ |
| Merge Sort | ✅ | ✅ | ✅ | ✅ |
| Fibonacci | ✅ | ✅ | ✅ | ✅ |
| FizzBuzz | ✅ | ✅ | — | ✅ |
| Reverse String | ✅ | — | — | — |
| Palindrome | ✅ | — | — | — |
| Two Sum | ✅ | — | — | — |
| DFS | ✅ | ✅ | ✅ | ✅ |
| BFS | ✅ | ✅ | ✅ | ✅ |
| Dijkstra | ✅ | ✅ | ✅ | ✅ |
| Reverse Linked List | ✅ | ✅ | ✅ | ✅ |
| Detect Cycle | ✅ | ✅ | — | ✅ |
| Find Middle LL | ✅ | — | — | — |
| Merge Sorted LL | ✅ | — | — | — |
| Inorder Traversal | ✅ | ✅ | ✅ | ✅ |
| Preorder Traversal | ✅ | — | — | — |
| Postorder Traversal | ✅ | — | — | — |
| Level Order | ✅ | ✅ | — | ✅ |
| LCS (DP) | ✅ | — | — | — |
| Coin Change (DP) | ✅ | — | — | — |
| Knapsack (DP) | ✅ | — | — | — |
| Edit Distance (DP) | ✅ | — | — | — |
| Max Subarray (DP) | ✅ | — | — | — |
| DP Fibonacci | ✅ | — | — | — |

### Natural Language Detection

All of these prompts resolve to working, executable code:

```python
code = gen.generate_code("Write a depth first search in Python")
code = gen.generate_code("Implement BFS in JavaScript")
code = gen.generate_code("Dijkstra shortest path in Go")
code = gen.generate_code("Quick sort in Rust")
code = gen.generate_code("Reverse a linked list in Python")
```

The `_ALGO_ALIASES` dictionary maps 40+ natural-language phrases to canonical algorithm names, for example:

```python
_ALGO_ALIASES = {
    "depth first search": "dfs",
    "breadth first search": "bfs",
    "shortest path": "dijkstra",
    "longest common subsequence": "longest common subsequence",
    "levenshtein": "edit distance",
    "kadane": "max subarray",
    # ... 40+ more mappings
}
```

### Language Detection Keywords

The system detects programming languages from source keywords, for example:

- **Python** — `def`, `class`, `import`, `print(`, `lambda`, `yield`, `self`
- **JavaScript** — `function`, `const`, `let`, `=>`, `console.log`
- **Rust** — `fn`, `let mut`, `impl`, `struct`, `trait`, `match`
- **Go** — `func`, `package`, `:=`, `defer`, `goroutine`
- **20+ more languages** with comprehensive keyword sets

### Cross-Language Fallback

If a language has no template for a requested algorithm, the generator falls back to a Python reference implementation with a translation hint:

```python
f"// {lang} version of:\n// {python_code_first_line}\n// TODO: translate to {lang}"
```

---

## 9. English Grammar & Natural Language Understanding

### Built-in Grammar Knowledge

The `ENGLISH_GRAMMAR` dictionary contains **62 entries across 6 categories**:

| Category | Entries | Topics |
|---|---|---|
| `parts_of_speech` | 9 | Nouns, pronouns, verbs, adjectives, adverbs, prepositions, conjunctions, determiners, interjections |
| `sentence_structure` | 10 | SVO order, simple/compound/complex sentences, 4 sentence types, appositives |
| `tenses` | 12 | All 12 English tenses with examples |
| `grammar_rules` | 14 | Subject-verb agreement, articles, parallelism, conditionals, reported speech, relative clauses |
| `common_mistakes` | 10 | Their/there/they're, affect/effect, lay/lie, who/whom, less/fewer |
| `writing_style` | 10 | Clarity, variety, transitions, tone, proofreading |

### Loading Grammar Knowledge

```python
kb = KnowledgeBase()
count = kb.load_grammar_knowledge()
print(f"Loaded {count} grammar facts")  # 62

# Search grammar specifically
results = kb.search_grammar("What is a verb?", topic="parts_of_speech")

# Search grammar across all topics
results = kb.search_grammar("How do I use tenses correctly?")
```

### Grammar-Aware Validation

`LearningValidator.validate_grammar_understanding()` checks:

1. **Sentence structure** — proper capitalization and punctuation
2. **Grammar term usage** — correct use of grammar terminology
3. **Response quality** — proper sentence formation

```python
validator = LearningValidator(kb)
score = validator.validate_grammar_understanding(
    "Explain what an adverb is.",
    "An adverb modifies a verb, adjective, or other adverb."
)
```

---

## 10. Semantic Search

### Search Pipeline

```
Query → Index Dirty? → Rebuild Index → Semantic Search → Fallback
                              ↓
                    ┌──────────────────┐
                    │  1. sentence-    │
                    │     transformers │ (deep semantic)
                    ├──────────────────┤
                    │  2. scikit-learn │ (sparse TF-IDF)
                    ├──────────────────┤
                    │  3. Pure Python  │ (always works)
                    └──────────────────┘
```

### Index Rebuilding

The index is rebuilt lazily whenever facts change:

```python
kb.add_fact("New fact here")   # Marks index as dirty
results = kb.search("query")   # Auto-rebuilds if dirty
```

### Fallback Behavior

If no semantic index is available yet (e.g. the first search after creation), the system falls back to word-overlap scoring with language-match and code-construct boosts.

### Language-Specific Search

```python
results = kb.search_by_language("Write a loop", "python")
```

### Performance

```
Knowledge Base Size    Pure Python    scikit-learn    sentence-transformers
1,000 facts            ~0.01s         ~0.001s          ~0.05s (incl. encode)
10,000 facts           ~0.1s          ~0.01s           ~0.2s
100,000 facts          ~1s            ~0.1s            ~1.5s
```

---

## 11. Deployment

### Model Server

Deploy your model as a REST API in one line:

```python
model.serve(port=8000)
```

This starts an HTTP server exposing:

```
POST /  — Ask a question
  Body: {"prompt": "What is Python?"}
  Response: {"response": "Python is...", "model": "Carvus", "mode": "general"}

GET /   — Get model info
  Response: {"name": "Carvus", "architecture": "transformer", ...}
```

### Export Formats

```python
# Native format
model.save("my_model.ct")

# Export to various formats
model.export("model.bin", format="bin")
model.export("model.json", format="json")
model.export("model.onnx", format="onnx")
model.export("model.gguf", format="gguf")
```

---

## 12. Agent System

### Goal-Oriented Agents

`AgentModel` (extends `Model`) lets you build agents around a goal, and compose them into teams.

```python
from carvustrain import AgentModel

# Create an agent with a specific goal
coder = AgentModel(name="CoderBot", goal="software engineer")
coder.train_agent(data="code_examples.txt")

# Create a research agent
researcher = AgentModel(name="ResearcherBot", goal="research")
researcher.train_agent(data="papers.txt")

# Create a team
planner = AgentModel(name="Planner", goal="planning")
planner.add_sub_agent("coder", coder)
planner.add_sub_agent("researcher", researcher)

# Orchestrate a task
results = planner.orchestrate("Build a web app")
print(results["planner"])      # Planning phase
print(results["coder"])        # Code generation
print(results["researcher"])   # Research context
```

---

## 13. Plugin System

CarvusTrain models can be extended with custom plugins that add new capabilities.

```python
# Create a plugin
class VisionPlugin:
    def __init__(self):
        self.name = "vision"
    def process(self, input_data):
        return f"[Vision processing: {input_data}]"

# Install it
model.install_plugin("vision", VisionPlugin())

# Extend with custom abilities
class CodeReviewPlugin:
    def __init__(self):
        self.name = "code_review"
    def review(self, code):
        suggestions = []
        if "import *" in code:
            suggestions.append("Avoid wildcard imports")
        if len(code.split("\n")) > 100:
            suggestions.append("Consider splitting into smaller functions")
        return suggestions

model.install_plugin("code_review", CodeReviewPlugin())
```

---

## 14. CarvusTrain Language (CTL)

CTL is a declarative DSL for AI training configuration, sitting on top of `train.ct`'s section format.

```ctl
model Carvus {
    architecture = "transformer"
    parameters = "base"
    vocab_size = 10000

    training {
        data = "train.ct"
        method = "continuous"
        epochs = forever
        memory = enabled
        validation = enabled
        accuracy_target = 0.95
    }

    knowledge {
        grammar = enabled
        languages = all
        code_training = structured
    }

    personality {
        style = "professional"
        humor = 20
        creativity = 80
    }

    deploy {
        server = enabled
        port = 8000
        export = ["ct", "json", "bin"]
    }
}
```

---

## 15. CLI Reference

CarvusTrain includes a full command-line tool named `carvustrain`.

| Command | Description |
| :--- | :--- |
| `carvustrain init` | Initialize a new workspace project |
| `carvustrain train` | Train a model from a dataset or config file |
| `carvustrain chat` | Interactive multi-turn terminal chat |
| `carvustrain predict` | Generate a prediction for a prompt |
| `carvustrain serve` | Deploy the model as a REST API server |
| `carvustrain export` | Export model to ONNX, GGUF, BIN, JSON, CT |
| `carvustrain evaluate` | Evaluate accuracy and loss on a dataset |
| `carvustrain benchmark` | Measure throughput (tokens/sec) and latency |
| `carvustrain convert` | Convert dataset format |
| `carvustrain doctor` | System diagnostic health check |
| `carvustrain config` | View and validate config files |
| `carvustrain install` | Display installation instructions for optional packages |
| `carvustrain update` | Check framework update status |
| `carvustrain version` | Print version and license info |

### Examples

**Create a new AI project**
```bash
carvustrain create my-ai
carvustrain init my-ai
```

**Train a model**
```bash
carvustrain train --data dataset.txt --name Carvus --epochs 10 --batch-size 32 --output model.ct
```

**Interactive chat**
```bash
carvustrain chat --model model.ct
```

**Generate a prediction**
```bash
carvustrain predict --model my_ai.ct --prompt "Hello"
```

**Deploy a REST API server**
```bash
carvustrain serve --model model.ct --port 8000
```

**Export to ONNX / GGUF**
```bash
carvustrain export --model model.ct --output model.onnx --format onnx
carvustrain export --model model.ct --output model.gguf --format gguf
```

**Evaluate & benchmark**
```bash
carvustrain evaluate --model my_ai.ct --data test.txt
carvustrain benchmark --model my_ai.ct --runs 50
```

**Convert datasets**
```bash
carvustrain convert --input data.csv --output data.json
```

**System diagnostics**
```bash
carvustrain doctor
carvustrain version
carvustrain config
```

---

## 16. Performance & Optimization

### Semantic Search Performance

```
Knowledge Base Size    Pure Python    scikit-learn    sentence-transformers
1,000 facts            ~0.01s         ~0.001s          ~0.05s (incl encode)
10,000 facts           ~0.1s          ~0.01s           ~0.2s
100,000 facts          ~1s            ~0.1s            ~1.5s
```

### Memory Optimization

```python
# For large knowledge bases, install fast dependencies
# pip install scikit-learn  -> 10x faster TF-IDF
# pip install faiss-cpu     -> 100x faster vector search
# pip install sentence-transformers -> deeper understanding
```

### Training Speed Tips

- Use `auto_train()` for optimal batch sizes
- Enable mixed precision: `mixed_precision=True`
- Use GPU: `device="cuda"` or `gpu=True`
- Set workers for data loading: `workers=4`

### Training Strategy by Data Size

| Data Size | Method | Batch Size | LR | Epochs |
|---|---|---|---|---|
| < 100KB | normal | 32 | 1e-3 | 5-10 |
| 100KB-1MB | normal | 16 | 5e-4 | 10-20 |
| 1MB-10MB | finetune | 8 | 1e-4 | 3-5 |
| > 10MB | streaming | 4 | 5e-5 | continuous |

---

## 17. Best Practices

### Knowledge Organization

```
[Knowledge]

## Core Identity
Your AI's identity and purpose

## Domain Knowledge
Facts organized by topic

## Code Patterns
Working code examples by language

## Grammar & Language
Natural language understanding rules
```

### Training

1. **Start with train.ct** — load all foundational knowledge from a well-structured configuration.
2. **Load grammar knowledge** — call `kb.load_grammar_knowledge()` for better NLU.
3. **Use structured data** — organize training text by topic with clear headings.
4. **Monitor learning metrics** — check accuracy, retention, and convergence.
5. **Enable early stopping** — set `EarlyStopping` in `train.ct` for automatic convergence detection.
6. **Use code validation** — enable `CodeValidation=True` to extract patterns during training.

### Code Generation

1. **Be specific** — "Write a DFS algorithm in Python" works better than "write code."
2. **Use natural language** — the alias mapping handles "depth first search" → "dfs" automatically.
3. **Specify the language** — if not auto-detected, the generator falls back to Python.
4. **Combine with the KB** — for complex code, first teach examples via `train.ct`, then ask for custom code.

### Knowledge Management

1. **Organize by category** — use metadata tags for filtering (grammar, code, general).
2. **Use diverse examples** — include multiple programming languages and algorithm variants.
3. **Balance code and natural language** — both are important for a well-rounded model.
4. **Update train.ct regularly** — add new knowledge and algorithm templates over time.

### Validation

1. **Track convergence** — use `check_learning_convergence()` to know when training is complete.
2. **Monitor all metrics** — don't just watch loss; accuracy, retention, and language understanding matter.
3. **Validate grammar understanding** — use `validate_grammar_understanding()` for NL tasks.
4. **Review validation history** — the `validation_history` list shows progress over time.
5. **Enable `LearnCheck=True`** in `train.ct`.
6. **Set `LearningAccuracy=0.95`** for high-quality learning.
7. **Use `EarlyStopping=10`** to prevent overfitting.

---

## 18. Extending CarvusTrain

### Adding a New Programming Language

1. Add keyword signatures to `PROGRAMMING_LANGUAGES` in `memory.py`:
   ```python
   "my_lang": {"keyword1", "keyword2", "syntax_pattern"},
   ```
2. Add a function skeleton to `FUNCTION_SKELETONS` in `inference.py`:
   ```python
   "my_lang": "function {name}({params}) { {body} }",
   ```
3. Add algorithm templates to `ALGORITHM_TEMPLATES`:
   ```python
   "my_lang": { "binary search": "...", "fibonacci": "..." },
   ```
4. Add code patterns to `_extract_code_patterns` in `KnowledgeBase`.

### Adding New Algorithm Templates

```python
from carvustrain.inference import CodeGenerator

# Add a new Python template
CodeGenerator.ALGORITHM_TEMPLATES["python"]["my algorithm"] = (
    "def my_algorithm(data):\n"
    "    result = []\n"
    "    for item in data:\n"
    "        result.append(process(item))\n"
    "    return result"
)

# Add a new language alias
CodeGenerator._ALGO_ALIASES["my algorithm description"] = "my algorithm"
```

1. Add the template to `ALGORITHM_TEMPLATES` for each target language.
2. Add a natural-language alias to `_ALGO_ALIASES`.
3. Add the algorithm to `train.ct`'s knowledge section.
4. (Optional) add a test case to the test suite.

### Adding New Grammar Knowledge

1. Add entries to `ENGLISH_GRAMMAR` in `memory.py` under the appropriate category.
2. Add corresponding entries to the `[Knowledge]` section in `train.ct`.
3. (Optional) add validation logic in `LearningValidator`.

### Custom Validators

```python
class CustomValidator(LearningValidator):
    def validate_custom_metric(self, texts, responses):
        # Custom validation logic
        return custom_score
```

---

## 19. Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ImportError` on startup | Package not installed in editable mode | Run `pip install -e .` |
| `SyntaxError` in `memory.py` | Unescaped quotes in string literals | Use `\"` for inner double quotes |
| RAG pipeline fails | Optional dependency missing | RAG is optional — the model works without it |
| Slow search | TF-IDF index rebuilds every search | Install scikit-learn: `pip install scikit-learn`; index is lazy-built, ensure facts aren't constantly changing |
| Memory errors | Batch size too large for available RAM | Reduce batch size, use streaming mode |
| Server won't start | Port already in use | Try a different port: `model.serve(port=9000)` |
| Model always returns Python code | Language detection needs more keywords | Add language-specific keywords to `PROGRAMMING_LANGUAGES` |
| Model not learning | `LearnCheck` disabled or data too sparse | Check `LearnCheck=True`, add more diverse training data |
| Learning accuracy stays at 0 | No validation data or no responses generated | Ensure training texts have extractable key terms |
| Grammar not working / returns nothing | Grammar knowledge not loaded | Call `kb.load_grammar_knowledge()` before asking grammar questions |
| Code generation returns TODO | Requested algorithm not yet templated for that language | Add a template or rely on the cross-language fallback |

### Debug Mode

```python
import carvustrain.logger as logger
logger.set_level("DEBUG")

# Now you'll see detailed logs about search, code generation, and validation
model.ask("Write a function in Python")
```

### Performance Tips

- For large knowledge bases, install scikit-learn for faster TF-IDF: `pip install scikit-learn`
- For deep semantic search, install sentence-transformers: `pip install sentence-transformers`
- Use `search_by_language()` instead of general `search()` when possible
- Organize facts by category metadata for faster filtering

---

## 20. Full API Reference

### `Model`

```python
class Model:
    def __init__(self, name="Carvus", architecture="transformer", parameters="base",
                 mode="general", config=None, auto_load=True, **kwargs)
    def learn(self, text: Union[str, List[str]]) -> None
    def train(self, data=None, method="normal", duration=10, epochs=None, batch_size=None,
              learning_rate=None, optimizer=None, device="auto", cpu=False, cuda=False,
              gpu=False, mixed_precision=False, workers=0, shuffle=True, seed=42,
              checkpoint=True, resume=False, validation_split=0.1, early_stopping=None,
              dropout=None, weight_decay=None, gradient_clipping=None, **kwargs) -> Dict
    def auto_train(self, data=None, auto_optimize=True) -> Dict
    def finetune(self, data, epochs=5, learning_rate=1e-4, **kwargs) -> Dict
    def ask(self, question: str) -> str
    def predict(self, input_text: str) -> str
    def generate(self, prompt: str, max_new_tokens=100, **kwargs) -> str
    def chat(self, message: str) -> str
    def evaluate(self, dataset=None) -> Dict[str, float]
    def benchmark(self, prompt="Who are you?", num_runs=20) -> Dict
    def save(self, filepath=None) -> str
    def export(self, output_path, format=None) -> str
    def load(self, filepath) -> "Model"
    def serve(self, port=8000, host="0.0.0.0") -> None
    def summary(self) -> None
    def statistics(self) -> Dict
    def memory(self) -> Dict
    def to_dict(self) -> Dict
    def enable_memory(self, enabled=True) -> None
    def set_personality(self, style="professional", humor=20, creativity=80) -> None
    def install_plugin(self, name, plugin) -> None
```

`ask()` also sets `model.answer` as a side effect, so `model.ask("...")` followed by `print(model.answer)` is a valid pattern.

### `AgentModel` (extends `Model`)

```python
class AgentModel(Model):
    def __init__(self, name="CarvusAgent", goal="software engineer", **kwargs)
    def train_agent(self, goal=None, data=None) -> Dict
    def add_sub_agent(self, name, agent) -> None
    def orchestrate(self, task) -> Dict[str, str]
```

### `KnowledgeBase`

```python
class KnowledgeBase:
    def __init__(self)
    def add_fact(self, text, meta=None) -> None
    def add_facts(self, texts: List[str]) -> None
    def clear(self) -> None
    def rebuild_index(self) -> None
    def search(self, query, top_k=3) -> List[Tuple[str, float]]
    def search_by_language(self, query, language, top_k=3) -> List[Tuple[str, float]]
    def search_grammar(self, query, topic=None, top_k=3) -> List[Tuple[str, float]]
    def load_grammar_knowledge(self) -> int
    def get_programming_languages(self) -> List[str]
    def get_facts_by_language(self, language) -> List[str]
    def get_facts_by_category(self, category) -> List[str]
    def has_code_knowledge(self) -> bool
    def get_average_learning_score(self) -> float
    def get_average_accuracy(self) -> float
    def get_learning_status(self) -> Dict
    def _detect_language(self, text) -> Optional[str]
```

### `LearningValidator`

```python
class LearningValidator:
    def __init__(self, knowledge_base)
    def validate_learning(self, training_texts, generated_responses) -> Dict[str, float]
    def validate_grammar_understanding(self, text, response) -> float
    def check_learning_convergence(self, threshold=0.85, window=5) -> bool
    def get_validation_summary(self) -> Dict[str, Any]
```

### `RAGPipeline`

```python
class RAGPipeline:
    def ingest_text(self, text, metadata=None) -> int
    def ingest_file(self, filepath) -> int
    def ingest_code(self, dirpath) -> int
    def retrieve(self, query, top_k=5) -> RetrievalResult
    def generate(self, query, model, top_k=3) -> str
    def clear(self) -> None
    @property
    def stats(self) -> Dict
```

### `CodeGenerator` (via `InferenceEngine`)

```python
class CodeGenerator:
    def __init__(self, knowledge_base)
    def generate_code(self, prompt, language=None) -> str
    def generate(self, prompt, max_length=40, language=None) -> str
    # Class-level dictionaries:
    # - FUNCTION_SKELETONS: Dict[str, str] — 14 languages
    # - CLASS_SKELETONS: Dict[str, str] — 6 languages
    # - LOOP_SKELETONS: Dict[str, str] — 10 languages
    # - CONDITIONAL_SKELETONS: Dict[str, str] — 7 languages
    # - DATA_STRUCTURE_EXAMPLES: Dict[str, Dict[str, str]] — 7 languages
    # - ALGORITHM_TEMPLATES: Dict[str, Dict[str, str]] — 4+ languages, 50+ templates
    # - COMMON_SNIPPETS: Dict[str, Dict[str, str]] — 4 languages
    # - _ALGO_ALIASES: Dict[str, str] — 40+ mappings

# Supported languages: python, javascript, typescript, java, cpp, rust, go,
#                       kotlin, swift, bash, ruby, php, r, scala
```

### Data & Tokenization

```python
Dataset            # Base dataset container
TextDataset        # Specialized text dataset
DataLoader          # Batching, shuffling, and iteration loader
Tokenizer           # Factory: WordTokenizer, CharTokenizer, BPETokenizer,
                    #          WordPieceTokenizer, SentencePieceTokenizer
DataParser          # Multi-format file parser (TXT, CSV, JSON, XML, YAML, Markdown, Folders)
CarvusTrainParser   # Parser for native [CarvusTrain] sectioned files
```

---

## 21. Changelog

All notable changes to the `carvustrain` project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [1.1.0] — 2026-07-25

**Fixed**
- **Import resolution** — changed all internal `from carvustrain.xxx` absolute imports to relative `from .xxx` imports across 13 package files. This fixes `ModuleNotFoundError` on case-sensitive filesystems and makes the package robust regardless of the installed module name.
- **CLI entry point** — corrected the console script reference in `pyproject.toml` to match the actual package module name.
- **RAG Pipeline** — added `from carvustrain.rag import RAGPipeline` for document ingestion and retrieval-augmented generation.
- **Tests** — added comprehensive test suite with `test_all_examples.py` covering all examples and core features.

### [1.0.0] — 2026-07-25

**Added**
- **Core Framework** — initial release of the CarvusTrain AI framework, built from scratch in Python.
- **High-Level Model API** — `Model`, `ChatModel`, `TextModel`, `LanguageModel`, and `CustomModel` classes.
- **Expressive Methods** — `train()`, `ask()`, `predict()`, `generate()`, `chat()`, `learn()`, `finetune()`, `evaluate()`, `export()`, `load()`, `save()`, `summary()`, `statistics()`, and `memory()`.
- **Dataset System** — multi-format data parser supporting TXT, CSV, JSON, JSONL, XML, YAML, Markdown, folder scanning, and auto-encoding detection (UTF-8, UTF-16, ASCII).
- **Custom Section Parser** — parser for native `[CarvusTrain]` config and knowledge files (`[CarvusTrain]`, `[Model]`, `[Training]`, `[Knowledge]`).
- **Tokenization Suite** — `WordTokenizer`, `CharTokenizer`, `SentenceTokenizer`, `BPETokenizer`, `WordPieceTokenizer`, `SentencePieceTokenizer`, and `CustomTokenizer` with vocabulary management.
- **Neural Layer Architecture** — Dense/Linear, Embedding, LayerNorm, RMSNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock, FeedForward, Dropout, Residual, and Sequential layers.
- **Optimizers & Schedulers** — Adam, AdamW, SGD, RMSprop, AdaGrad with weight decay, gradient clipping, step/cosine/warmup/plateau schedulers.
- **Trainer Engine** — support for normal, incremental, continuous, forever, streaming, fine-tuning, self-supervised, supervised, unsupervised, and reinforcement learning flows.
- **Exporting Suite** — support for saving and exporting models to `.ct`, `.bin`, `.onnx`, `.json`, and `.gguf` formats.
- **Command-Line Interface** — full `carvustrain` CLI with `init`, `train`, `predict`, `chat`, `export`, `evaluate`, `benchmark`, `convert`, `version`, `doctor`, `config`, `install`, `update`.
- **Documentation & Examples** — extensive Markdown documentation under `docs/` and runnable Python scripts under `examples/`.
- **Test Suite** — unit and integration tests for modules under `tests/`.

---

*CarvusTrain — Making AI Development Accessible to Everyone*
*July 2026*