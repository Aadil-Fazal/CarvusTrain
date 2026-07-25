# CarvusTrain — Developer Guide

> **Version:** 1.0  
> **Last updated:** July 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quick Start](#2-quick-start)
3. [Core Components](#3-core-components)
   - [Model](#31-model)
   - [KnowledgeBase](#32-knowledgebase)
   - [LearningValidator](#33-learningvalidator)
   - [CodeGenerator](#34-codegenerator)
   - [InferenceEngine](#35-inferenceengine)
   - [Trainer](#36-trainer)
4. [Training Configuration (train.ct)](#4-training-configuration-trainct)
5. [Code Generation](#5-code-generation)
6. [English Grammar & Natural Language Understanding](#6-english-grammar--natural-language-understanding)
7. [Learning Validation](#7-learning-validation)
8. [Semantic Search](#8-semantic-search)
9. [Algorithm Templates](#9-algorithm-templates)
10. [Best Practices](#10-best-practices)
11. [Extending CarvusTrain](#11-extending-carvustrain)
12. [Troubleshooting](#12-troubleshooting)
13. [API Reference](#13-api-reference)

---

## 1. Architecture Overview

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

### Key Design Principles

- **Knowledge-first**: The model learns by accumulating facts in a semantic knowledge base, not by training billions of parameters.
- **Real code generation**: Algorithm templates generate syntactically valid, working code with no placeholders.
- **Grammar-aware**: Built-in English grammar knowledge helps the model understand and generate natural language.
- **Validation-driven**: Learning accuracy, knowledge retention, and language understanding are tracked and validated.
- **Zero external deps**: Core functionality works with pure Python — optional upgrades via scikit-learn or sentence-transformers.

---

## 2. Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/your-repo/carvustrain.git
cd carvustrain
pip install -e .
```

### Creating a Model

```python
from carvustrain import ChatModel

# Create a model
model = ChatModel(name="MyAssistant")
print(f"Model: {model.name}")
```

### Loading Knowledge

```python
# Load from train.ct (auto-loads if file exists in CWD)
model = ChatModel(auto_load=True)

# Or load manually
model.knowledge_base.load_grammar_knowledge()  # Load 62 grammar facts
model.knowledge_base.add_fact("Python is a programming language.")
model.knowledge_base.add_fact("The Earth orbits the Sun.")
```

### Training

```python
# Train with text data
data = [
    "Python supports multiple programming paradigms.",
    "JavaScript is used for web development.",
    "Rust provides memory safety without garbage collection.",
]
model.train(data, method="normal", epochs=5)
```

### Asking Questions

```python
# Ask a coding question
answer = model.ask("Write a function that performs binary search in Python")
print(answer)

# Ask a grammar question
answer = model.ask("What are the parts of speech in English?")
print(answer)

# Ask a general question
answer = model.ask("Who are you?")
print(answer)
```

### Code Generation

```python
# Generate code directly
code = model.inference_engine.generate_code(
    "Write a Dijkstra shortest path algorithm in Go",
    language="go"
)
print(code)
```

---

## 3. Core Components

### 3.1 Model

The `Model` class is the primary high-level interface. Subclasses include:

| Class | Purpose |
|---|---|
| `Model` | Base model with full feature set |
| `ChatModel` | Optimized for interactive chat |
| `TextModel` | Optimized for text generation |
| `LanguageModel` | Optimized for language modeling |
| `CustomModel` | User-defined architecture |

**Key methods:**

| Method | Description |
|---|---|
| `learn(text)` | Ingest facts into knowledge base with validation |
| `train(data, ...)` | Full training pipeline with metrics |
| `ask(question)` | Answer questions using knowledge + code gen |
| `generate(prompt)` | Generate text continuation or code |
| `chat(message)` | Multi-turn interactive conversation |
| `predict(input)` | Alias for `ask()` |
| `evaluate(dataset)` | Evaluate model with learning validation metrics |
| `finetune(data, ...)` | Fine-tune on new data |
| `save(filepath)` | Save model to .ct file |
| `load(filepath)` | Load model from .ct file |
| `summary()` | Print model architecture summary |
| `statistics()` | Get model statistics |

### 3.2 KnowledgeBase

The `KnowledgeBase` is the model's memory. It stores facts, detects programming languages, extracts code patterns, and provides semantic search.

**Core features:**

- **Fact storage**: Add, search, and retrieve text facts
- **Programming language detection**: Auto-detects 20+ languages
- **Code pattern extraction**: Extracts function signatures and class definitions
- **Semantic search**: TF-IDF (pure Python), optional scikit-learn, optional sentence-transformers
- **Grammar knowledge**: Built-in English grammar with 62 entries across 6 categories
- **Learning validation**: Tracks accuracy, learning scores, and metrics

**Key methods:**

```python
kb = KnowledgeBase()

# Add facts
kb.add_fact("Python is dynamically typed.")
kb.add_facts(["Fact 1", "Fact 2"])
kb.load_grammar_knowledge()  # Load built-in grammar (62 facts)

# Search
results = kb.search("What is Python?")
results = kb.search_by_language("Write a function", "python")
results = kb.search_grammar("What is a noun?", topic="parts_of_speech")

# Query
langs = kb.get_programming_languages()
facts = kb.get_facts_by_language("python")
grammar = kb.get_facts_by_category("grammar")
status = kb.get_learning_status()
```

### 3.3 LearningValidator

Validates that the model correctly learns and retains knowledge.

**Metrics tracked:**

| Metric | Description |
|---|---|
| `accuracy` | Overall learning accuracy (0-1) |
| `comprehension_score` | Understanding of learned material |
| `knowledge_retention` | Ability to retain key terms |
| `language_understanding` | Correct language usage in responses |

**Key methods:**

```python
validator = LearningValidator(knowledge_base)

# Validate learning
metrics = validator.validate_learning(training_texts, responses)
# Returns: {accuracy, comprehension_score, knowledge_retention, language_understanding}

# Validate grammar understanding
grammar_score = validator.validate_grammar_understanding(text, response)

# Check convergence
converged = validator.check_learning_convergence(threshold=0.85, window=5)

# Get summary
summary = validator.get_validation_summary()
```

### 3.4 CodeGenerator

Synthesizes real, working code snippets for 14+ programming languages.

**Supported languages:** Python, JavaScript, TypeScript, Java, C++, Rust, Go, Kotlin, Swift, Bash, Ruby, PHP, R, Scala

**Code generation pipeline:**

1. **Algorithm templates**: Match against 50+ pre-built algorithm implementations
2. **Common snippets**: File I/O, HTTP requests, SQL queries, etc.
3. **Function generation**: Infer name, params, body from natural language
4. **Class generation**: Build class skeletons with methods
5. **Data structure examples**: List, dict, map, vector, etc.
6. **Control flow**: Loops, conditionals
7. **KB fallback**: Retrieve relevant knowledge fact

**Algorithm templates available:**

| Category | Algorithms |
|---|---|
| **Searching** | Binary search, linear search |
| **Sorting** | Quick sort, merge sort |
| **Graph** | DFS, BFS, Dijkstra |
| **Linked List** | Reverse, detect cycle, find middle, merge sorted |
| **Tree** | Inorder, preorder, postorder, level-order traversal |
| **DP** | Fibonacci (iterative + DP), LCS, coin change, knapsack, edit distance, max subarray |
| **String** | Reverse string, palindrome check, two sum |
| **Math** | Fibonacci, fizzbuzz |

### 3.5 InferenceEngine

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

### 3.6 Trainer

Core training engine with learning validation.

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

---

## 4. Training Configuration (train.ct)

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
| `[Knowledge]` | Free-form text with `##` headings | Knowledge facts loaded into KnowledgeBase |

### Knowledge Organization

The `[Knowledge]` section organizes knowledge by topic:

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

---

## 5. Code Generation

### How It Works

The `CodeGenerator` uses a multi-stage pipeline to synthesize code:

```
Prompt → Language Detection → Algorithm Match → Snippet Match
    → Function Generation → Class Generation → Data Structure
    → Control Flow → KB Fallback
```

### Algorithm Detection

The `_ALGO_ALIASES` dictionary maps natural language phrases to algorithm names:

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

### Natural Language Detection

The system detects programming languages from text keywords:

- **Python**: `def`, `class`, `import`, `print(`, `lambda`, `yield`, `self`
- **JavaScript**: `function`, `const`, `let`, `=>`, `console.log`
- **Rust**: `fn`, `let mut`, `impl`, `struct`, `trait`, `match`
- **Go**: `func`, `package`, `:=`, `defer`, `goroutine`
- **20+ more languages** with comprehensive keyword sets

### Cross-Language Fallback

If a language has no template for a requested algorithm, the generator falls back to Python and provides a translation hint:

```python
f"// {lang} version of:\n// {python_code_first_line}\n// TODO: translate to {lang}"
```

---

## 6. English Grammar & Natural Language Understanding

### Built-in Grammar Knowledge

The `ENGLISH_GRAMMAR` dictionary contains 62 entries across 6 categories:

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
# Auto-load all 62 grammar facts
kb = KnowledgeBase()
count = kb.load_grammar_knowledge()
print(f"Loaded {count} grammar facts")  # 62

# Search grammar specifically
results = kb.search_grammar("What is a verb?", topic="parts_of_speech")

# Search grammar across all topics
results = kb.search_grammar("How do I use tenses correctly?")
```

### Grammar-Aware Validation

The `LearningValidator.validate_grammar_understanding()` method checks:

1. **Sentence structure**: Proper capitalization and punctuation
2. **Grammar term usage**: Correct use of grammar terminology
3. **Response quality**: Proper sentence formation

---

## 7. Learning Validation

### Metrics

The validation system tracks 4 key metrics:

1. **Accuracy**: Weighted combination of retention, understanding, and comprehension
2. **Comprehension Score**: How well the model understands learned material
3. **Knowledge Retention**: Ability to recall key terms from training
4. **Language Understanding**: Correct language identification in responses

### Training Integration

During training, the validator runs after each epoch:

```
Epoch 1/10
  LearnCheck → Accuracy: 0.6234 | Retention: 0.5812 | Language: 0.7150
Epoch 2/10
  LearnCheck → Accuracy: 0.7812 | Retention: 0.7431 | Language: 0.8322
Epoch 3/10
  LearnCheck → Accuracy: 0.8921 | Retention: 0.8610 | Language: 0.9145
  ✓ Learning converged at epoch 3 (accuracy > 0.85)
```

### Grammar-Enhanced Validation

The `LearningValidator` also includes grammar-specific validation:

```python
validator = LearningValidator(kb)
score = validator.validate_grammar_understanding(
    "Explain what an adverb is.",
    "An adverb modifies a verb, adjective, or other adverb."
)
```

---

## 8. Semantic Search

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

The index is rebuilt lazily when facts change:

```python
kb.add_fact("New fact here")        # Marks index as dirty
results = kb.search("query")         # Auto-rebuilds if dirty
```

### Fallback Behavior

If no semantic index is available (e.g., first search after creation), the system uses word-overlap scoring with language-match and code-construct boosts.

### Language-Specific Search

```python
# Search only within Python-related facts
results = kb.search_by_language("Write a loop", "python")
```

---

## 9. Algorithm Templates

### Available Templates by Language

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

### Adding Custom Templates

You can extend the `ALGORITHM_TEMPLATES` dictionary in `CodeGenerator`:

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

---

## 10. Best Practices

### Training

1. **Start with train.ct**: Load all foundational knowledge from a well-structured configuration.
2. **Load grammar knowledge**: Call `kb.load_grammar_knowledge()` for better NLU.
3. **Use structured data**: Organize training text by topic with clear headings.
4. **Monitor learning metrics**: Check accuracy, retention, and convergence.
5. **Enable early stopping**: Set `EarlyStopping` in train.ct for automatic convergence detection.
6. **Use code validation**: Enable `CodeValidation=True` to extract patterns during training.

### Code Generation

1. **Be specific**: "Write a DFS algorithm in Python" works better than "write code."
2. **Use natural language**: The alias mapping handles "depth first search" → "dfs" automatically.
3. **Specify the language**: If not auto-detected, the generator falls back to Python.
4. **Combine with KB**: For complex code, first teach examples via train.ct, then ask for custom code.

### Knowledge Management

1. **Organize by category**: Use metadata tags for filtering (grammar, code, general).
2. **Use diverse examples**: Include multiple programming languages and algorithm variants.
3. **Balance code and natural language**: Both are important for a well-rounded model.
4. **Update train.ct regularly**: Add new knowledge and algorithm templates over time.

### Validation

1. **Track convergence**: Use `check_learning_convergence()` to know when training is complete.
2. **Monitor all metrics**: Don't just watch loss — accuracy, retention, and language understanding matter.
3. **Validate grammar understanding**: Use `validate_grammar_understanding()` for NL tasks.
4. **Review validation history**: The `validation_history` list shows progress over time.

---

## 11. Extending CarvusTrain

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

1. Add the template to `ALGORITHM_TEMPLATES` for each target language.
2. Add a natural-language alias to `_ALGO_ALIASES`.
3. Add the algorithm to train.ct's knowledge section.
4. (Optional) Add a test case to the test suite.

### Adding New Grammar Knowledge

1. Add entries to `ENGLISH_GRAMMAR` in `memory.py` under the appropriate category.
2. Add corresponding entries to the `[Knowledge]` section in `train.ct`.
3. (Optional) Add validation logic in `LearningValidator`.

### Custom Validators

Extend `LearningValidator` with custom validation methods:

```python
class CustomValidator(LearningValidator):
    def validate_custom_metric(self, texts, responses):
        # Custom validation logic
        return custom_score
```

---

## 12. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|---|---|---|
| `SyntaxError` in memory.py | Unescaped quotes in string literals | Use `\"` for inner double quotes |
| Model always returns Python code | Language detection needs more keywords | Add language-specific keywords to `PROGRAMMING_LANGUAGES` |
| Slow search on large KB | TF-IDF index rebuilds every search | The index is lazy-built; ensure facts aren't constantly changing |
| Learning accuracy stays at 0 | No validation data or no responses generated | Ensure training texts have extractable key terms |
| Code generation returns TODO | Algorithm not implemented for target language | Add template or use cross-language fallback |
| Grammar search returns nothing | Grammar knowledge not loaded | Call `kb.load_grammar_knowledge()` |

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

## 13. API Reference

### Model

```python
class Model:
    def __init__(self, name="Carvus", config=None, auto_load=True, **kwargs)
    def learn(self, text: Union[str, List[str]]) -> None
    def train(self, data, method="normal", duration=10, epochs=None, batch_size=None,
              learning_rate=None, optimizer=None, device="auto", cpu=False, cuda=False,
              gpu=False, mixed_precision=False, workers=0, shuffle=True, seed=42,
              checkpoint=True, resume=False, validation_split=0.1, early_stopping=None,
              dropout=None, weight_decay=None, gradient_clipping=None, **kwargs) -> Dict
    def ask(self, question: str) -> str
    def predict(self, input_text: str) -> str
    def generate(self, prompt, max_new_tokens=100, **kwargs) -> str
    def chat(self, message: str) -> str
    def finetune(self, data, epochs=5, learning_rate=1e-4, **kwargs) -> Dict
    def evaluate(self, dataset=None) -> Dict[str, float]
    def benchmark(self, prompt="Who are you?", num_runs=20) -> Dict
    def save(self, filepath=None) -> str
    def export(self, output_path, format=None) -> str
    def load(self, filepath) -> Model
    def summary(self) -> None
    def statistics(self) -> Dict
    def memory(self) -> Dict
    def to_dict(self) -> Dict
```

### KnowledgeBase

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

### LearningValidator

```python
class LearningValidator:
    def __init__(self, knowledge_base)
    def validate_learning(self, training_texts, generated_responses) -> Dict[str, float]
    def validate_grammar_understanding(self, text, response) -> float
    def check_learning_convergence(self, threshold=0.85, window=5) -> bool
    def get_validation_summary(self) -> Dict[str, Any]
```

### CodeGenerator

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
```

---

*For more information, see the [architecture documentation](architecture.md) and [API reference](api_reference.md).*
