"""
CarvusTrain — End-to-End Example Test Runner
Run: python test_all_examples.py

This runs all 4 official examples plus additional integration tests.
"""
import os
import sys
import traceback

# Add project root to path for source imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0

# --- Can we import the package? ---
try:
    import CarvusTrain as carvustrain
    USING_SOURCE = True
except ImportError:
    try:
        import carvustrain
        USING_SOURCE = False
    except ImportError as e:
        print(f"FATAL: Cannot import CarvusTrain: {e}")
        print("Make sure you're running from the project root or have it installed.")
        sys.exit(1)


def run_test(name, fn):
    """Run a single test and report pass/fail."""
    global PASS, FAIL
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"\n  ✅ PASS: {name}")
        PASS += 1
    except Exception as e:
        print(f"\n  ❌ FAIL: {name}")
        traceback.print_exc()
        FAIL += 1


# ═══════════════════════════════════════════════════════════════
# Example 1: Basic Usage
# ═══════════════════════════════════════════════════════════════
def test_example1_basic_usage():
    """Example 1: Basic model training, inference, save (01_basic_usage.py)"""
    model = carvustrain.Model(name="TestCarvus")

    # Create training data
    data_file = os.path.join(PROJECT_ROOT, "_test_train.txt")
    with open(data_file, "w", encoding="utf-8") as f:
        f.write("Carvus is an advanced artificial intelligence assistant built with CarvusTrain.\n")
        f.write("CarvusTrain simplifies deep learning model training.\n")

    # Train the model
    print("  Training model...")
    model.train(data=data_file, method="normal", duration=2)

    # Save model
    model_path = os.path.join(PROJECT_ROOT, "_test_carvus.ct")
    saved_path = model.save(model_path)
    assert os.path.exists(saved_path), f"Model file not saved at {saved_path}"
    print(f"  Saved model: {saved_path}")

    # Inference
    answer = model.ask("Who are you?")
    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    print(f"  Inference: {answer[:120]}")

    # Summary
    model.summary()

    # Memory stats
    stats = model.memory()
    assert isinstance(stats, dict)
    print(f"  Knowledge facts: {stats['stored_facts']}")

    # Cleanup
    for f in [data_file, model_path, saved_path]:
        if os.path.exists(f):
            os.remove(f)


# ═══════════════════════════════════════════════════════════════
# Example 2: Custom .ct Carvus File
# ═══════════════════════════════════════════════════════════════
def test_example2_custom_file():
    """Example 2: Custom CarvusTrain file parsing (02_custom_carvus_file.py)"""
    custom_file = os.path.join(PROJECT_ROOT, "_test_model.ct")
    with open(custom_file, "w", encoding="utf-8") as f:
        f.write("[CarvusTrain]\n\n[Model]\nName=CarvusExpert\n\n[Training]\n"
                "Method=normal\nDuration=forever\n\n[Knowledge]\n"
                "Carvus is an advanced artificial intelligence assistant.\n"
                "CarvusTrain provides modular layers, tokenizers, and exporters.\n")

    model = carvustrain.Model()
    model.train(data=custom_file, duration=2)

    ans = model.ask("What is Carvus?")
    assert isinstance(ans, str), f"Expected str, got {type(ans)}"
    print(f"  Answer: {ans[:120]}")

    if os.path.exists(custom_file):
        os.remove(custom_file)


# ═══════════════════════════════════════════════════════════════
# Example 3: Fine-tuning & Multi-Format Export
# ═══════════════════════════════════════════════════════════════
def test_example3_finetune_export():
    """Example 3: Fine-tuning and multi-format export (03_finetune_and_export.py)"""
    model = carvustrain.Model(name="TestCarvus")
    model.train(data=["Base knowledge about machine learning."], epochs=2)
    model.finetune(data=["Specialized knowledge on transformers."], epochs=2)

    formats = ["ct", "json", "bin", "onnx", "gguf"]
    for fmt in formats:
        out_file = os.path.join(PROJECT_ROOT, f"_test_export.{fmt}")
        model.export(out_file, format=fmt)
        assert os.path.exists(out_file), f"Export {fmt} failed!"
        size = os.path.getsize(out_file)
        assert size > 0, f"{fmt} export is empty!"
        print(f"  {fmt}: {size:>8} bytes")
        os.remove(out_file)


# ═══════════════════════════════════════════════════════════════
# Example 4: ChatModel
# ═══════════════════════════════════════════════════════════════
def test_example4_chat():
    """Example 4: ChatModel interactive conversation (04_chat_model_interactive.py)"""
    chat = carvustrain.ChatModel(name="TestChat")
    chat.learn([
        "Carvus is an AI assistant.",
        "CarvusTrain is an open-source Python deep learning library.",
    ])

    for user_input in ["Hello! Who are you?", "What is CarvusTrain?"]:
        reply = chat.chat(user_input)
        assert isinstance(reply, str), f"Expected str, got {type(reply)}"
        print(f"  Q: {user_input}")
        print(f"  A: {reply[:120]}")


# ═══════════════════════════════════════════════════════════════
# train.py Quick Start
# ═══════════════════════════════════════════════════════════════
def test_quick_start_demo():
    """Quick start demo from train.py"""
    bot = carvustrain.ChatModel(name="CarvusDemo")
    count = bot.knowledge_base.load_grammar_knowledge()
    assert count > 0, "No grammar facts loaded!"
    print(f"  Grammar facts loaded: {count}")

    bot.learn("Carvus is an AI assistant built with CarvusTrain.")
    bot.learn("CarvusTrain is an AI development ecosystem.")
    response = bot.chat("Who are you?")
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    print(f"  Chat response: {response[:120]}")
    print(f"  Total facts: {len(bot.knowledge_base)}")


# ═══════════════════════════════════════════════════════════════
# CLI Tests
# ═══════════════════════════════════════════════════════════════
def test_cli_help():
    """CLI --help display"""
    from CarvusTrain.cli import main
    try:
        main(["--help"])
    except SystemExit:
        pass  # argparse exits with 0 on --help
    print("  CLI help OK")


def test_cli_version():
    """CLI version command"""
    from CarvusTrain.cli import main
    try:
        main(["version"])
    except SystemExit:
        pass
    print("  CLI version OK")


def test_cli_doctor():
    """CLI doctor command"""
    from CarvusTrain.cli import main
    try:
        main(["doctor"])
    except SystemExit:
        pass
    print("  CLI doctor OK")


def test_cli_create():
    """CLI create command"""
    from CarvusTrain.cli import main
    try:
        main(["create", "_test_cli_model", "--architecture", "transformer"])
    except SystemExit:
        pass

    model_dir = os.path.join(PROJECT_ROOT, "_test_cli_model")
    assert os.path.exists(model_dir), f"Directory not created: {model_dir}"
    ct_file = os.path.join(model_dir, "train.ct")
    assert os.path.exists(ct_file), f"train.ct not created in {model_dir}"
    print(f"  Created model in: {model_dir}")

    # Cleanup
    import shutil
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)


# ═══════════════════════════════════════════════════════════════
# Knowledge Base Tests
# ═══════════════════════════════════════════════════════════════
def test_knowledge_search():
    """Knowledge base semantic search"""
    from CarvusTrain.memory import KnowledgeBase

    kb = KnowledgeBase()
    kb.add_facts([
        "Python is a programming language used for web development and AI.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers are a neural network architecture for NLP.",
    ])

    results = kb.search("programming", top_k=2)
    assert len(results) > 0, "Search returned no results!"
    best_text, best_score = results[0]
    print(f"  Best match: '{best_text[:70]}...' score={best_score:.4f}")

    # Language-specific search
    lang_results = kb.search_by_language("python", "python", top_k=1)
    print(f"  Language search: {len(lang_results)} results")


def test_grammar():
    """Grammar knowledge base"""
    from CarvusTrain.memory import KnowledgeBase

    kb = KnowledgeBase()
    count = kb.load_grammar_knowledge()
    assert count > 50, f"Expected >50 grammar facts, got {count}"
    print(f"  Grammar facts: {count}")

    results = kb.search_grammar("verb tense", topic="tenses", top_k=2)
    assert len(results) > 0, "Grammar search failed!"
    print(f"  Grammar search: {len(results)} results")


# ═══════════════════════════════════════════════════════════════
# Tokenizer Tests
# ═══════════════════════════════════════════════════════════════
def test_tokenizers():
    """All tokenizer types"""
    from CarvusTrain.tokenizer import Tokenizer

    for tok_type in ["word", "char", "sentence", "bpe", "wordpiece"]:
        tok = Tokenizer.create(tok_type)
        tok.train_on_texts(["Hello world! This is a test of tokenization."])
        tokens = tok.encode("Hello world!")
        decoded = tok.decode(tokens)
        assert len(tokens) > 0, f"{tok_type}: no tokens!"
        assert len(decoded) > 0, f"{tok_type}: empty decode!"
        print(f"  {tok_type}: {len(tokens)} tokens -> '{decoded}'")


# ═══════════════════════════════════════════════════════════════
# Configuration Tests
# ═══════════════════════════════════════════════════════════════
def test_config():
    """Configuration dataclasses"""
    from CarvusTrain.configuration import (
        CarvusConfig, ModelConfig, TrainingConfig, InferenceConfig, load_config_file
    )

    cfg = CarvusConfig()
    assert cfg.model.vocab_size == 10000
    assert cfg.training.epochs == 10
    assert cfg.inference.temperature == 0.7
    assert cfg.inference.top_k == 50
    print(f"  Default config: {cfg.model.name}, {cfg.model.model_type}")

    # Test JSON config loading
    import json
    cfg_path = os.path.join(PROJECT_ROOT, "_test_config.json")
    with open(cfg_path, "w") as f:
        json.dump({"model": {"name": "CustomAI", "num_layers": 8}}, f)
    loaded = load_config_file(cfg_path)
    assert loaded.model.name == "CustomAI"
    assert loaded.model.num_layers == 8
    print(f"  JSON config loaded: {loaded.model.name}, {loaded.model.num_layers} layers")
    os.remove(cfg_path)


# ═══════════════════════════════════════════════════════════════
# Dataset Tests
# ═══════════════════════════════════════════════════════════════
def test_datasets():
    """Dataset and DataLoader"""
    from CarvusTrain.dataset import Dataset, TextDataset, DataLoader

    ds = Dataset.from_texts(["a", "b", "c", "d", "e"])
    assert len(ds) == 5
    train, val = ds.split(val_ratio=0.2, seed=42)
    assert len(train) + len(val) == 5
    print(f"  Dataset: {len(train)} train, {len(val)} val")

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert len(batches[0]) == 2
    print(f"  DataLoader: {len(batches)} batches")

    td = TextDataset.from_texts(["hello world"])
    texts = td.get_all_texts()
    assert len(texts) == 1
    print(f"  TextDataset: {len(texts)} texts")


# ═══════════════════════════════════════════════════════════════
# Optimizer Tests
# ═══════════════════════════════════════════════════════════════
def test_optimizers():
    """Optimizer creation and state"""
    from CarvusTrain.optimizer import Adam, AdamW, SGD, RMSprop, AdaGrad

    for opt_cls in [Adam, AdamW, SGD, RMSprop, AdaGrad]:
        opt = opt_cls(lr=0.001)
        assert opt.lr == 0.001
        state = opt.state_dict()
        assert "lr" in state
        opt.load_state_dict(state)
        print(f"  {opt_cls.__name__}: OK")


# ═══════════════════════════════════════════════════════════════
# Save/Load Cycle
# ═══════════════════════════════════════════════════════════════
def test_save_load():
    """Model save and load"""
    model = carvustrain.Model(name="SaveLoadTest")
    model.learn("CarvusTrain is a great AI framework.")
    save_path = os.path.join(PROJECT_ROOT, "_test_sl.ct")
    model.save(save_path)
    assert os.path.exists(save_path)

    model2 = carvustrain.Model(name="Reloaded")
    model2.load(save_path)
    assert len(model2.knowledge_base) >= 1
    print(f"  Save/load: {len(model2.knowledge_base)} facts restored")

    if os.path.exists(save_path):
        os.remove(save_path)


# ═══════════════════════════════════════════════════════════════
# Preprocessing / Postprocessing
# ═══════════════════════════════════════════════════════════════
def test_preprocessing():
    """Text preprocessing"""
    from CarvusTrain.preprocessing import TextPreprocessor, pad_sequence, chunk_text

    pre = TextPreprocessor(lowercase=True, remove_html_tags=True)
    assert pre.clean("<p>Hello World!</p>") == "hello world!"
    print(f"  Clean: OK")

    padded = pad_sequence([1, 2, 3], max_length=5, pad_value=0)
    assert padded == [1, 2, 3, 0, 0]
    print(f"  Padding: {padded}")

    chunks = chunk_text("word " * 100, chunk_size=20, overlap=5)
    assert len(chunks) > 3
    print(f"  Chunking: {len(chunks)} chunks")


def test_postprocessing():
    """Logit and text postprocessing"""
    from CarvusTrain.postprocessing import LogitProcessor, TextPostprocessor

    logits = [1.0, 2.0, 3.0, 0.5, -1.0]
    scaled = LogitProcessor.apply_temperature(logits, 0.5)
    assert len(scaled) == len(logits)
    print(f"  Temperature scaling: OK")

    token = LogitProcessor.sample_top_k_top_p(logits, top_k=3, top_p=0.9)
    assert 0 <= token < len(logits)
    print(f"  Sampled: token {token}")

    post = TextPostprocessor()
    assert post.process("Hello<eos>world") == "Hello"
    print(f"  Stop sequence: OK")


# ═══════════════════════════════════════════════════════════════
# Agent Model
# ═══════════════════════════════════════════════════════════════
def test_agent():
    """AgentModel and sub-agent orchestration"""
    agent = carvustrain.AgentModel(name="TestAgent", goal="software engineer")
    assert agent.goal == "software engineer"
    print(f"  Agent: {agent.name}, goal={agent.goal}")

    sub = carvustrain.AgentModel(name="SubAgent")
    agent.add_sub_agent("planner", sub)
    results = agent.orchestrate("Test task")
    assert "planner" in results
    assert "executor" in results
    print(f"  Orchestration: {list(results.keys())}")


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════
def test_evaluation():
    """Model evaluation and benchmarking"""
    model = carvustrain.Model(name="EvalTest")
    model.learn("Python is a programming language.")
    model.learn("Machine learning is AI.")

    eval_results = model.evaluate()
    assert "accuracy" in eval_results
    assert "learning_accuracy" in eval_results
    print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
    print(f"  Learning: {eval_results.get('learning_accuracy', 0):.4f}")

    bench = model.benchmark("Who are you?", num_runs=3)
    assert "avg_latency_ms" in bench
    assert bench["num_runs"] == 3
    print(f"  Benchmark: {bench['avg_latency_ms']:.2f}ms avg")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  CarvusTrain — End-to-End Test Suite")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Import: {'source' if USING_SOURCE else 'installed'}")

    # Run all tests
    run_test("Basic Usage (Example 1)", test_example1_basic_usage)
    run_test("Custom .ct File (Example 2)", test_example2_custom_file)
    run_test("Fine-tune & Export (Example 3)", test_example3_finetune_export)
    run_test("ChatModel (Example 4)", test_example4_chat)
    run_test("Quick Start Demo (train.py)", test_quick_start_demo)
    run_test("CLI --help", test_cli_help)
    run_test("CLI version", test_cli_version)
    run_test("CLI doctor", test_cli_doctor)
    run_test("CLI create", test_cli_create)
    run_test("Knowledge Base Search", test_knowledge_search)
    run_test("Grammar Knowledge", test_grammar)
    run_test("Tokenizers", test_tokenizers)
    run_test("Configuration", test_config)
    run_test("Dataset & DataLoader", test_datasets)
    run_test("Optimizers", test_optimizers)
    run_test("Save/Load Cycle", test_save_load)
    run_test("Preprocessing", test_preprocessing)
    run_test("Postprocessing", test_postprocessing)
    run_test("AgentModel", test_agent)
    run_test("Evaluation & Benchmark", test_evaluation)

    # Summary
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'='*60}")

    if FAIL > 0:
        print(f"\n  ❌ {FAIL} test(s) failed. See errors above.")
        sys.exit(1)
    else:
        print("\n  🎉 All tests passed! CarvusTrain is working correctly.")
        print(f"  {PASS} tests covering all examples and features.")
