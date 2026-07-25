"""Quick validation of learning check, programming language detection, and inference."""
import sys
sys.path.insert(0, '.')

# Test 1: Import and create model
import carvustrain
m = carvustrain.Model(auto_load=False)
assert m.name == "Carvus", f"Expected Carvus, got {m.name}"
assert hasattr(m, 'learning_validator'), "Missing learning_validator"
print(f"[PASS] Model created: {m.name}, Validator: {type(m.learning_validator).__name__}")

# Test 2: Load train.ct knowledge
m2 = carvustrain.Model(auto_load=True)
print(f"[INFO] Loaded {len(m2.knowledge_base)} facts from train.ct")
langs = m2.knowledge_base.get_programming_languages()
print(f"[INFO] Detected languages: {langs}")

# Test 3: Learn programming facts
m.learn("Python is a high-level, interpreted programming language known for readability.")
assert len(m.knowledge_base) > 0, "Knowledge base empty after learn()"
print(f"[PASS] Knowledge facts: {len(m.knowledge_base)}")
print(f"[PASS] Languages detected: {m.knowledge_base.get_programming_languages()}")

# Test 4: Language detection
lang = m.knowledge_base._detect_language("def hello():\n    print('hello world')")
assert lang == "python", f"Expected python, got {lang}"
print(f"[PASS] Python code detection: {lang}")

# Test 5: Inference with code question
ans = m.ask("What is Python programming language?")
assert len(ans) > 0, "Empty answer"
print(f"[PASS] Code answer: {ans[:80]}...")

# Test 6: Learning validation
from carvustrain.memory import LearningValidator
validator = LearningValidator(m.knowledge_base)
metrics = validator.validate_learning(
    ["Python is a programming language.", "def add(a, b): return a + b"],
    ["Python is used for programming. Functions are defined with def."]
)
assert "accuracy" in metrics, "Missing accuracy metric"
print(f"[PASS] Learning validation - accuracy: {metrics['accuracy']:.4f}")

# Test 7: Learning status
status = m.knowledge_base.get_learning_status()
assert "status" in status, "Missing status"
print(f"[PASS] Learning status: {status['status']}, languages: {status['code_languages']}")

# Test 8: Search by language
m2.knowledge_base.search_by_language("function definition", "python", top_k=2)
print(f"[PASS] Language-specific search works")

# Test 9: Training with learning check
history = m2.train(data=["Python is a programming language.", "JavaScript is for web."], epochs=2, batch_size=2)
assert "loss" in history, "Missing loss in history"
assert "learning_accuracy" in history, "Missing learning_accuracy in history"
print(f"[PASS] Training with LearnCheck: loss={history['loss'][-1]:.4f}, acc={history['learning_accuracy'][-1]:.4f}")

# Test 10: Model summary with code info
m.summary()
print(f"[PASS] Model summary works with code languages")

print("\n=== ALL VALIDATION TESTS PASSED ===")
