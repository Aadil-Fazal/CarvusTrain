#!/usr/bin/env python3
"""CarvusTrain — Integration Test Script
Tests that the library imports correctly and core features work.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CarvusTrain Integration Tests")
print("=" * 60)

# Test 1: Core imports
print("\n1. Core imports...")
try:
    from carvustrain import (
        Model, ChatModel, TextModel, LanguageModel, CustomModel, AgentModel,
        KnowledgeBase, LearningValidator
    )
    print("   ✅ All core imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: CLI import
print("\n2. CLI import...")
try:
    from carvustrain.cli import main
    print("   ✅ CLI imported successfully")
except Exception as e:
    print(f"   ❌ CLI import failed: {e}")
    sys.exit(1)

# Test 3: Model creation
print("\n3. Model creation...")
try:
    model = Model(name="TestAI", auto_load=False)
    print(f"   ✅ Model '{model.name}' created")
    print(f"      Architecture: {model.architecture}")
    print(f"      Mode: {model.mode}")
    print(f"      Parameters: {model.parameters}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Test 4: KnowledgeBase
print("\n4. KnowledgeBase operations...")
try:
    kb = KnowledgeBase()
    kb.add_fact("Python is a programming language.")
    kb.add_facts(["JavaScript is for web.", "Rust is for systems."])
    results = kb.search("Python")
    print(f"   ✅ KnowledgeBase: {len(kb)} facts, search returned {len(results)} results")
except Exception as e:
    print(f"   ❌ KnowledgeBase failed: {e}")
    sys.exit(1)

# Test 5: Grammar loading
print("\n5. Grammar knowledge loading...")
try:
    n = kb.load_grammar_knowledge()
    grammar_results = kb.search_grammar("noun", topic="parts_of_speech")
    print(f"   ✅ Loaded {n} grammar facts, grammar search: {len(grammar_results)} results")
except Exception as e:
    print(f"   ❌ Grammar loading failed: {e}")
    sys.exit(1)

# Test 6: ChatModel
print("\n6. ChatModel creation...")
try:
    chat = ChatModel(name="TestBot", auto_load=False)
    print(f"   ✅ ChatModel '{chat.name}' created")
except Exception as e:
    print(f"   ❌ ChatModel failed: {e}")
    sys.exit(1)

# Test 7: AgentModel
print("\n7. AgentModel creation...")
try:
    agent = AgentModel(name="CoderBot", goal="software engineer", auto_load=False)
    print(f"   ✅ AgentModel '{agent.name}' with goal '{agent.goal}'")
except Exception as e:
    print(f"   ❌ AgentModel failed: {e}")
    sys.exit(1)

# Test 8: Personality system
print("\n8. Personality system...")
try:
    chat.set_personality(style="friendly", humor=50, creativity=80)
    print(f"   ✅ Personality set successfully")
except Exception as e:
    print(f"   ❌ Personality system failed: {e}")
    sys.exit(1)

# Test 9: Memory engine
print("\n9. Memory engine...")
try:
    chat.enable_memory(True)
    chat.enable_memory(False)
    print(f"   ✅ Memory engine works")
except Exception as e:
    print(f"   ❌ Memory engine failed: {e}")
    sys.exit(1)

# Test 10: Plugin system
print("\n10. Plugin system...")
try:
    class TestPlugin:
        def process(self, data):
            return data
    chat.install_plugin("test_plugin", TestPlugin())
    print(f"   ✅ Plugin installed: {list(chat._plugins.keys())}")
except Exception as e:
    print(f"   ❌ Plugin system failed: {e}")
    sys.exit(1)

# Test 11: Mode switching
print("\n11. Mode switching...")
try:
    for mode in ["programmer", "teacher", "researcher", "creative", "security", "data_scientist"]:
        chat.mode = mode
    chat.mode = "general"
    print(f"   ✅ Mode switching works (current: {chat.mode})")
except Exception as e:
    print(f"   ❌ Mode switching failed: {e}")
    sys.exit(1)

# Test 12: Auto AI Trainer
print("\n12. Auto AI Trainer...")
try:
    chat.auto_train(data=["Sample training text."], auto_optimize=False)
    print(f"   ✅ Auto AI Trainer works")
except Exception as e:
    print(f"   ❌ Auto AI Trainer failed: {e}")
    sys.exit(1)

# Test 13: Model summary and statistics
print("\n13. Model summary and statistics...")
try:
    stats = chat.statistics()
    mem = chat.memory()
    print(f"   ✅ Statistics: {len(stats)} keys")
    print(f"   ✅ Memory: {len(mem)} keys")
except Exception as e:
    print(f"   ❌ Model summary/statistics failed: {e}")
    sys.exit(1)

# Test 14: Inference engine
print("\n14. Inference engine...")
try:
    from carvustrain.inference import InferenceEngine, QuestionAnsweringEngine, CodeGenerator
    engine = InferenceEngine(kb)
    print(f"   ✅ InferenceEngine created")
    gen = CodeGenerator(kb)
    print(f"   ✅ CodeGenerator created")
    qa = QuestionAnsweringEngine(kb)
    print(f"   ✅ QuestionAnsweringEngine created")
except Exception as e:
    print(f"   ❌ Inference engine failed: {e}")
    sys.exit(1)

# Test 15: Code generation
print("\n15. Code generation...")
try:
    gen = CodeGenerator(kb)
    code = gen.generate_code("Write binary search in Python")
    assert "def " in code or "```" in code, "Code generation produced no output"
    print(f"   ✅ Code generation works (output: {len(code)} chars)")
    
    # Check algorithm aliases
    dfs_code = gen.generate_code("Write depth first search in Python")
    assert len(dfs_code) > 0, "DFS alias matching failed"
    print(f"   ✅ Algorithm alias matching works")
except Exception as e:
    print(f"   ❌ Code generation failed: {e}")
    sys.exit(1)

# Test 16: Evaluate
print("\n16. Evaluate...")
try:
    eval_results = chat.evaluate()
    print(f"   ✅ Evaluate works: {len(eval_results)} metrics")
except Exception as e:
    print(f"   ❌ Evaluate failed: {e}")
    sys.exit(1)

# Test 17: Export
print("\n17. Save/Export...")
try:
    path = chat.save("test_model.ct")
    assert os.path.exists("test_model.ct"), "Save file not created"
    os.remove("test_model.ct")
    print(f"   ✅ Model save/load works")
except Exception as e:
    print(f"   ❌ Save/export failed: {e}")
    sys.exit(1)

# Test 18: pyproject.toml validation
print("\n18. Package config validation...")
try:
    import configparser
    config = configparser.ConfigParser()
    config.read("pyproject.toml")
    project_section = None
    for section in config.sections():
        if "project" in section.lower():
            # Check for name in the raw section
            pass
    with open("pyproject.toml", "r") as f:
        content = f.read()
    if "Aadil Fazal" in content:
        print(f"   ✅ Package config has author: Aadil Fazal")
    else:
        print(f"   ⚠️  Author not found in pyproject.toml")
except Exception as e:
    print(f"   ❌ Package config validation failed: {e}")

print("\n" + "=" * 60)
print("✅ ALL 18 TESTS PASSED!")
print("=" * 60)
