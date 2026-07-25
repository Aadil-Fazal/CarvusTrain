"""Quick verification script to test the import fixes."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CarvusTrain Import Verification")
print("=" * 60)

# Test 1: Basic package import
print("\n1. Testing: import carvustrain")
try:
    import carvustrain
    print(f"   ✅ OK - {carvustrain.__file__}")
except Exception as e:
    print(f"   ❌ FAIL - {e}")

# Test 2: Star import (all public API)
print("\n2. Testing: from carvustrain import *")
try:
    exec("from carvustrain import *")
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAIL - {e}")

# Test 3: Key classes
print("\n3. Testing: Key class imports")
classes_to_test = [
    "Model", "ChatModel", "TextModel", "LanguageModel",
    "Trainer", "Dataset", "Tokenizer", "DataParser",
    "CarvusTrainParser", "ModelExporter", "Evaluator",
    "CarvusConfig", "Adam", "AdamW",
]
for cls_name in classes_to_test:
    try:
        obj = getattr(carvustrain, cls_name, None)
        if obj:
            print(f"   ✅ {cls_name}")
        else:
            print(f"   ❌ {cls_name} - not found")
    except Exception as e:
        print(f"   ❌ {cls_name} - {e}")

# Test 4: CLI entry point
print("\n4. Testing: CLI module import")
try:
    from carvustrain.cli import main
    print("   ✅ OK - cli.main is importable")
except Exception as e:
    print(f"   ❌ FAIL - {e}")

# Test 5: Sub-modules
print("\n5. Testing: Sub-module imports")
sub_modules = [
    "configuration", "dataset", "evaluation", "exceptions",
    "exporter", "inference", "memory", "model", "optimizer",
    "parser", "scheduler", "tokenizer", "trainer",
    "callbacks", "layers", "losses", "metrics", "activation",
    "logger", "utils", "version", "constants",
]
for mod in sub_modules:
    try:
        __import__(f"carvustrain.{mod}")
        print(f"   ✅ carvustrain.{mod}")
    except Exception as e:
        print(f"   ❌ carvustrain.{mod} - {e}")

print("\n" + "=" * 60)
print("Verification complete!")
print("=" * 60)
