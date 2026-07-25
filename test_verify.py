"""Quick verification script for CarvusTrain imports and basic functionality."""
import sys
sys.path.insert(0, '.')

try:
    import carvustrain
    print(f"✓ carvustrain imported successfully")
    
    m = carvustrain.Model(auto_load=False)
    print(f"✓ Model created: {m.name}")
    
    m = carvustrain.ChatModel(auto_load=False)
    print(f"✓ ChatModel created: {m.name}")
    
    from carvustrain.inference import CodeGenerator
    print(f"✓ CodeGenerator imported")
    
    # Verify new algorithm templates exist
    cg = CodeGenerator.__new__(CodeGenerator)
    langs = ["python", "javascript", "rust", "go"]
    for lang in langs:
        templates = CodeGenerator.ALGORITHM_TEMPLATES.get(lang, {})
        n = len(templates)
        names = list(templates.keys())[:3]
        print(f"✓ {lang}: {n} algorithm templates ({', '.join(names)}...)")
    
    # Verify _ALGO_ALIASES
    n_aliases = len(CodeGenerator._ALGO_ALIASES)
    print(f"✓ {n_aliases} algorithm aliases defined")
    
    # Verify English grammar in memory
    from carvustrain.memory import ENGLISH_GRAMMAR
    cats = list(ENGLISH_GRAMMAR.keys())
    print(f"✓ ENGLISH_GRAMMAR with {len(cats)} categories: {', '.join(cats)}")
    total_entries = sum(len(v) for v in ENGLISH_GRAMMAR.values())
    print(f"✓ {total_entries} total grammar entries")
    
    # Verify inference
    from carvustrain.inference import QuestionAnsweringEngine, TextGenerator, InferenceEngine
    print(f"✓ All inference engines imported")
    
    from carvustrain.memory import KnowledgeBase, LearningValidator
    print(f"✓ Memory components imported")
    
    print("\n✓ ALL VERIFICATIONS PASSED")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
