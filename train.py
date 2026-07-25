"""CarvusTrain — Quick Start Demo
Run: python train.py
"""
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# 1. Make sure the package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Option A: allow running straight from the source tree
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from carvustrain import Model
except ImportError:
    try:
        # Option B: case-sensitive filesystem fallback
        from CarvusTrain import Model  # type: ignore
    except ImportError:
        # Option C: package hasn't been installed at all — do an editable install
        print("CarvusTrain not found. Installing from source...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", PROJECT_ROOT],
            )
        except subprocess.CalledProcessError:
            print("\n[ERROR] Auto-install failed. Please run manually:")
            print(f"  {sys.executable} -m pip install -e {PROJECT_ROOT}")
            sys.exit(1)
        print("Installation complete.\n")
        from carvustrain import Model


# ---------------------------------------------------------------------------
# 2. Demo
# ---------------------------------------------------------------------------
bot = Model(name="CarvusDemo")
bot.knowledge_base.load_grammar_knowledge()
bot.learn("Carvus is an AI assistant built with CarvusTrain.")
bot.learn("CarvusTrain is an AI development ecosystem.")
response = bot.chat("Who are you?")
print(f"\nCarvus: {response}")
print(f"Facts loaded: {len(bot.knowledge_base)}")
