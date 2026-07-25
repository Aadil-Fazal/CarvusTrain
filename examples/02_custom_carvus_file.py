"""Custom CarvusTrain file format example parsing sectioned knowledge files."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import carvustrain
except ImportError:
    import CarvusTrain as carvustrain


def main():
    print("=== Custom CarvusTrain File Parsing Example ===")

    custom_file = "model_spec.ct"
    with open(custom_file, "w", encoding="utf-8") as f:
        f.write("""[CarvusTrain]

[Model]
Name=CarvusExpert

[Training]
Method=normal
Duration=forever
Shuffle=random
Output=output.txt

[Knowledge]
Carvus is an advanced artificial intelligence assistant designed from scratch.
CarvusTrain provides modular layers, tokenizers, optimizers, and exporters.
Deep learning models learn features directly from structured and unstructured data.
""")

    # Initialize model and train directly passing custom section file
    model = carvustrain.Model()
    model.train(data=custom_file, duration=2)

    # Query knowledge base
    print("\n--- Question Answering ---")
    ans = model.ask("What is Carvus?")
    print(f"Answer: {ans}")

    # Clean up
    if os.path.exists(custom_file):
        os.remove(custom_file)


if __name__ == "__main__":
    main()
