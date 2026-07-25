"""Basic usage example demonstrating CarvusTrain public API syntax."""

import os
import sys

# Allow running from source without pip install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import carvustrain
except ImportError:
    import CarvusTrain as carvustrain


def main():
    print("=== CarvusTrain Basic Usage Example ===")

    # 1. Initialize Model
    model = carvustrain.Model(name="Carvus")

    # Create dummy training data file
    data_file = "train.txt"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write("Carvus is an advanced artificial intelligence assistant built with CarvusTrain.\n")
        f.write("CarvusTrain simplifies deep learning model training with clean Python syntax.\n")
        f.write("Artificial intelligence is revolutionizing modern technology.\n")

    # 2. Train Model
    print("\n--- Training Model ---")
    model.train(data=data_file, method="normal", duration=3)

    # 3. Save Model
    model_path = "carvus.ct"
    model.save(model_path)
    print(f"\nSaved model to '{model_path}'")

    # 4. Perform Q&A Inference
    print("\n--- Model Inference ---")
    answer = model.ask("Who are you?")
    print(f"Question: Who are you?")
    print(f"Answer: {model.answer}")

    # 5. Model Summary & Memory Stats
    print("\n--- Model Summary ---")
    model.summary()
    print("\n--- Model Memory Stats ---")
    print(model.memory())

    # Cleanup temporary example files
    if os.path.exists(data_file):
        os.remove(data_file)
    if os.path.exists(model_path):
        os.remove(model_path)


if __name__ == "__main__":
    main()
