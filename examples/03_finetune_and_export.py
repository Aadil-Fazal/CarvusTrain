"""Fine-tuning and multi-format exporting example (.ct, .json, .onnx, .gguf)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import carvustrain
except ImportError:
    import CarvusTrain as carvustrain


def main():
    print("=== Fine-Tuning and Exporting Example ===")

    # Initial training
    model = carvustrain.Model(name="BaseCarvus")
    model.train(data=["Base knowledge about machine learning."], epochs=2)

    # Fine-tuning on new domain facts
    print("\n--- Fine-Tuning ---")
    model.finetune(data=["Specialized knowledge on transformer neural network architectures."], epochs=3)

    # Export to different formats
    print("\n--- Exporting Model ---")
    formats = ["ct", "json", "bin", "onnx", "gguf"]
    for fmt in formats:
        out_file = f"exported_model.{fmt}"
        model.export(out_file, format=fmt)
        print(f"Exported to format '{fmt}': {out_file}")
        if os.path.exists(out_file):
            os.remove(out_file)


if __name__ == "__main__":
    main()
