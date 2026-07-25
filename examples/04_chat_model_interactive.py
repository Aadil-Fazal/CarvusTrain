"""ChatModel example demonstrating multi-turn interactive conversations."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import carvustrain
except ImportError:
    import CarvusTrain as carvustrain


def main():
    print("=== ChatModel Multi-Turn Interactive Example ===")

    chat_model = carvustrain.ChatModel(name="CarvusAssistant")
    chat_model.learn([
        "Carvus is an AI assistant.",
        "CarvusTrain is an open-source Python deep learning library.",
    ])

    prompts = [
        "Hello! Who are you?",
        "What is CarvusTrain?",
        "Thank you!",
    ]

    for user_input in prompts:
        print(f"\nUser: {user_input}")
        reply = chat_model.chat(user_input)
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
