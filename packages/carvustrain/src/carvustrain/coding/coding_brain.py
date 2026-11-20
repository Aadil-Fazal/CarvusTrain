import re

from .model import CodingModel


def generate(model: CodingModel, question: str, memory: list[dict]) -> tuple[str, str | None]:
    context = "\n".join(row["text"] for row in memory)
    prompt = f"Reference notes:\n{context}\n\n{question}" if context else question
    answer = model.answer(prompt)
    if not answer:
        raise ValueError("The model produced an empty answer")
    match = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    return answer, match.group(1) if match else None
