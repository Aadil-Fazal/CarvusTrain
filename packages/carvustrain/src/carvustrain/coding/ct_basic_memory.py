import re
from pathlib import Path

from .database import connect


class BasicMemory:
    def __init__(self, root: Path):
        self.root = root

    def add(self, text: str) -> int:
        if not text.strip():
            raise ValueError("Memory text is empty")
        with connect(self.root) as db:
            return db.execute("INSERT INTO memory(text) VALUES (?)", (text,)).lastrowid

    def retrieve(self, entry_id: int) -> dict | None:
        with connect(self.root) as db:
            row = db.execute("SELECT * FROM memory WHERE id = ?", (entry_id,)).fetchone()
            return dict(row) if row else None

    def search(self, query: str, limit: int = 3) -> list[dict]:
        words = set(re.findall(r"\w+", query.casefold()))
        if not words or limit < 1:
            return []
        matches = []
        with connect(self.root) as db:
            for row in db.execute("SELECT * FROM memory"):
                score = len(words.intersection(re.findall(r"\w+", row["text"].casefold())))
                if score:
                    matches.append((score, row["id"], dict(row)))
        return [item[2] for item in sorted(matches, reverse=True)[:limit]]

    def delete(self, entry_id: int) -> bool:
        with connect(self.root) as db:
            return bool(db.execute("DELETE FROM memory WHERE id = ?", (entry_id,)).rowcount)
