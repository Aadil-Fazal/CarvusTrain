import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect(root: Path):
    if not (root / "project.json").is_file():
        raise ValueError(f"No coding workspace at {root}")
    db = sqlite3.connect(root / "workspace.sqlite3")
    db.row_factory = sqlite3.Row
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, config TEXT NOT NULL, fingerprint TEXT NOT NULL,
                state TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
                cancel INTEGER NOT NULL DEFAULT 0, error TEXT, parameters INTEGER
            )""")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS active_job ON jobs(state) WHERE state = 'running'")
            yield db
    finally:
        db.close()



@contextmanager
def training_lock(root: Path):
    # A separate database lets monitoring and cancellation continue during training.
    if not (root / "project.json").is_file():
        raise ValueError(f"No coding workspace at {root}")
    lock = sqlite3.connect(root / "training.lock.sqlite3", timeout=0)
    try:
        try:
            lock.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                raise ValueError("A training job is already running in this workspace") from exc
            raise
        with connect(root) as db:
            db.execute("UPDATE jobs SET state='failed', error='Training process exited unexpectedly' "
                       "WHERE state='running'")
        yield
    finally:
        lock.close()
