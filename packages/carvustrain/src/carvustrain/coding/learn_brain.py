import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable
import shutil
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .database import connect, training_lock
from .model import CodingModel, weight_size
from .module_call import ModuleCall, Project
from .schemas import TrainingConfig

logger = logging.getLogger(__name__)


class LearningBrain:
    def __init__(self, root: Path):
        self.root = root

    def validate(self, extra: bool = False) -> list[dict[str, str]]:
        return ModuleCall.load(self.root).datasets(self.root, extra)

    def status(self, job_id: str | None = None):
        with connect(self.root) as db:
            if job_id is None:
                return [self._describe(row) for row in db.execute("SELECT * FROM jobs ORDER BY rowid DESC")]
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown job {job_id}")
            return self._describe(row)

    def _describe(self, row) -> dict:
        result = dict(row)
        result["config"] = json.loads(row["config"])
        result["checkpoint"] = str(self._checkpoint(row["id"], row["step"])) if row["step"] else None
        result["log"] = str(self.root / "jobs" / row["id"] / "training.jsonl")
        return result

    def _checkpoint(self, job_id: str, step: int) -> Path:
        return self.root / "jobs" / job_id / f"step-{step}"

    def checkpoint(self, job_id: str) -> Path:
        job = self.status(job_id)
        if not job["checkpoint"]:
            raise ValueError(f"Job {job_id} has no saved training step")
        return Path(job["checkpoint"])

    def cancel(self, job_id: str) -> bool:
        with connect(self.root) as db:
            return bool(db.execute(
                "UPDATE jobs SET cancel = 1 WHERE id = ? AND state = 'running'", (job_id,),
            ).rowcount)

    def train(
        self, config: TrainingConfig | None = None, *, resume: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        with training_lock(self.root):
            return self._train(config, resume, on_progress)

    def _train(
        self, config: TrainingConfig | None, resume: str | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> dict:
        previous = self.status(resume) if resume else None
        if previous and previous["state"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled jobs can be resumed")
        config = TrainingConfig(**previous["config"]) if previous else config or TrainingConfig()
        config.validate()
        project = Project.load(self.root)
        rows = self.validate(config.extra)
        base = project.source(self.root)
        files = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(base.iterdir()) if p.is_file()]
        payload = json.dumps([asdict(project), str(base), files, rows], sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        job_id = resume or uuid4().hex
        completed = previous["step"] if previous else 0
        try:
            with connect(self.root) as db:
                if previous:
                    if previous["fingerprint"] != fingerprint:
                        raise ValueError("Project, base model or dataset changed; start a new job")
                    updated = db.execute(
                        "UPDATE jobs SET state='running', error=NULL, cancel=0 "
                        "WHERE id=? AND state IN ('failed','cancelled') AND step=?", (job_id, completed),
                    )
                    if not updated.rowcount:
                        raise ValueError("Job changed while requesting resume")
                else:
                    db.execute("INSERT INTO jobs(id, config, fingerprint, state) VALUES (?, ?, ?, 'running')",
                               (job_id, json.dumps(asdict(config)), fingerprint))
        except sqlite3.IntegrityError as exc:
            raise ValueError("A training job is already running in this workspace") from exc

        try:
            source = self._checkpoint(job_id, completed) if completed else base
            size = weight_size(source)
            if shutil.disk_usage(self.root).free < 2 * size * (config.steps - completed + 1):
                raise ValueError("Insufficient checkpoint disk space; use fewer steps or a smaller base")
            model = CodingModel(source, project.device, project.max_tokens, training=True)
            model.start_training(config.learning_rate)
            folder = self.root / "jobs" / job_id
            folder.mkdir(parents=True, exist_ok=True)
            for step in range(completed + 1, config.steps + 1):
                if self.status(job_id)["cancel"]:
                    self._finish(job_id, "cancelled")
                    break
                destination = self._checkpoint(job_id, step)
                if not destination.exists():
                    loss = model.train_step(rows[(step - 1) % len(rows)])
                    model.save(destination)
                    with (folder / "training.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": step, "loss": loss}) + "\n")
                else:
                    # A prior process published this step before recording progress.
                    model = CodingModel(destination, project.device, project.max_tokens, training=True)
                    model.start_training(config.learning_rate)
                with connect(self.root) as db:
                    db.execute("UPDATE jobs SET step=?, parameters=? WHERE id=?",
                               (step, model.parameter_count, job_id))
                if on_progress:
                    on_progress(step, config.steps)
            else:
                self._finish(job_id, "completed")
        except KeyboardInterrupt:
            self._finish(job_id, "cancelled")
        except Exception as exc:
            logger.exception("Coding training job %s failed", job_id)
            self._finish(job_id, "failed", str(exc))
        return self.status(job_id)

    def _finish(self, job_id: str, state: str, error: str | None = None):
        with connect(self.root) as db:
            db.execute("UPDATE jobs SET state=?, error=? WHERE id=?", (state, error, job_id))
