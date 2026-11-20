import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleCall:
    basic_training: str = "local"
    basic_data: str = "data.cl"
    more_training: str | None = "train.cl"
    call_code: str = "model.py"
    learn: str = "learn_brain"

    @classmethod
    def load(cls, root: Path) -> "ModuleCall":
        wiring = BrainWiring.load(root)
        path = root / wiring.learning
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid learning configuration {path}: {exc}") from exc

    def datasets(self, root: Path, extra: bool) -> list[dict[str, str]]:
        if (self.basic_training, self.call_code, self.learn) != ("local", "model.py", "learn_brain"):
            raise ValueError("Supported module calls are local, model.py and learn_brain")
        records = []
        for name in ([self.basic_data, self.more_training] if extra else [self.basic_data]):
            if not isinstance(name, str) or Path(name).suffix != ".cl":
                raise ValueError("Each selected dataset must be a workspace .cl file")
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Dataset missing or outside workspace: {name}")
            records.extend(read_dataset(path))
        return records


@dataclass(frozen=True)
class BrainWiring:
    """Allowlisted routes for a coding workspace."""

    learning: str = "learn.brain"
    coding: str = "coding.brain"
    memory: bool = True

    @classmethod
    def load(cls, root: Path) -> "BrainWiring":
        path = root / "main.brain"
        if not path.exists() and (root / "module_call.json").is_file():
            return cls(learning="module_call.json")
        try:
            wiring = cls(**json.loads(path.read_text(encoding="utf-8")))
            if wiring.learning != "learn.brain" or wiring.coding != "coding.brain":
                raise ValueError("Expected bundled routes learn.brain and coding.brain")
            if type(wiring.memory) is not bool:
                raise ValueError("memory must be a JSON boolean")
            model = json.loads((root / wiring.coding).read_text(encoding="utf-8"))
            if model != {"implementation": "model.py"}:
                raise ValueError("coding.brain must select the bundled model.py implementation")
            return wiring
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid brain wiring at {path}: {exc}") from exc


@dataclass(frozen=True)
class Project:
    name: str
    base: str | None
    type: str = "coding"
    max_tokens: int = 256
    device: str = "cpu"

    @classmethod
    def load(cls, root: Path):
        project = cls(**json.loads((root / "project.json").read_text(encoding="utf-8")))
        project.validate()
        return project

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        if self.type != "coding" or self.device not in {"cpu", "cuda"}:
            raise ValueError("Expected a coding model and device cpu or cuda")
        if type(self.max_tokens) is not int or not 32 <= self.max_tokens <= 1024:
            raise ValueError("max_tokens must be an integer from 32 to 1024")

    def source(self, root: Path) -> Path:
        if not self.base or not (root / self.base).is_dir():
            raise ValueError("Set base in project.json to a local safetensors model directory")
        return (root / self.base).resolve()


def create(
    root: Path, name: str, base: Path | None = None, *, device: str = "cpu", max_tokens: int = 256,
) -> Project:
    project = Project(name, str(base.resolve()) if base else None, max_tokens=max_tokens, device=device)
    project.validate()
    root.mkdir()
    files = {
        "project.json": asdict(project),
        "main.brain": asdict(BrainWiring()),
        "learn.brain": asdict(ModuleCall()),
        "coding.brain": {"implementation": "model.py"},
        "data.cl": {"prompt": "Reverse a Python string.", "completion": "Use text[::-1]."},
        "train.cl": {
            "prompt": "Remove duplicates in order from hashable values.",
            "completion": "In Python, return list(dict.fromkeys(values)).",
        },
    }
    for filename, content in files.items():
        (root / filename).write_text(json.dumps(content) + "\n", encoding="utf-8")
    (root / "run_carvustrain.py").write_text(
        "from pathlib import Path\nfrom carvustrain.coding.cli import workspace_main\n\n"
        "if __name__ == '__main__':\n    workspace_main(root=Path(__file__).resolve().parent)\n",
        encoding="utf-8",
    )
    return project


def read_dataset(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
                    raise ValueError("expected exactly prompt and completion")
                if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                    raise ValueError("prompt and completion must be non-empty strings")
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            records.append(row)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records
