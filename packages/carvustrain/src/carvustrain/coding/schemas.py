import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodingRequest:
    prompt: str
    checkpoint: str | None = None
    use_memory: bool = True
    validate_code: bool = False


@dataclass(frozen=True)
class ValidationResult:
    exit_status: int | None
    stderr: str = ""
    stdout: str = ""
    executed: bool = False


@dataclass
class CodingResponse:
    model: str
    source: str
    parameter_count: int | None = None
    answer: str | None = None
    code: str | None = None
    memory: list[dict] = field(default_factory=list)
    validation: ValidationResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 10
    learning_rate: float = 0.0001
    extra: bool = False

    def validate(self) -> None:
        if (type(self.steps) is not int or not 1 <= self.steps <= 1000
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1):
            raise ValueError("Use 1–1000 steps and a finite learning rate in (0, 1]")
