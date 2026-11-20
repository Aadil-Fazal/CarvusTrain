from pathlib import Path

from .code_runner import validate_python
from .coding_brain import generate
from .ct_basic_memory import BasicMemory
from .learn_brain import LearningBrain
from .model import CodingModel
from .module_call import BrainWiring, Project
from .schemas import CodingRequest, CodingResponse


class MainBrain:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.memory = BasicMemory(self.root)
        self.learning = LearningBrain(self.root)

    def ask(self, request: CodingRequest) -> CodingResponse:
        result = CodingResponse(model=self.root.name, source="unavailable")
        try:
            if not request.prompt.strip():
                raise ValueError("Enter a coding question")
            project = Project.load(self.root)
            result.model = project.name
            wiring = BrainWiring.load(self.root)
            path = self.learning.checkpoint(request.checkpoint) if request.checkpoint else project.source(self.root)
            model = CodingModel(path, project.device, project.max_tokens)
            result.source = f"fine-tuned:{request.checkpoint}" if request.checkpoint else "base"
            result.parameter_count = model.parameter_count
            if request.use_memory and wiring.memory:
                result.memory = self.memory.search(request.prompt)
            result.answer, result.code = generate(model, request.prompt, result.memory)
            if request.validate_code:
                if result.code is None:
                    raise ValueError("The model returned no fenced Python code to validate")
                result.validation = validate_python(result.code)
                if result.validation.exit_status != 0:
                    result.error = "Python syntax validation failed"
        except Exception as exc:
            result.error = str(exc)
        return result
