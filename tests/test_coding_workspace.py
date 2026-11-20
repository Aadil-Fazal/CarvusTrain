import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from carvustrain.cli import main
from carvustrain.coding import learn_brain, main_brain
from carvustrain.coding.code_runner import validate_python
from carvustrain.coding.main_brain import MainBrain
from carvustrain.coding.module_call import create, read_dataset
from carvustrain.coding.schemas import CodingRequest, TrainingConfig


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "Test-model"
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"fixture")
    create(root, "Test-model", base)
    return root


def test_module_calls_and_dataset(workspace):
    brain = MainBrain(workspace)
    assert len(brain.learning.validate()) == 1
    assert len(brain.learning.validate(extra=True)) == 2
    (workspace / "train.cl").write_text('{"prompt":"missing completion"}\n')
    with pytest.raises(ValueError, match="train.cl:1"):
        brain.learning.validate(extra=True)
    with pytest.raises(ValueError, match="empty"):
        (workspace / "train.cl").write_text("\n")
        read_dataset(workspace / "train.cl")
    path = workspace / "learn.brain"
    path.write_text(json.dumps({"call_code": "unsupported.py"}))
    with pytest.raises(ValueError, match="Supported module calls"):
        brain.learning.validate()


def test_memory_and_routing(workspace, monkeypatch):
    class Model:
        parameter_count = 42

        def __init__(self, path, *args):
            assert path.name == "base"

        def answer(self, prompt):
            assert "Python slicing" in prompt
            return "```python\nprint('hello')\n```"

    monkeypatch.setattr(main_brain, "CodingModel", Model)
    brain = MainBrain(workspace)
    entry_id = brain.memory.add("Python slicing")
    response = MainBrain(workspace).ask(CodingRequest("Explain slicing", validate_code=True))
    assert response.model == "Test-model"
    assert response.source == "base"
    assert response.answer and response.code and response.error is None
    assert response.validation.exit_status == 0
    assert not response.validation.executed
    assert brain.memory.search("Explain slicing") == [brain.memory.retrieve(entry_id)]
    assert brain.memory.delete(entry_id)
    assert brain.memory.retrieve(entry_id) is None


def test_validation_never_evaluates_python(tmp_path):
    marker = tmp_path / "marker"
    assert validate_python(f"open({str(marker)!r}, 'w').close()").exit_status == 0
    assert not marker.exists()
    assert validate_python("if").exit_status != 0
    assert validate_python("x" * 65537).exit_status is None


def test_training_failure_cancel_resume_and_reload(workspace, monkeypatch):
    brain = MainBrain(workspace)
    seen_sources = []

    class Model:
        parameter_count = 42

        def __init__(self, path, *args, **kwargs):
            seen_sources.append(path.name)

        def start_training(self, lr):
            pass

        def train_step(self, example):
            raise RuntimeError("training failed")

        def save(self, path):
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"fixture")

        def answer(self, prompt):
            return "A checkpoint answer"

    monkeypatch.setattr(learn_brain, "CodingModel", Model)
    monkeypatch.setattr(main_brain, "CodingModel", Model)
    job = brain.learning.train(TrainingConfig(steps=2))
    assert (job["state"], job["step"], job["error"]) == ("failed", 0, "training failed")

    def cancel_during_step(model, example):
        assert brain.learning.cancel(job["id"])
        return 0.5

    monkeypatch.setattr(Model, "train_step", cancel_during_step)
    cancelled = brain.learning.train(resume=job["id"])
    assert (cancelled["state"], cancelled["step"]) == ("cancelled", 1)
    monkeypatch.setattr(Model, "train_step", lambda self, example: 0.4)
    complete = brain.learning.train(resume=job["id"])
    assert (complete["state"], complete["step"]) == ("completed", 2)
    assert seen_sources == ["base", "base", "step-1"]
    answer = brain.ask(CodingRequest("Question", checkpoint=job["id"]))
    assert answer.source == f"fine-tuned:{job['id']}"
    assert seen_sources[-1] == "step-2"


def test_bare_cli_scaffolds_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main([])
    dataset = tmp_path / ".carvustrain" / "data.cl"
    dataset.write_text('{"prompt":"mine","completion":"keep it"}\n')
    main([])
    main(["coding", "validate", "--extra"])
    assert '"examples": 2' in capsys.readouterr().out
    assert read_dataset(dataset)[0]["prompt"] == "mine"


def test_real_local_backend():
    with TemporaryDirectory(prefix="coding-smoke-", dir=Path(__file__).resolve().parent) as directory:
        _train_local_model(Path(directory))
    assert not Path(directory).exists()


def _train_local_model(tmp_path):
    if any(find_spec(name) is None
           for name in ("torch", "transformers", "psutil", "tokenizers")):
        pytest.skip("coding extra unavailable")
    import torch
    from unittest.mock import patch
    from carvustrain.coding.model import CodingModel
    from carvustrain.coding.terminal import _run_test_model

    with patch("builtins.input", return_value=""):
        root = _run_test_model(tmp_path)
    brain = MainBrain(root)
    job, = brain.learning.status()
    assert job["state"] == "completed", job["error"]
    base = root / "base"
    loaded = CodingModel(brain.learning.checkpoint(job["id"]), "cpu", 64)
    assert loaded.parameter_count == job["parameters"]
    original = CodingModel(base, "cpu", 64)
    assert any(not torch.equal(before, after) for before, after in
               zip(original.network.parameters(), loaded.network.parameters()))
    assert isinstance(loaded.answer("Use Python"), str)



def test_training_lock_recovers_abandoned_job(workspace):
    from carvustrain.coding.database import connect, training_lock

    with training_lock(workspace), pytest.raises(ValueError, match="already running"):
        with training_lock(workspace):
            pytest.fail("A second worker acquired the training lock")
    with connect(workspace) as db:
        db.execute("INSERT INTO jobs(id, config, fingerprint, state) VALUES ('old', '{}', '', 'running')")
    with training_lock(workspace):
        assert MainBrain(workspace).learning.status("old")["state"] == "failed"


def test_brain_configuration_and_legacy_fallback(workspace):
    brain = MainBrain(workspace)
    wiring = workspace / "coding.brain"
    wiring.write_text('{"implementation": "unknown.py"}')
    with pytest.raises(ValueError, match="bundled model.py"):
        brain.learning.validate()
    answer = brain.ask(CodingRequest("Explain slicing"))
    assert "bundled model.py" in answer.error
    (workspace / "learn.brain").rename(workspace / "module_call.json")
    (workspace / "main.brain").unlink()
    assert len(brain.learning.validate()) == 1


def test_interactive_menu_creates_and_validates_workspace(tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(["1", "My coder", "my-coder", "", "cpu", "64", "4", "y", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    main([])

    result = json.loads((tmp_path / "my-coder" / "project.json").read_text())
    assert (result["name"], result["device"], result["max_tokens"]) == ("My coder", "cpu", 64)
    assert "Validated 2 training examples." in capsys.readouterr().out


def test_interactive_download_requires_confirmation(tmp_path, monkeypatch):
    import subprocess

    from carvustrain.coding.terminal import run_menu

    answers = iter(["9", "", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected download"))
    run_menu(tmp_path)
