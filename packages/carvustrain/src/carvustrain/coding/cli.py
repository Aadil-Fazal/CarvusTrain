import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .main_brain import MainBrain
from .module_call import Project
from .schemas import CodingRequest, TrainingConfig


def workspace_main(args: list[str] | None = None, *, root: Path | None = None):
    parser = argparse.ArgumentParser(prog="carvustrain coding")
    default = root or Path.cwd()
    if root is None and not (default / "project.json").is_file() and (default / ".carvustrain").is_dir():
        default = default / ".carvustrain"
    parser.add_argument("--workspace", type=Path, default=default)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("test-model", help="Create a new workspace and train a tiny CPU model")
    demo.add_argument("directory", type=Path, help="New workspace directory")
    demo.add_argument("--steps", type=int, default=1)
    demo.add_argument("--learning-rate", type=float, default=0.0001)
    commands.add_parser("info")
    validate = commands.add_parser("validate")
    validate.add_argument("--extra", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--extra", action="store_true")
    commands.add_parser("resume").add_argument("job")
    commands.add_parser("cancel").add_argument("job")
    commands.add_parser("status").add_argument("job", nargs="?")
    ask = commands.add_parser("ask")
    ask.add_argument("prompt")
    ask.add_argument("--checkpoint", help="Load the latest saved step of this job")
    ask.add_argument("--validate-code", action="store_true")
    ask.add_argument("--no-memory", action="store_true")
    memory = commands.add_parser("memory").add_subparsers(dest="operation", required=True)
    memory.add_parser("add").add_argument("text")
    memory.add_parser("search").add_argument("text")
    memory.add_parser("get").add_argument("id", type=int)
    memory.add_parser("delete").add_argument("id", type=int)

    options = parser.parse_args(args)
    brain = MainBrain(options.workspace)
    failed = False
    try:
        if options.command == "test-model":
            from .demo_model import train_test_model

            result = train_test_model(options.directory.expanduser().resolve(),
                                      TrainingConfig(options.steps, options.learning_rate))
            failed = result["state"] != "completed"
        elif options.command == "info":
            result = asdict(Project.load(brain.root))
        elif options.command == "validate":
            result = {"examples": len(brain.learning.validate(options.extra))}
        elif options.command in {"train", "resume"}:
            if options.command == "resume":
                result = brain.learning.train(resume=options.job)
            else:
                result = brain.learning.train(TrainingConfig(options.steps, options.learning_rate, options.extra))
            failed = result["state"] != "completed"
        elif options.command == "status":
            result = brain.learning.status(options.job)
        elif options.command == "cancel":
            result = {"cancel_requested": brain.learning.cancel(options.job)}
            failed = not result["cancel_requested"]
        elif options.command == "ask":
            result = asdict(brain.ask(CodingRequest(
                options.prompt, options.checkpoint, not options.no_memory, options.validate_code,
            )))
            failed = result["error"] is not None
        elif options.operation == "add":
            result = {"id": brain.memory.add(options.text)}
        elif options.operation == "search":
            result = brain.memory.search(options.text)
        elif options.operation == "get":
            result = brain.memory.retrieve(options.id)
            failed = result is None
        else:
            result = {"deleted": brain.memory.delete(options.id)}
            failed = not result["deleted"]
    except (OSError, ValueError, TypeError) as exc:
        result = {"error": str(exc)}
        failed = True
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)
