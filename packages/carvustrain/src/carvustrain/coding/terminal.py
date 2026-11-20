"""Interactive terminal controls for local coding-model workspaces."""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from contextvars import ContextVar
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

from ..version import __version__
from .demo_model import train_test_model
from .main_brain import MainBrain
from .model import weight_size
from .module_call import Project, create
from .schemas import CodingRequest, TrainingConfig

_BRAND = "1;36"
_ACCENT = "1;35"
_SUCCESS = "1;32"
_CAUTION = "1;33"
_ERROR = "1;31"
_COLOR_MODE: ContextVar[str] = ContextVar("terminal_color_mode", default="auto")
# A terminal-sized silhouette of the mark in logo.png. Each pixel occupies two columns.
_LOGO = (
    "  #### ",
    " #    #",
    " #   # ",
    "#      ",
    "#    # ",
    "#  ##  ",
    " ###   ",
)

_MENU = (
    ("Workspace", (("1", "Create workspace"), ("2", "Open workspace"),
                   ("3", "Model settings"), ("4", "Validate data"), ("8", "Memory"))),
    ("Model", (("5", "Train model"), ("6", "Build test model"), ("7", "Ask model"),
               ("9", "Runtime setup"), ("0", "Exit"))),
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe(value: object) -> str:
    return _CONTROL.sub(lambda match: ascii(match[0])[1:-1], str(value))


def _color_enabled() -> bool:
    mode = _COLOR_MODE.get()
    return mode == "always" or (
        mode == "auto"
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
        and bool(os.name != "nt" or os.environ.get("WT_SESSION") or os.environ.get("TERM"))
    )


def _motion_enabled() -> bool:
    return sys.stdout.isatty() and "CARVUSTRAIN_NO_MOTION" not in os.environ


def _style(text: str, code: str = _BRAND) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _color_enabled() else text


def _section(title: str) -> None:
    print(f"\n  {_style(title.upper())}")


def _field(label: str, value: object, *, color: str | None = None) -> None:
    text = _safe(value)
    print(f"  {_style(f'{label:<13}', '1')} {_style(text, color) if color else text}")


def _masthead(width: int) -> None:
    title = _style("CARVUSTRAIN")
    subtitle = "Coding workspace"
    animate = _motion_enabled()
    pixel = "##"
    try:
        "█".encode(sys.stdout.encoding or "ascii")
    except (LookupError, UnicodeError):
        pass
    else:
        if sys.stdout.isatty():
            pixel = "██"

    details = {2: f"{title}  /  {__version__}", 3: subtitle}
    for index, row in enumerate(_LOGO):
        mark = "".join(pixel if cell == "#" else "  " for cell in row)
        color = _ACCENT if animate and index < 2 else _BRAND
        if width >= 44:
            print(f"  {_style(mark, color)}    {details.get(index, '')}", flush=animate)
        else:
            print(f"  {_style(mark.rstrip(), color)}", flush=animate)
        if animate and index < len(_LOGO) - 1:
            sleep(0.10)
    if width < 44:
        print(f"  {title}  /  {__version__}")


def _dashboard(active: Path | None, *, show_logo: bool = True) -> None:
    width = shutil.get_terminal_size(fallback=(76, 24)).columns
    rule = "-" * min(72, width)
    print(f"\n{_style(rule, '36')}")
    if show_logo:
        _masthead(width)
    else:
        print(f"  {_style('CARVUSTRAIN')}  /  {__version__}")
    print(_style(rule, "36"))
    if active is None:
        _field("Workspace", "Create or open a workspace to begin")
    else:
        _field("Workspace", active)
        try:
            project = Project.load(active)
        except (OSError, ValueError, TypeError) as exc:
            _field("Settings", exc)
        else:
            _field("Model", f"{project.name}  |  {project.device.upper()}  |  {project.max_tokens} context tokens")
            _field("Base model", project.base or "Set a local model in Model settings")

    def option(item: tuple[str, str], accent: str) -> str:
        key, label = item
        return f"{_style(f'[{key}]', accent)} {label}"

    if width >= 64:
        (left, left_items), (right, right_items) = _MENU
        print(f"\n  {_style(left.upper())}{' ' * (31 - len(left))}{_style(right.upper(), _ACCENT)}")
        for first, second in zip(left_items, right_items):
            print(f"  {option(first, _BRAND)}{' ' * (27 - len(first[1]))}{option(second, _ACCENT)}")
    else:
        for (heading, items), accent in zip(_MENU, (_BRAND, _ACCENT)):
            print(f"\n  {_style(heading.upper(), accent)}")
            for item in items:
                print(f"  {option(item, accent)}")
    print(_style(rule, "36"))
    if active is None:
        print(f"  First run? Try {_style('[6]', _ACCENT)} for a CPU test model.")
    else:
        print(f"  Check data with {_style('[4]')}. Ask a model with {_style('[7]', _ACCENT)}.")


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{_safe(default)}]" if default else ""
    return input(f"  {_style(label, '1')}{suffix} > ").strip() or default


def _confirm(label: str) -> bool:
    return _prompt(f"{label} (y/N)").lower() in {"y", "yes"}


def _install_runtime() -> None:
    if _confirm("Download and install the coding runtime into this Python environment?"):
        subprocess.run([sys.executable, "-m", "pip", "install", "carvustrain[coding]"], check=True)
        print("Runtime installed. Restart the terminal menu before training.")


def _device(default: str = "cpu") -> str:
    selected = _prompt("Device (cpu/cuda)", default).lower()
    if selected not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda.")
    if selected == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise ValueError("Install the coding runtime first, or select CPU for the workspace.") from exc
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment. Choose CPU or install CUDA-enabled PyTorch.")
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    return selected


def _model_settings(project: Project, *, root: Path) -> Project:
    base = _prompt("Local base-model folder (Enter to keep current)", project.base or "")
    path = (root / Path(base.strip('"')).expanduser()).resolve() if base else None
    if path is not None:
        size = weight_size(path)
        print(f"Base weights: {size / 1024**2:,.1f} MiB. Parameter count is measured when loaded.")
    updated = replace(
        project,
        base=str(path) if path else None,
        device=_device(project.device),
        max_tokens=int(_prompt("Context token limit (32-1024)", str(project.max_tokens))),
    )
    updated.validate()
    return updated


def _save_settings(root: Path, project: Project) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        path = Path(stream.name)
        try:
            json.dump(asdict(project), stream, indent=2)
        except BaseException:
            stream.close()
            path.unlink(missing_ok=True)
            raise
    try:
        path.replace(root / "project.json")
    finally:
        path.unlink(missing_ok=True)


def _progress(step: int, total: int) -> None:
    width = 24
    filled = width * step // total
    if _motion_enabled() and _color_enabled() and shutil.get_terminal_size().columns >= 56:
        bar = _style("#" * filled, "36") + "-" * (width - filled)
        print(f"\r  [{bar}]  Saved step {step}/{total:<4}", end="", flush=True)
    else:
        print(f"  Saved step {step}/{total}", flush=True)


def _train(brain: MainBrain) -> None:
    device = Project.load(brain.root).device.upper()
    config = TrainingConfig(
        steps=int(_prompt("Training steps (1-1000)", "2")),
        learning_rate=float(_prompt("Learning rate", "0.0001")),
        extra=_confirm("Include train.cl?"),
    )
    if not _confirm(f"Start local training on {device}?"):
        return
    print("Preparing model and data. Press Ctrl+C to cancel.")
    try:
        job = brain.learning.train(config, on_progress=_progress)
    finally:
        print()
    _show_training_result(job)


def _show_training_result(job: dict) -> None:
    _section("Training result")
    state = job["state"]
    state_color = {"completed": _SUCCESS, "cancelled": _CAUTION, "failed": _ERROR}.get(state, _BRAND)
    _field("Status", f"{state.capitalize()} | {job['step']}/{job['config']['steps']} saved steps", color=state_color)
    _field("Job ID", job["id"])
    if job["checkpoint"]:
        _field("Checkpoint", job["checkpoint"])
    if job["error"]:
        _field("Error", job["error"], color=_ERROR)


def _run_test_model(current: Path) -> Path:
    print("Builds a tiny model from random weights, then trains it locally on CPU.")
    print("Use it to verify the pipeline. Its answers are for testing only.")
    target = Path(_prompt("Test workspace folder", str(current / "CarvusTrain-test"))).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Choose a new workspace folder. {target} already exists.")
    print("Training one step and saving a checkpoint...")
    job = train_test_model(target, TrainingConfig(steps=1), on_progress=_progress)
    print()
    _show_training_result(job)
    return target


def _memory(brain: MainBrain) -> None:
    print("  [1] Add note      [2] Search notes\n"
          "  [3] View note     [4] Delete note\n"
          "  [0] Back")
    action = _prompt("Memory action", "0").lower()
    action = {"1": "add", "2": "search", "3": "get", "4": "delete"}.get(action, action)
    memory = brain.memory
    if action == "add":
        _field("Saved note", f"#{memory.add(_prompt('Text'))}")
    elif action in {"search", "get"}:
        if action == "search":
            notes = memory.search(_prompt("Search text"))
        else:
            note = memory.retrieve(int(_prompt("Note ID")))
            notes = [note] if note else []
        for note in notes:
            _field(f"Note #{note['id']}", note["text"])
        if not notes:
            print("  No matching notes.")
    elif action == "delete":
        if _confirm("Delete a saved memory?"):
            print("  Deleted." if memory.delete(int(_prompt("Note ID"))) else "  Memory entry missing.")
    elif action != "0":
        print("  Choose a number from the memory menu.")


def _workspace_action(choice: str, brain: MainBrain) -> None:
    if choice == "4":
        count = len(brain.learning.validate(_confirm("Include train.cl?")))
        print(_style(f"Validated {count} training examples.", _SUCCESS))
    elif choice == "5":
        _train(brain)
    elif choice == "7":
        checkpoints = [job for job in brain.learning.status() if job["checkpoint"]]
        _field("[0]", "Base model")
        for index, job in enumerate(checkpoints, 1):
            _field(f"[{index}]", f"Checkpoint {job['id'][:8]}  |  {job['step']} saved steps")
        selected = int(_prompt("Use model", "1" if checkpoints else "0"))
        if not 0 <= selected <= len(checkpoints):
            raise ValueError("Choose one of the listed models.")
        checkpoint = checkpoints[selected - 1]["id"] if selected else None
        question = _prompt("Coding question")
        if not question:
            raise ValueError("Enter a coding question.")
        check_syntax = _confirm("Check Python syntax in the answer?")
        if check_syntax:
            print("  The syntax checker looks for a fenced Python block in the answer.")
        print("  Generating response...", flush=True)
        result = brain.ask(CodingRequest(question, checkpoint, validate_code=check_syntax))
        _section("Model response")
        _field("Model", result.model)
        _field("Source", result.source)
        if result.parameter_count is not None:
            _field("Parameters", f"{result.parameter_count:,}")
        if result.memory:
            _field("Memory", f"{len(result.memory)} matching notes")
        if result.answer:
            _section("Answer")
            for line in _safe(result.answer).split("\n"):
                print(_style(line, _ACCENT) if line.lstrip().startswith("```") else line)
        if result.validation:
            exit_status = result.validation.exit_status
            if exit_status is None:
                _field("Python syntax", "Check unavailable", color=_CAUTION)
            elif exit_status == 0:
                _field("Python syntax", "Passed", color=_SUCCESS)
            else:
                _field("Python syntax", f"Failed (exit {exit_status})", color=_ERROR)
            if result.validation.stderr:
                print(_safe(result.validation.stderr.rstrip()))
        if result.error:
            _field("Error", result.error, color=_ERROR)
    elif choice == "8":
        _memory(brain)


def run_menu(current: Path, *, color: str = "auto") -> None:
    """Run the menu without changing the parent terminal's working directory."""
    token = _COLOR_MODE.set(color)
    try:
        _run_menu(current)
    finally:
        _COLOR_MODE.reset(token)


def _run_menu(current: Path) -> None:
    active = None
    for candidate in (current, current / ".carvustrain"):
        if (candidate / "project.json").is_file() and any(
            (candidate / name).is_file() for name in ("main.brain", "module_call.json")
        ):
            active = candidate
            break
    show_logo = True
    while True:
        try:
            _dashboard(active, show_logo=show_logo)
            show_logo = False
            choice = _prompt("Enter a menu number")
            if choice == "0":
                return
            for _, items in _MENU:
                for key, label in items:
                    if key == choice:
                        _section(label)
            if choice == "1":
                name = _prompt("Model name", "Test-model")
                target = Path(_prompt("Workspace folder", name)).expanduser().resolve()
                project = _model_settings(Project(name, None), root=current)
                create(target, name, Path(project.base) if project.base else None,
                       device=project.device, max_tokens=project.max_tokens)
                active = target
                print(f"  Created {_safe(target)}. Edit data.cl and train.cl to add examples.")
            elif choice == "2":
                target = Path(_prompt("Workspace folder", str(active or current))).expanduser().resolve()
                Project.load(target)
                active = target
            elif choice == "6":
                active = _run_test_model(current)
            elif choice == "9":
                _install_runtime()
            elif choice in {"3", "4", "5", "7", "8"}:
                if active is None:
                    print("Create or open a workspace first.")
                elif choice == "3":
                    _save_settings(active, _model_settings(Project.load(active), root=active))
                    print("Saved model settings.")
                else:
                    _workspace_action(choice, MainBrain(active))
            else:
                print("Choose a number from the menu.")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except (OSError, ValueError, TypeError, sqlite3.Error, subprocess.CalledProcessError) as exc:
            print(f"  {_style('Error', _ERROR)}: {_safe(exc)}")
