"""Compile Python source without executing its bytecode."""

import subprocess
import sys
import tempfile

from .schemas import ValidationResult


def validate_python(code: str, timeout: float = 3) -> ValidationResult:
    source = code.encode("utf-8")
    if len(source) > 65536:
        return ValidationResult(None, stderr="Validation limit is 64 KiB")
    try:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c",
                 "import sys; compile(sys.stdin.buffer.read().decode('utf-8'), '<generated>', 'exec')"],
                input=source, capture_output=True, cwd=directory, timeout=timeout,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ValidationResult(None, stderr=str(exc))
    return ValidationResult(result.returncode, result.stderr.decode(errors="replace"),
                            result.stdout.decode(errors="replace"))
