"""Bootstrap helpers — activate CarvusTrain without ModuleNotFoundError.

Training is never started here. Callers import and then train explicitly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_monorepo_root(start: Optional[str] = None) -> Optional[Path]:
    """Walk parents until carvustrain (root-level or packages/) is found."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for _ in range(12):
        # Check root-level carvustrain/ package (new layout)
        if (current / "carvustrain" / "__init__.py").is_file():
            return current
        # Check packages/carvustrain layout (legacy)
        candidate = current / "packages" / "carvustrain"
        if candidate.is_dir() and (
            (candidate / "pyproject.toml").is_file()
            or (candidate / "src" / "carvustrain").is_dir()
        ):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _src_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    packages = root / "packages"
    if packages.is_dir():
        for pkg in sorted(packages.iterdir()):
            src = pkg / "src"
            if src.is_dir():
                paths.append(src)
    # Root-level carvustrain/ package (new layout)
    if (root / "carvustrain" / "__init__.py").is_file():
        paths.append(root)
    return paths


def activate(install_if_missing: bool = True) -> str:
    """Ensure ``carvustrain`` is importable. Returns resolved package path.

    Safe to call from any script that uses CarvusTrain. Does **not** train.
    """
    try:
        import carvustrain as ct  # noqa: WPS433

        return str(Path(ct.__file__).resolve().parent)
    except ImportError:
        pass

    root = find_monorepo_root()
    if root is None:
        root = find_monorepo_root(os.getcwd())

    if root is not None:
        for src in _src_paths(root):
            s = str(src)
            if s not in sys.path:
                sys.path.insert(0, s)

    try:
        import carvustrain as ct  # noqa: WPS433

        return str(Path(ct.__file__).resolve().parent)
    except ImportError:
        if not install_if_missing:
            raise

    pkg_root = None
    if root is not None:
        # Check root-level carvustrain/ first (new layout)
        if (root / "carvustrain" / "__init__.py").is_file():
            pkg_root = root
        else:
            pkg_root = root / "packages" / "carvustrain"
            if not pkg_root.is_dir():
                pkg_root = root if (root / "pyproject.toml").is_file() else None

    if pkg_root is None:
        raise ImportError(
            "CarvusTrain not found. Install with: pip install -e .  or  pip install carvustrain"
        )

    # Only install if pkg_root has pyproject.toml
    if (pkg_root / "pyproject.toml").is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(pkg_root)])
    sys.modules.pop("carvustrain", None)
    import carvustrain as ct  # noqa: WPS433

    return str(Path(ct.__file__).resolve().parent)


def activate_ecosystem() -> List[str]:
    """Put all monorepo package src dirs on sys.path. Returns added paths."""
    root = find_monorepo_root() or find_monorepo_root(os.getcwd())
    added: List[str] = []
    if root is None:
        return added
    for src in _src_paths(root):
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    return added
