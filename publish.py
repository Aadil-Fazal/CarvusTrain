"""Build and publish the CarvusTrain package to PyPI."""
import os
import shutil
import subprocess
import sys

from CarvusTrain.version import __version__

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n[{desc}]")
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[:500]}")
        print(f"  STDERR: {result.stderr[:500]}")
        print(f"  [FAILED] {desc} — exit code {result.returncode}")
        sys.exit(1)
    print(f"  [OK] {desc}")


def main() -> None:
    print("=" * 60)
    print(f"CarvusTrain v{__version__} — Build & Publish to PyPI")
    print("=" * 60)

    # Step 1: Clean old builds
    for d in ["dist", "build"]:
        p = os.path.join(PROJECT_ROOT, d)
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"  Cleaned: {d}/")

    for item in os.listdir(PROJECT_ROOT):
        if item.endswith(".egg-info") and os.path.isdir(item):
            shutil.rmtree(os.path.join(PROJECT_ROOT, item))
            print(f"  Cleaned: {item}")

    # Step 2: Install build & twine if needed
    run_cmd(
        [sys.executable, "-m", "pip", "install", "--quiet", "build", "twine"],
        "Install build & twine",
    )

    # Step 3: Build the wheel
    run_cmd(
        [sys.executable, "-m", "build"],
        "Build wheel & sdist",
    )

    # Step 4: Upload to PyPI
    token = os.environ.get("PYPI_TOKEN")
    if not token:
        print("\n[ERROR] PYPI_TOKEN environment variable not set.")
        print("  Set it and re-run, or paste your token below.\n")
        token = input("Paste your PyPI API token: ").strip()
        if not token:
            print("No token provided. Aborting.")
            sys.exit(1)

    run_cmd(
        [
            sys.executable, "-m", "twine", "upload",
            "dist/*",
            "--username", "__token__",
            "--password", token,
        ],
        "Upload to PyPI",
    )

    print("\n" + "=" * 60)
    print(f"[SUCCESS] CarvusTrain v{__version__} published to PyPI!")
    print("=" * 60)
    print("\n  Install it with:")
    print("    pip install carvustrain")
    print()


if __name__ == "__main__":
    main()
