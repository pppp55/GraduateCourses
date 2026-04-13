from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

EXPECTED_MAJOR = 3
EXPECTED_MINOR = 10
PREFERRED_VERSION = "3.10.20"

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
PROJECT = ROOT / "src" / "project.py"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> None:
    if sys.version_info[:2] != (EXPECTED_MAJOR, EXPECTED_MINOR):
        raise SystemExit(
            f"This project should be set up with Python {EXPECTED_MAJOR}.{EXPECTED_MINOR}.x. "
            f"Current interpreter: {platform.python_version()}.\n"
            f"Recommended: Python {PREFERRED_VERSION}.\n"
            f"Windows: py -3.10 setup_and_run.py\n"
            f"macOS/Linux: python3.10 setup_and_run.py"
        )

    if platform.python_version() != PREFERRED_VERSION:
        print(
            f"[Warning] Official runtime is Python {PREFERRED_VERSION}, "
            f"but current interpreter is {platform.python_version()}. Continuing..."
        )

    if not REQ_FILE.exists():
        raise SystemExit(f"Missing requirements.txt: {REQ_FILE}")
    if not PROJECT.exists():
        raise SystemExit(f"Missing src/project.py: {PROJECT}")
    if not DATA_DIR.exists():
        raise SystemExit(f"Missing data directory: {DATA_DIR}")

    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    py = str(venv_python())
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", str(REQ_FILE)])

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    run(
        [
            py,
            str(PROJECT),
            "--data_dir", str(DATA_DIR),
            "--output_dir", str(OUTPUT_DIR),
            "--figure_dir", str(FIGURE_DIR),
        ],
        env=env,
    )

    print("\nEnvironment ready and project completed.")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
