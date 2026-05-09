from __future__ import annotations

import argparse
import subprocess
import sys

from embedding_utils import project_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Project 1-2 embedding evaluations."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            str(project_path("embeddings", "cbow.vec")),
            str(project_path("embeddings", "skipgram.vec")),
        ],
    )
    parser.add_argument("--student-id", default="225040065")
    parser.add_argument("--out-dir", default=str(project_path("results")))
    args = parser.parse_args()

    commands = [
        [
            sys.executable,
            str(project_path("scripts", "knn_evaluation.py")),
            "--models",
            *args.models,
            "--student-id",
            args.student_id,
            "--out-dir",
            args.out_dir,
        ],
        [
            sys.executable,
            str(project_path("scripts", "golden_standard_evaluation.py")),
            "--models",
            *args.models,
            "--student-id",
            args.student_id,
            "--out-dir",
            args.out_dir,
        ],
        [
            sys.executable,
            str(project_path("scripts", "analogy_evaluation.py")),
            "--models",
            *args.models,
            "--student-id",
            args.student_id,
            "--out-dir",
            args.out_dir,
        ],
    ]

    for command in commands:
        print("\n$", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=project_path())


if __name__ == "__main__":
    main()
