from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from embedding_utils import (
    deterministic_sample,
    load_embeddings,
    model_name_from_path,
    nearest_to_vector,
    normalize_token,
    project_path,
    resolve_seed,
    write_csv,
)


def read_analogy_questions(path: Path) -> OrderedDict[str, list[tuple[str, str, str, str]]]:
    categories: OrderedDict[str, list[tuple[str, str, str, str]]] = OrderedDict()
    current_category: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(":"):
                current_category = stripped[1:].strip()
                categories[current_category] = []
                continue
            parts = stripped.split()
            if len(parts) == 4 and current_category is not None:
                categories[current_category].append(tuple(part.lower() for part in parts))
    return categories


def evaluate_model(
    model_path: Path,
    analogy_path: Path,
    sample_count: int,
    seed: int,
    plot_categories: int,
    out_dir: Path,
) -> dict:
    model_name = model_name_from_path(model_path)
    embeddings = load_embeddings(model_path)
    categories = read_analogy_questions(analogy_path)

    all_rows: list[dict] = []
    category_rows: list[dict] = []
    covered_for_sampling: list[dict] = []
    plot_candidates: OrderedDict[str, dict] = OrderedDict()

    total_questions = 0
    total_covered = 0
    total_correct = 0

    for category, questions in categories.items():
        category_total = len(questions)
        category_covered = 0
        category_correct = 0
        total_questions += category_total

        for word_a, word_b, word_c, expected in questions:
            covered = all(embeddings.has(word) for word in (word_a, word_b, word_c, expected))
            row = {
                "model": model_name,
                "category": category,
                "word_a": word_a,
                "word_b": word_b,
                "word_c": word_c,
                "expected": expected,
                "covered": int(covered),
                "predicted": "",
                "cosine_similarity": "",
                "correct": "",
            }
            if covered:
                predicted_vector = embeddings.vector(word_b) - embeddings.vector(word_a) + embeddings.vector(word_c)
                predicted, similarity, _ = nearest_to_vector(
                    embeddings,
                    predicted_vector,
                    exclude=(word_a, word_b, word_c),
                )
                is_correct = normalize_token(predicted) == normalize_token(expected)
                row["predicted"] = predicted
                row["cosine_similarity"] = f"{similarity:.6f}"
                row["correct"] = int(is_correct)
                category_covered += 1
                total_covered += 1
                if is_correct:
                    category_correct += 1
                    total_correct += 1
                covered_for_sampling.append(row)
                if category not in plot_candidates:
                    plot_candidates[category] = {
                        "row": row,
                        "predicted_vector": predicted_vector,
                        "nearest_vector": embeddings.vector(predicted),
                    }
            all_rows.append(row)

        category_rows.append({
            "model": model_name,
            "category": category,
            "total_questions": category_total,
            "covered_questions": category_covered,
            "correct": category_correct,
            "accuracy": f"{category_correct / category_covered:.6f}" if category_covered else "",
        })

    sample_rows = deterministic_sample(covered_for_sampling, sample_count, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / f"{model_name}_analogy_all.csv",
        all_rows,
        ["model", "category", "word_a", "word_b", "word_c", "expected", "covered", "predicted", "cosine_similarity", "correct"],
    )
    write_csv(
        out_dir / f"{model_name}_analogy_by_category.csv",
        category_rows,
        ["model", "category", "total_questions", "covered_questions", "correct", "accuracy"],
    )
    write_csv(
        out_dir / f"{model_name}_analogy_sample_{sample_count}.csv",
        sample_rows,
        ["model", "category", "word_a", "word_b", "word_c", "expected", "covered", "predicted", "cosine_similarity", "correct"],
    )
    _write_vector_plots(model_name, list(plot_candidates.values())[:plot_categories], out_dir)

    return {
        "model": model_name,
        "total_questions": total_questions,
        "covered_questions": total_covered,
        "correct": total_correct,
        "accuracy": total_correct / total_covered if total_covered else float("nan"),
    }


def _write_vector_plots(model_name: str, candidates: list[dict], out_dir: Path) -> None:
    plot_dir = out_dir / f"{model_name}_analogy_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    for candidate in candidates:
        row = candidate["row"]
        category = row["category"].replace("/", "_")
        safe_name = f"{category}_{row['word_a']}_{row['word_b']}_{row['word_c']}"
        predicted_vector = np.asarray(candidate["predicted_vector"], dtype=float)
        nearest_vector = np.asarray(candidate["nearest_vector"], dtype=float)
        dims = list(range(1, len(predicted_vector) + 1))

        csv_path = plot_dir / f"{safe_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dimension", "predicted_vector", "nearest_word_vector"])
            for dim, pred_value, near_value in zip(dims, predicted_vector, nearest_vector):
                writer.writerow([dim, pred_value, near_value])

        if plt is None:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(dims, predicted_vector, linestyle="--", label="D_pred")
        plt.plot(dims, nearest_vector, label=f"nearest: {row['predicted']}")
        plt.xlabel("Embedding dimension")
        plt.ylabel("Vector value")
        plt.title(
            f"{row['category']}: {row['word_a']}:{row['word_b']} :: "
            f"{row['word_c']}:{row['expected']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{safe_name}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analogy reasoning evaluation.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(project_path("embeddings", "cbow.vec")), str(project_path("embeddings", "skipgram.vec"))],
    )
    parser.add_argument("--analogies", default=str(project_path("data", "analogical reasoning task.txt")))
    parser.add_argument("--student-id", default=os.environ.get("STUDENT_ID", "0"))
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--plot-categories", type=int, default=5)
    parser.add_argument("--out-dir", default=str(project_path("results")))
    args = parser.parse_args()

    seed = resolve_seed(args.student_id)
    summaries = []
    for model in args.models:
        model_path = Path(model)
        if not model_path.exists():
            print(f"[SKIP] Missing embedding file: {model_path}")
            continue
        summary = evaluate_model(
            model_path,
            Path(args.analogies),
            args.sample_count,
            seed,
            args.plot_categories,
            Path(args.out_dir),
        )
        summaries.append(summary)
        print(
            f"{summary['model']}: For {summary['covered_questions']} analogy questions with all words "
            f"presented in the vocabulary, {summary['correct']} were correctly answered. "
            f"Accuracy = {summary['accuracy']:.6f}"
        )

    if not summaries:
        raise SystemExit("No embedding files were evaluated. Put cbow.vec and skipgram.vec in embeddings/ first.")


if __name__ == "__main__":
    main()
