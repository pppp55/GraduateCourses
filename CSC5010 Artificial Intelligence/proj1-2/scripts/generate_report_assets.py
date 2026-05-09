from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "report" / "assets"


def save_knn_overall() -> None:
    rows = []
    for model in ["cbow", "skipgram"]:
        df = pd.read_csv(RESULTS / f"{model}_knn_summary.csv")
        rows.append((model.upper() if model == "cbow" else "Skip-Gram", pd.to_numeric(df["average_similarity"], errors="coerce").mean()))
    labels, values = zip(*rows)
    plt.figure(figsize=(5.8, 3.4))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.ylim(0, max(values) * 1.2)
    plt.ylabel("Average top-10 cosine similarity")
    plt.title("KNN overall average similarity")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS / "knn_overall.png", dpi=180)
    plt.close()


def save_knn_detail() -> None:
    cbow = pd.read_csv(RESULTS / "cbow_knn_summary.csv")
    skip = pd.read_csv(RESULTS / "skipgram_knn_summary.csv")
    detail_words = sorted(pd.read_csv(RESULTS / "cbow_knn_detail_words.csv")["query"].unique())
    cbow = cbow[cbow["query"].isin(detail_words)].set_index("query")
    skip = skip[skip["query"].isin(detail_words)].set_index("query")
    x = range(len(detail_words))
    width = 0.35
    cbow_values = [float(cbow.loc[w, "average_similarity"]) for w in detail_words]
    skip_values = [float(skip.loc[w, "average_similarity"]) for w in detail_words]
    plt.figure(figsize=(6.4, 3.5))
    plt.bar([i - width / 2 for i in x], cbow_values, width=width, label="CBOW", color="#4C78A8")
    plt.bar([i + width / 2 for i in x], skip_values, width=width, label="Skip-Gram", color="#F58518")
    plt.xticks(list(x), detail_words)
    plt.ylabel("Average top-10 cosine")
    plt.title("Seed-selected KNN query words")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "knn_detail_words.png", dpi=180)
    plt.close()


def save_simlex_scatter() -> None:
    plt.figure(figsize=(6.2, 4.0))
    for model, color, label in [("cbow", "#4C78A8", "CBOW"), ("skipgram", "#F58518", "Skip-Gram")]:
        df = pd.read_csv(RESULTS / f"{model}_simlex_sample_20.csv")
        plt.scatter(df["standard_similarity"], df["scaled_similarity"], s=32, alpha=0.78, color=color, label=label)
    plt.plot([0, 10], [0, 10], linestyle="--", color="#666666", linewidth=1)
    plt.xlabel("SimLex human standard score")
    plt.ylabel("Embedding scaled similarity")
    plt.title("SimLex-999 sampled pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "simlex_sample_scatter.png", dpi=180)
    plt.close()


def save_analogy_category() -> None:
    cbow = pd.read_csv(RESULTS / "cbow_analogy_by_category.csv")
    skip = pd.read_csv(RESULTS / "skipgram_analogy_by_category.csv")
    categories = cbow["category"].tolist()
    labels = [c.replace("gram", "g") for c in categories]
    x = range(len(categories))
    width = 0.35
    plt.figure(figsize=(8.5, 4.2))
    plt.bar([i - width / 2 for i in x], cbow["accuracy"], width=width, label="CBOW", color="#4C78A8")
    plt.bar([i + width / 2 for i in x], skip["accuracy"], width=width, label="Skip-Gram", color="#F58518")
    plt.xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Accuracy")
    plt.title("Analogy accuracy by category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "analogy_category_accuracy.png", dpi=180)
    plt.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_knn_overall()
    save_knn_detail()
    save_simlex_scatter()
    save_analogy_category()
    print(ASSETS)


if __name__ == "__main__":
    main()
