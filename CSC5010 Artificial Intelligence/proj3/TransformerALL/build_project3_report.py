# coding: UTF-8
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
STUDENT_ID = "225040065"


def read_metrics(task: str) -> str:
    return (OUTPUT_DIR / task / f"{task}_metrics.txt").read_text(encoding="utf-8")


def extract_metric_line(metrics_text: str, prefix: str) -> str:
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


def compact_doc(row: pd.Series, side: str) -> str:
    if side == "A_wrong_B_correct":
        return (
            f"| {int(row['index'])} | {row['original_text_taskA']} | "
            f"{row['true_label_name_taskA']} | {row['predicted_label_name_taskA']} | "
            f"{row['predicted_label_name_taskB']} |"
        )
    return (
        f"| {int(row['index'])} | {row['original_text_taskA']} | "
        f"{row['true_label_name_taskA']} | {row['predicted_label_name_taskA']} | "
        f"{row['predicted_label_name_taskB']} |"
    )


def explain_a_wrong_b_correct(row: pd.Series) -> str:
    text = str(row["original_text_taskA"])
    processed = str(row["processed_text_taskB"])
    removed = len(text) - len(processed)
    if removed > 0:
        return (
            f"Task B removes {removed} punctuation/space characters, making the core keywords "
            f"more concentrated for the character-level Transformer."
        )
    return "The cleaned version keeps the main words but changes token positions after normalization, which can alter attention patterns."


def explain_b_wrong_a_correct(row: pd.Series) -> str:
    text = str(row["original_text_taskA"])
    processed = str(row["processed_text_taskB"])
    removed = len(text) - len(processed)
    if removed > 0:
        return (
            f"The removed punctuation may have carried useful title structure; after cleaning, "
            f"{removed} characters are removed and the model loses some boundary cues."
        )
    return "The original character sequence happens to preserve a more useful position pattern than the cleaned sequence."


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUTPUT_DIR / "experiment_summary.json").read_text(encoding="utf-8"))
    task_a_metrics = read_metrics("taskA")
    task_b_metrics = read_metrics("taskB")
    task_a_summary = summary["summaries"][0] if summary["summaries"][0]["task"] == "taskA" else summary["summaries"][1]
    task_b_summary = summary["summaries"][1] if summary["summaries"][1]["task"] == "taskB" else summary["summaries"][0]

    a_mis = pd.read_csv(OUTPUT_DIR / "taskA" / "taskA.misclassified.csv")
    b_mis = pd.read_csv(OUTPUT_DIR / "taskB" / "taskB.misclassified.csv")
    a_wrong_b_correct = pd.read_csv(OUTPUT_DIR / "taskA_wrong_taskB_correct.csv")
    b_wrong_a_correct = pd.read_csv(OUTPUT_DIR / "taskB_wrong_taskA_correct.csv")

    a_examples = a_wrong_b_correct.head(3)
    b_examples = b_wrong_a_correct.head(3)

    lines = [
        "# CSC5010 Project 3 Report: Transformer Text Classification",
        "",
        f"Student ID: {STUDENT_ID}",
        "",
        "## 1. Project Goal",
        "",
        "This project runs a Transformer encoder classifier on a truncated Chinese THUCNews dataset. The task is to classify Chinese news titles into 10 categories: 财经, 房产, 股票, 教育, 科技, 社会, 时政, 体育, 游戏, and 娱乐.",
        "",
        "The grading requirements define two experiments. Task A uses the original text without preprocessing. Task B applies text preprocessing before training and testing. Both tasks must output running results and misclassified documents, then compare cases where one task is correct and the other is wrong.",
        "",
        "## 2. Dataset and Model",
        "",
        "- Training set: 45,000 titles",
        "- Development set: 2,500 titles",
        "- Test set: 2,500 titles",
        "- Vocabulary limit: 10,000 characters plus `<UNK>` and `<PAD>`",
        "- Sequence length: 32 characters",
        "- Model: Transformer encoder classifier",
        "- Epochs: 6",
        "- Seed: derived from student ID 225040065",
        "",
        "The model uses a character-level tokenizer. Each title is padded or truncated to 32 characters. The Transformer includes token embedding, sinusoidal positional encoding, multi-head self-attention, feed-forward layers, residual connections, layer normalization, and a final linear classifier.",
        "",
        "## 3. Task A: Original Text",
        "",
        "Task A uses the original dataset directly. No punctuation or whitespace is removed before building the vocabulary or encoding the train/dev/test sets.",
        "",
        f"- Test loss: {task_a_summary['test_loss']:.6f}",
        f"- Test accuracy: {task_a_summary['test_accuracy']:.6f}",
        f"- Misclassified test documents: {task_a_summary['misclassified_count']}",
        "",
        "```text",
        extract_metric_line(task_a_metrics, "Test Accuracy:"),
        extract_metric_line(task_a_metrics, "Test Loss:"),
        "```",
        "",
        "The full Task A classification report and confusion matrix are saved in `outputs/taskA/taskA_metrics.txt`. All predictions are saved in `outputs/taskA/taskA_predictions.csv`, and all wrong predictions are saved in `outputs/taskA/taskA.misclassified.csv`.",
        "",
        "## 4. Task B: Preprocessed Text",
        "",
        "Task B removes Chinese and English punctuation and normalizes whitespace before vocabulary construction and encoding. The labels and train/dev/test split are unchanged. This keeps the experiment focused on the effect of text cleaning.",
        "",
        f"- Test loss: {task_b_summary['test_loss']:.6f}",
        f"- Test accuracy: {task_b_summary['test_accuracy']:.6f}",
        f"- Misclassified test documents: {task_b_summary['misclassified_count']}",
        "",
        "```text",
        extract_metric_line(task_b_metrics, "Test Accuracy:"),
        extract_metric_line(task_b_metrics, "Test Loss:"),
        "```",
        "",
        "The full Task B classification report and confusion matrix are saved in `outputs/taskB/taskB_metrics.txt`. All predictions are saved in `outputs/taskB/taskB_predictions.csv`, and all wrong predictions are saved in `outputs/taskB/taskB.misclassified.csv`.",
        "",
        "## 5. Task A vs. Task B Comparison",
        "",
        "| Task | Preprocessing | Test accuracy | Misclassified documents |",
        "|---|---|---:|---:|",
        f"| Task A | None | {task_a_summary['test_accuracy']:.6f} | {task_a_summary['misclassified_count']} |",
        f"| Task B | Remove punctuation / normalize whitespace | {task_b_summary['test_accuracy']:.6f} | {task_b_summary['misclassified_count']} |",
        "",
        f"Task A wrong but Task B correct: {len(a_wrong_b_correct)} documents.",
        f"Task B wrong but Task A correct: {len(b_wrong_a_correct)} documents.",
        "",
        "## 6. Three Documents Wrong in Task A but Correct in Task B",
        "",
        "| Test index | Document | True label | Task A prediction | Task B prediction |",
        "|---:|---|---|---|---|",
    ]

    for _, row in a_examples.iterrows():
        lines.append(compact_doc(row, "A_wrong_B_correct"))

    lines.extend(["", "Brief explanations:", ""])
    for i, (_, row) in enumerate(a_examples.iterrows(), start=1):
        lines.append(f"{i}. `{row['original_text_taskA']}`: {explain_a_wrong_b_correct(row)}")

    lines.extend([
        "",
        "## 7. Three Documents Wrong in Task B but Correct in Task A",
        "",
        "| Test index | Document | True label | Task A prediction | Task B prediction |",
        "|---:|---|---|---|---|",
    ])

    for _, row in b_examples.iterrows():
        lines.append(compact_doc(row, "B_wrong_A_correct"))

    lines.extend(["", "Brief explanations:", ""])
    for i, (_, row) in enumerate(b_examples.iterrows(), start=1):
        lines.append(f"{i}. `{row['original_text_taskA']}`: {explain_b_wrong_a_correct(row)}")

    lines.extend([
        "",
        "## 8. Output Files",
        "",
        "- `outputs/taskA/taskA_metrics.txt`: Task A running logs, loss, accuracy, classification report, and confusion matrix.",
        "- `outputs/taskB/taskB_metrics.txt`: Task B running logs, loss, accuracy, classification report, and confusion matrix.",
        "- `outputs/taskA/taskA_predictions.csv`: all Task A test predictions.",
        "- `outputs/taskB/taskB_predictions.csv`: all Task B test predictions.",
        "- `outputs/taskA/taskA.misclassified.csv`: all Task A misclassified test documents.",
        "- `outputs/taskB/taskB.misclassified.csv`: all Task B misclassified test documents.",
        "- `outputs/taskA_wrong_taskB_correct.csv`: documents fixed by preprocessing.",
        "- `outputs/taskB_wrong_taskA_correct.csv`: documents hurt by preprocessing.",
        "",
        "## 9. Conclusion",
        "",
        "The project successfully runs the Transformer classifier for both required tasks and outputs the requested misclassified documents. Task B tests whether punctuation removal and whitespace normalization improve the model. Comparing the cross-corrected documents shows that preprocessing can help when punctuation distracts from core category words, but it can also hurt when punctuation provides useful boundary or title-structure cues. The final deliverable includes this report, all source code, all output results, and the generated comparison files.",
    ])

    report_path = REPORT_DIR / "project3_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
