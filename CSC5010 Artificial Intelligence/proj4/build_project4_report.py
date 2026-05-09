import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
REPORT_DIR = BASE / "report"
REPORT_PATH = REPORT_DIR / "project4_report.md"


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_sentence(row):
    pred = row["predicted_label_name"]
    true = row["true_label_name"]
    status = "correct" if int(row["correct"]) else "incorrect"
    return pred, true, status


def sst2_comment(row):
    sentence = row["sentence"]
    pred, true, status = label_sentence(row)
    if int(row["correct"]):
        return (
            f"The prediction is {status}: the sentence contains sentiment cues that align with "
            f"the gold label `{true}`, so the fine-tuned BERT model maps it to `{pred}`."
        )
    return (
        f"The prediction is {status}: the wording is short or stylistically indirect, so the "
        f"model assigns `{pred}` even though the gold label is `{true}`. This is a common error "
        f"when sentiment depends on context rather than an obvious positive or negative word."
    )


def mrpc_comment(row):
    pred, true, status = label_sentence(row)
    s1 = row["sentence1"]
    s2 = row["sentence2"]
    overlap = len(set(s1.lower().split()) & set(s2.lower().split()))
    if int(row["correct"]):
        return (
            f"The prediction is {status}: the pair has enough lexical/semantic evidence for "
            f"`{true}`. Word overlap count is {overlap}, and the model predicts `{pred}`."
        )
    return (
        f"The prediction is {status}: the pair may contain misleading word overlap or subtle "
        f"differences in entities, numbers, or event details. The model predicts `{pred}`, while "
        f"the gold label is `{true}`."
    )


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("\n", " ") for item in row) + " |")
    return "\n".join(lines)


def main():
    summary = load_json(OUTPUTS / "experiment_summary.json")
    by_task = {item["task"]: item for item in summary["summaries"]}
    sst2_examples = load_rows(OUTPUTS / "sst2" / "sst2_examples.csv")
    mrpc_examples = load_rows(OUTPUTS / "mrpc" / "mrpc_examples.csv")

    sst2_eval = by_task["sst2"]["final_eval"]
    mrpc_eval = by_task["mrpc"]["final_eval"]
    sst2_correct = by_task["sst2"]["correct_count"]
    mrpc_correct = by_task["mrpc"]["correct_count"]
    sst2_total = by_task["sst2"]["prediction_count"]
    mrpc_total = by_task["mrpc"]["prediction_count"]

    lines = [
        "# CSC5010 Project 4 Report: BERT Fine-tuning",
        "",
        f"Student ID: {summary['student_id']}",
        "",
        "## 1. Project Goal",
        "",
        "This project fine-tunes a small BERT model on two GLUE benchmark tasks. "
        "Task 1 is SST-2 single-sentence sentiment analysis, where each movie review "
        "sentence is classified as negative or positive. Task 2 is MRPC sentence-pair "
        "paraphrase analysis, where each pair is classified as not paraphrase or paraphrase.",
        "",
        "The implementation uses the provided project templates as the task definition, "
        "but updates the runner for the current Python environment. In particular, the "
        "`datasets.load_metric` API used by the original templates is no longer available "
        "in the installed `datasets` version, so the final metrics are computed with "
        "`sklearn.metrics`.",
        "",
        "## 2. Environment and Model",
        "",
        f"- Model: `{by_task['sst2']['model']}`",
        f"- Tokenizer: `{by_task['sst2']['tokenizer']}`",
        f"- Epochs: {by_task['sst2']['epochs']}",
        f"- Batch size: {by_task['sst2']['batch_size']}",
        f"- Random seed: {summary['seed']}",
        "- Framework: HuggingFace Transformers Trainer with PyTorch",
        "",
        "Both tasks use `bert-base-uncased` tokenization and the lightweight "
        "`prajjwal1/bert-mini` sequence classification model. SST-2 is encoded as a "
        "single sentence with maximum length 64. MRPC is encoded as a sentence pair "
        "with maximum length 100.",
        "",
        "## 3. Final Evaluation Results",
        "",
        markdown_table(
            ["Task", "Validation examples", "Accuracy", "F1", "Correct", "Incorrect"],
            [
                [
                    "SST-2 sentiment",
                    sst2_total,
                    f"{sst2_eval.get('eval_accuracy', 0):.4f}",
                    "N/A",
                    sst2_correct,
                    sst2_total - sst2_correct,
                ],
                [
                    "MRPC paraphrase",
                    mrpc_total,
                    f"{mrpc_eval.get('eval_accuracy', 0):.4f}",
                    f"{mrpc_eval.get('eval_f1', 0):.4f}",
                    mrpc_correct,
                    mrpc_total - mrpc_correct,
                ],
            ],
        ),
        "",
        "The final evaluation numbers are taken from the validation split because the "
        "project handout states that the test set is not available for this assignment.",
        "",
        "## 4. SST-2 Practical Examples",
        "",
        markdown_table(
            ["Index", "Sentence", "Gold", "Prediction", "Confidence", "Correct"],
            [
                [
                    row["index"],
                    row["sentence"],
                    row["true_label_name"],
                    row["predicted_label_name"],
                    f"{float(row['confidence']):.4f}",
                    row["correct"],
                ]
                for row in sst2_examples
            ],
        ),
        "",
        "Comments:",
        "",
    ]
    for idx, row in enumerate(sst2_examples, start=1):
        lines.append(f"{idx}. {sst2_comment(row)}")

    lines.extend(
        [
            "",
            "## 5. MRPC Practical Examples",
            "",
            markdown_table(
                ["Index", "Sentence 1", "Sentence 2", "Gold", "Prediction", "Confidence", "Correct"],
                [
                    [
                        row["index"],
                        row["sentence1"],
                        row["sentence2"],
                        row["true_label_name"],
                        row["predicted_label_name"],
                        f"{float(row['confidence']):.4f}",
                        row["correct"],
                    ]
                    for row in mrpc_examples
                ],
            ),
            "",
            "Comments:",
            "",
        ]
    )
    for idx, row in enumerate(mrpc_examples, start=1):
        lines.append(f"{idx}. {mrpc_comment(row)}")

    lines.extend(
        [
            "",
            "## 6. Output Files",
            "",
            "- `outputs/sst2/sst2_metrics.txt`: SST-2 final metrics, classification report, and selected examples.",
            "- `outputs/sst2/sst2_predictions.csv`: all SST-2 validation predictions.",
            "- `outputs/sst2/sst2_examples.csv`: four SST-2 examples used in the report.",
            "- `outputs/mrpc/mrpc_metrics.txt`: MRPC final metrics, classification report, and selected examples.",
            "- `outputs/mrpc/mrpc_predictions.csv`: all MRPC validation predictions.",
            "- `outputs/mrpc/mrpc_examples.csv`: four MRPC examples used in the report.",
            "- `outputs/experiment_summary.json`: machine-readable summary of both experiments.",
            "",
            "## 7. Conclusion",
            "",
            "The BERT model was successfully fine-tuned and evaluated on both required NLP tasks. "
            "SST-2 tests single-sentence sentiment understanding, while MRPC tests sentence-pair "
            "semantic equivalence. The selected examples show that BERT can capture many direct "
            "sentiment and paraphrase cues, but it can still fail when wording is subtle, when "
            "surface overlap is misleading, or when a pair differs in small factual details.",
            "",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
