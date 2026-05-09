import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not paraphrase", 1: "paraphrase"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CSC5010 Project 4 BERT experiments.")
    parser.add_argument("--student-id", default="225040065")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="prajjwal1/bert-mini")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--sst2-max-length", type=int, default=64)
    parser.add_argument("--mrpc-max-length", type=int, default=100)
    parser.add_argument("--task", choices=["all", "sst2", "mrpc"], default="all")
    return parser.parse_args()


def seed_from_student_id(student_id):
    digits = "".join(ch for ch in str(student_id) if ch.isdigit())
    return int(digits[-9:]) if digits else 42


def compute_metrics_for_task(task):
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        metrics = {"accuracy": accuracy_score(labels, preds)}
        if task == "mrpc":
            metrics["f1"] = f1_score(labels, preds)
        return metrics

    return compute


def tokenize_dataset(task, dataset, tokenizer, max_length):
    if task == "sst2":
        return dataset.map(
            lambda examples: tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            ),
            batched=True,
        )
    return dataset.map(
        lambda examples: tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )


def format_for_torch(encoded):
    encoded = encoded.map(lambda examples: {"labels": examples["label"]}, batched=True)
    columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    encoded.set_format(type="torch", columns=columns)
    return encoded


def write_predictions(task, dataset_split, predictions, output_path):
    logits = predictions.predictions
    pred_ids = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    labels = np.array(predictions.label_ids)

    rows = []
    for idx, pred_id in enumerate(pred_ids):
        label = int(labels[idx])
        pred_id = int(pred_id)
        row = {
            "index": idx,
            "true_label": label,
            "true_label_name": LABELS[task][label],
            "predicted_label": pred_id,
            "predicted_label_name": LABELS[task][pred_id],
            "confidence": float(probs[idx][pred_id]),
            "correct": int(label == pred_id),
        }
        if task == "sst2":
            row["sentence"] = dataset_split[idx]["sentence"]
        else:
            row["sentence1"] = dataset_split[idx]["sentence1"]
            row["sentence2"] = dataset_split[idx]["sentence2"]
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def choose_examples(rows, seed):
    rng = random.Random(seed)
    correct_rows = [row for row in rows if int(row["correct"]) == 1]
    wrong_rows = [row for row in rows if int(row["correct"]) == 0]
    chosen = []
    chosen.extend(rng.sample(correct_rows, min(2, len(correct_rows))))
    chosen.extend(rng.sample(wrong_rows, min(2, len(wrong_rows))))
    chosen.sort(key=lambda row: int(row["index"]))
    return chosen


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_task(task, args, tokenizer, seed):
    print(f"\n===== Running {task.upper()} =====")
    output_root = Path(args.output_dir) / task
    dataset = load_dataset("glue", task)
    max_length = args.sst2_max_length if task == "sst2" else args.mrpc_max_length
    encoded = tokenize_dataset(task, dataset, tokenizer, max_length)
    encoded = format_for_torch(encoded)

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2, return_dict=True)
    training_args = TrainingArguments(
        output_dir=str(output_root / "trainer"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1 if task == "sst2" else 0.0,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics_for_task(task),
    )

    train_result = trainer.train()
    final_metrics = trainer.evaluate(encoded["validation"])
    predictions = trainer.predict(encoded["validation"])

    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / f"bert-for-{task}"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    rows = write_predictions(task, dataset["validation"], predictions, output_root / f"{task}_predictions.csv")
    examples = choose_examples(rows, seed + (1 if task == "sst2" else 2))
    write_csv(examples, output_root / f"{task}_examples.csv")

    y_true = [int(row["true_label"]) for row in rows]
    y_pred = [int(row["predicted_label"]) for row in rows]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[LABELS[task][0], LABELS[task][1]],
        digits=4,
        zero_division=0,
    )

    metrics_payload = {
        "task": task,
        "student_id": args.student_id,
        "seed": seed,
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "train_runtime": train_result.metrics,
        "final_eval": final_metrics,
        "prediction_count": len(rows),
        "correct_count": int(sum(int(row["correct"]) for row in rows)),
        "incorrect_count": int(sum(1 - int(row["correct"]) for row in rows)),
    }

    with (output_root / f"{task}_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write("\n\nSelected practical examples:\n")
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics_payload["final_eval"], ensure_ascii=False, indent=2))
    return metrics_payload


def main():
    args = parse_args()
    seed = seed_from_student_id(args.student_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    tasks = ["mrpc", "sst2"] if args.task == "all" else [args.task]
    summaries = []
    for task in tasks:
        summaries.append(run_task(task, args, tokenizer, seed))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "student_id": args.student_id,
                "seed": seed,
                "tasks": tasks,
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
