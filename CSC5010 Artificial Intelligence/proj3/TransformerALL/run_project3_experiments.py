# coding: UTF-8
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from models.Transformer import Config, Model
from utils import DatasetIterater, get_time_dif, PAD, UNK


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "THUCNews" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_VOCAB_SIZE = 10000

PUNCT_RE = re.compile(
    r"[\s\t\r\n，。！？、：；“”‘’（）《》〈〉【】\[\]{}()"
    r",.!?:;\-—_/\\@#$%^&*+=~`|<>…·￥]"
)


@dataclass
class Example:
    index: int
    original_text: str
    processed_text: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_text(text: str) -> str:
    cleaned = PUNCT_RE.sub("", text)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    return [char for char in text]


def load_examples(split: str, task: str) -> list[Example]:
    examples: list[Example] = []
    path = DATA_DIR / f"{split}.txt"
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            processed = preprocess_text(text) if task == "taskB" else text
            if not processed:
                processed = text
            examples.append(Example(idx, text, processed, int(label)))
    return examples


def build_vocab(examples: list[Example], max_size: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in tqdm(examples, desc="Building vocabulary"):
        for token in tokenize(example.processed_text):
            counts[token] = counts.get(token, 0) + 1
    vocab_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:max_size]
    vocab = {token: idx for idx, (token, _) in enumerate(vocab_items)}
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    return vocab


def encode_examples(examples: list[Example], vocab: dict[str, int], pad_size: int) -> list[tuple[list[int], int, int]]:
    encoded = []
    for example in tqdm(examples, desc="Encoding dataset"):
        tokens = tokenize(example.processed_text)
        seq_len = len(tokens)
        if len(tokens) < pad_size:
            tokens.extend([PAD] * (pad_size - len(tokens)))
        else:
            tokens = tokens[:pad_size]
            seq_len = pad_size
        token_ids = [vocab.get(token, vocab.get(UNK)) for token in tokens]
        encoded.append((token_ids, example.label, seq_len))
    return encoded


def build_iterator(encoded: list[tuple[list[int], int, int]], config: Config) -> DatasetIterater:
    return DatasetIterater(encoded, config.batch_size, config.device)


def evaluate_model(config: Config, model: Model, data_iter: DatasetIterater):
    model.eval()
    total_loss = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in tqdm(data_iter, desc="Evaluating"):
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            total_loss += float(loss.item())
            labels_np = labels.data.cpu().numpy()
            preds_np = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, preds_np)
    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = total_loss / max(len(data_iter), 1)
    return acc, avg_loss, labels_all, predict_all


def train_one_task(task: str, seed: int, epochs: int) -> dict:
    set_seed(seed)
    config = Config("THUCNews", "random")
    config.num_epochs = epochs
    class_names = config.class_list

    task_dir = OUTPUT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_examples("train", task)
    dev_examples = load_examples("dev", task)
    test_examples = load_examples("test", task)

    vocab = build_vocab(train_examples)
    config.n_vocab = len(vocab)
    train_data = encode_examples(train_examples, vocab, config.pad_size)
    dev_data = encode_examples(dev_examples, vocab, config.pad_size)
    test_data = encode_examples(test_examples, vocab, config.pad_size)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    model = Model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logs: list[str] = []
    start_time = time.time()
    total_batch = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        labels_last = None
        outputs_last = None
        for trains, labels in tqdm(train_iter, desc=f"{task} Epoch {epoch + 1}/{config.num_epochs}"):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            total_batch += 1
            labels_last = labels
            outputs_last = outputs

        if labels_last is not None and outputs_last is not None:
            true = labels_last.data.cpu()
            pred = torch.max(outputs_last.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, pred)
        else:
            train_acc = 0.0

        dev_acc, dev_loss, _, _ = evaluate_model(config, model, dev_iter)
        msg = (
            f"Epoch {epoch + 1}/{config.num_epochs} | batches={total_batch} | "
            f"train_loss={epoch_loss / max(len(train_iter), 1):.4f} | "
            f"last_batch_train_acc={train_acc:.4f} | "
            f"dev_loss={dev_loss:.4f} | dev_acc={dev_acc:.4f} | "
            f"time={get_time_dif(start_time)}"
        )
        print(msg)
        logs.append(msg)

    test_acc, test_loss, labels_all, preds_all = evaluate_model(config, model, test_iter)
    report = metrics.classification_report(
        labels_all,
        preds_all,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    confusion = metrics.confusion_matrix(labels_all, preds_all)

    predictions_path = task_dir / f"{task}_predictions.csv"
    misclassified_path = task_dir / f"{task}.misclassified.csv"
    write_predictions(predictions_path, test_examples, labels_all, preds_all, class_names)
    write_misclassified(misclassified_path, test_examples, labels_all, preds_all, class_names)

    metrics_text = io.StringIO()
    with redirect_stdout(metrics_text):
        print(f"Task: {task}")
        print(f"Preprocessing: {'remove punctuation and normalize whitespace' if task == 'taskB' else 'none'}")
        print(f"Device: {config.device}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Train/dev/test sizes: {len(train_examples)}/{len(dev_examples)}/{len(test_examples)}")
        print("\nEpoch logs:")
        for line in logs:
            print(line)
        print(f"\nTest Loss: {test_loss:.6f}")
        print(f"Test Accuracy: {test_acc:.6f}")
        print("\nPrecision, Recall and F1-Score:")
        print(report)
        print("Confusion Matrix:")
        print(confusion)
    (task_dir / f"{task}_metrics.txt").write_text(metrics_text.getvalue(), encoding="utf-8")

    torch.save(model.state_dict(), task_dir / f"{task}_Transformer.ckpt")
    (task_dir / f"{task}_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "task": task,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "misclassified_count": int(np.sum(labels_all != preds_all)),
        "predictions_path": str(predictions_path),
        "misclassified_path": str(misclassified_path),
        "metrics_path": str(task_dir / f"{task}_metrics.txt"),
    }


def write_predictions(
    path: Path,
    examples: list[Example],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "original_text",
                "processed_text",
                "true_label",
                "true_label_name",
                "predicted_label",
                "predicted_label_name",
                "correct",
            ],
        )
        writer.writeheader()
        for example, true_label, pred_label in zip(examples, labels, preds):
            writer.writerow({
                "index": example.index,
                "original_text": example.original_text,
                "processed_text": example.processed_text,
                "true_label": int(true_label),
                "true_label_name": class_names[int(true_label)],
                "predicted_label": int(pred_label),
                "predicted_label_name": class_names[int(pred_label)],
                "correct": int(true_label == pred_label),
            })


def write_misclassified(
    path: Path,
    examples: list[Example],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "original_text",
                "processed_text",
                "true_label",
                "true_label_name",
                "predicted_label",
                "predicted_label_name",
            ],
        )
        writer.writeheader()
        for example, true_label, pred_label in zip(examples, labels, preds):
            if true_label == pred_label:
                continue
            writer.writerow({
                "index": example.index,
                "original_text": example.original_text,
                "processed_text": example.processed_text,
                "true_label": int(true_label),
                "true_label_name": class_names[int(true_label)],
                "predicted_label": int(pred_label),
                "predicted_label_name": class_names[int(pred_label)],
            })


def compare_tasks() -> dict:
    import pandas as pd

    task_a = pd.read_csv(OUTPUT_DIR / "taskA" / "taskA_predictions.csv")
    task_b = pd.read_csv(OUTPUT_DIR / "taskB" / "taskB_predictions.csv")
    merged = task_a.merge(
        task_b,
        on="index",
        suffixes=("_taskA", "_taskB"),
    )

    a_wrong_b_correct = merged[(merged["correct_taskA"] == 0) & (merged["correct_taskB"] == 1)].copy()
    b_wrong_a_correct = merged[(merged["correct_taskA"] == 1) & (merged["correct_taskB"] == 0)].copy()
    a_wrong_b_correct.to_csv(OUTPUT_DIR / "taskA_wrong_taskB_correct.csv", index=False, encoding="utf-8-sig")
    b_wrong_a_correct.to_csv(OUTPUT_DIR / "taskB_wrong_taskA_correct.csv", index=False, encoding="utf-8-sig")

    return {
        "taskA_wrong_taskB_correct": int(len(a_wrong_b_correct)),
        "taskB_wrong_taskA_correct": int(len(b_wrong_a_correct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSC5010 Project 3 Task A and Task B experiments.")
    parser.add_argument("--student-id", default="225040065")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--tasks", nargs="+", default=["taskA", "taskB"], choices=["taskA", "taskB"])
    args = parser.parse_args()

    seed = int("".join(ch for ch in args.student_id if ch.isdigit()) or "1") % (2**32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task in args.tasks:
        summaries.append(train_one_task(task, seed=seed, epochs=args.epochs))

    comparison = {}
    if set(args.tasks) == {"taskA", "taskB"}:
        comparison = compare_tasks()

    summary = {
        "student_id": args.student_id,
        "seed": seed,
        "epochs": args.epochs,
        "summaries": summaries,
        "comparison": comparison,
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
