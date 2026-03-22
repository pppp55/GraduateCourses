#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("Plot training/evaluation results for Task3.")
	parser.add_argument(
		"--outputs_dir",
		type=str,
		default=str(Path(__file__).resolve().parent / "outputs"),
		help="Directory that contains experiment subfolders.",
	)
	parser.add_argument(
		"--pic_dir",
		type=str,
		default=str(Path(__file__).resolve().parent / "outputs" / "pic"),
		help="Directory to save generated figures.",
	)
	parser.add_argument(
		"--max_eval_points",
		type=int,
		default=1500,
		help="Max number of per-batch points per run in line plots to keep images readable.",
	)
	return parser.parse_args()


def _load_json(path: Path) -> Dict:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _save_fig(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


def _collect_runs(outputs_dir: Path) -> Tuple[List[Dict], List[Dict]]:
	train_runs: List[Dict] = []
	eval_runs: List[Dict] = []

	for run_dir in sorted([p for p in outputs_dir.iterdir() if p.is_dir()]):
		train_summary_path = run_dir / "train_summary.json"
		if train_summary_path.exists():
			data = _load_json(train_summary_path)
			data["run_name"] = run_dir.name
			train_runs.append(data)

		eval_candidates = sorted(run_dir.glob("eval*.json"))
		for eval_path in eval_candidates:
			data = _load_json(eval_path)
			data["run_name"] = run_dir.name
			data["eval_file"] = eval_path.name
			eval_runs.append(data)

	return train_runs, eval_runs


def _plot_train_loss_curves(train_runs: List[Dict], pic_dir: Path) -> None:
	if not train_runs:
		return

	plt.figure(figsize=(10, 6))
	for run in train_runs:
		history = run.get("history", [])
		if not history:
			continue
		epochs = [x["epoch"] for x in history]
		losses = [x["train_loss"] for x in history]
		plt.plot(epochs, losses, marker="o", linewidth=1.5, markersize=3, label=run["run_name"])

	plt.xlabel("Epoch")
	plt.ylabel("Train SiLog Loss")
	plt.title("Training Loss Curves")
	plt.grid(alpha=0.25)
	plt.legend(fontsize=8)
	_save_fig(pic_dir / "train_loss_curves.png")


def _plot_lr_curves(train_runs: List[Dict], pic_dir: Path) -> None:
	if not train_runs:
		return

	plt.figure(figsize=(10, 6))
	for run in train_runs:
		history = run.get("history", [])
		if not history:
			continue
		epochs = [x["epoch"] for x in history]
		lrs = [x.get("lr", np.nan) for x in history]
		plt.plot(epochs, lrs, marker=".", linewidth=1.3, label=run["run_name"])

	plt.xlabel("Epoch")
	plt.ylabel("Learning Rate")
	plt.title("Learning Rate Schedules")
	plt.grid(alpha=0.25)
	plt.legend(fontsize=8)
	_save_fig(pic_dir / "lr_curves.png")


def _plot_epoch_time(train_runs: List[Dict], pic_dir: Path) -> None:
	if not train_runs:
		return

	plt.figure(figsize=(10, 6))
	for run in train_runs:
		history = run.get("history", [])
		if not history:
			continue
		epochs = [x["epoch"] for x in history]
		times = [x.get("epoch_time_sec", np.nan) for x in history]
		plt.plot(epochs, times, marker=".", linewidth=1.3, label=run["run_name"])

	plt.xlabel("Epoch")
	plt.ylabel("Epoch Time (sec)")
	plt.title("Training Time Per Epoch")
	plt.grid(alpha=0.25)
	plt.legend(fontsize=8)
	_save_fig(pic_dir / "epoch_time_curves.png")


def _plot_absrel_comparison(eval_runs: List[Dict], pic_dir: Path) -> None:
	if not eval_runs:
		return

	labels = []
	values = []
	for r in eval_runs:
		labels.append(f"{r['run_name']}\n{r.get('mode', 'unknown')}")
		values.append(float(r.get("abs_rel", np.nan)))

	order = np.argsort(values)
	labels = [labels[i] for i in order]
	values = [values[i] for i in order]

	plt.figure(figsize=(12, 6))
	bars = plt.bar(range(len(values)), values)
	for b, v in zip(bars, values):
		plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

	plt.xticks(range(len(values)), labels, rotation=35, ha="right")
	plt.ylabel("AbsRel (lower is better)")
	plt.title("Evaluation AbsRel Comparison")
	plt.grid(axis="y", alpha=0.25)
	_save_fig(pic_dir / "eval_absrel_bar.png")


def _plot_data_scale_effect(train_runs: List[Dict], eval_runs: List[Dict], pic_dir: Path) -> None:
	eval_map = {r["run_name"]: float(r.get("abs_rel", np.nan)) for r in eval_runs if "abs_rel" in r}
	points = []
	for run in train_runs:
		run_name = run["run_name"]
		if run_name not in eval_map:
			continue
		samples = run.get("max_train_samples", None)
		epochs = run.get("epochs", None)
		if samples is None or epochs is None:
			continue
		points.append((run_name, int(samples), int(epochs), eval_map[run_name]))

	if not points:
		return

	points = sorted(points, key=lambda x: x[1])
	sample_vals = [p[1] for p in points]
	absrel_vals = [p[3] for p in points]
	labels = [f"{p[0]}\nE{p[2]}-N{p[1]}" for p in points]

	plt.figure(figsize=(10, 6))
	plt.plot(sample_vals, absrel_vals, marker="o", linewidth=2)
	for x, y, lbl in zip(sample_vals, absrel_vals, labels):
		plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)

	plt.xscale("log")
	plt.xlabel("Max Training Samples (log scale)")
	plt.ylabel("Validation AbsRel")
	plt.title("Data Scale Effect: Samples vs AbsRel")
	plt.grid(alpha=0.25)
	_save_fig(pic_dir / "data_scale_effect.png")


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
	if len(x) <= max_points:
		return x, y
	idx = np.linspace(0, len(x) - 1, max_points).astype(int)
	return x[idx], y[idx]


def _plot_eval_batch_curves(eval_runs: List[Dict], pic_dir: Path, max_eval_points: int) -> None:
	runs_with_batch = [r for r in eval_runs if isinstance(r.get("per_batch", None), list) and len(r["per_batch"]) > 0]
	if not runs_with_batch:
		return

	plt.figure(figsize=(11, 6))
	for r in runs_with_batch:
		batch = r["per_batch"]
		steps = np.asarray([int(x["step"]) for x in batch])
		vals = np.asarray([float(x["abs_rel"]) for x in batch])
		steps, vals = _downsample_xy(steps, vals, max_eval_points)
		plt.plot(steps, vals, linewidth=1.0, alpha=0.9, label=f"{r['run_name']} ({r.get('mode', 'unknown')})")

	plt.xlabel("Evaluation Step")
	plt.ylabel("Per-batch AbsRel")
	plt.title("Per-batch AbsRel Curves")
	plt.grid(alpha=0.25)
	plt.legend(fontsize=7, ncol=2)
	_save_fig(pic_dir / "eval_per_batch_curves.png")


def _plot_eval_hist(eval_runs: List[Dict], pic_dir: Path) -> None:
	runs_with_batch = [r for r in eval_runs if isinstance(r.get("per_batch", None), list) and len(r["per_batch"]) > 0]
	if not runs_with_batch:
		return

	plt.figure(figsize=(10, 6))
	for r in runs_with_batch:
		vals = np.asarray([float(x["abs_rel"]) for x in r["per_batch"]])
		plt.hist(vals, bins=50, alpha=0.35, label=f"{r['run_name']} ({r.get('mode', 'unknown')})", density=True)

	plt.xlabel("Per-batch AbsRel")
	plt.ylabel("Density")
	plt.title("Per-batch AbsRel Distribution")
	plt.grid(alpha=0.25)
	plt.legend(fontsize=7)
	_save_fig(pic_dir / "eval_per_batch_hist.png")


def _write_summary(train_runs: List[Dict], eval_runs: List[Dict], pic_dir: Path) -> None:
	summary = {
		"num_train_runs": len(train_runs),
		"num_eval_runs": len(eval_runs),
		"train_runs": [
			{
				"run_name": r.get("run_name"),
				"epochs": r.get("epochs"),
				"max_train_samples": r.get("max_train_samples"),
				"last_train_loss": (r.get("history") or [{}])[-1].get("train_loss") if r.get("history") else None,
			}
			for r in train_runs
		],
		"eval_runs": [
			{
				"run_name": r.get("run_name"),
				"mode": r.get("mode"),
				"abs_rel": r.get("abs_rel"),
				"num_batches": r.get("num_batches"),
			}
			for r in eval_runs
		],
	}
	out = pic_dir / "summary.json"
	out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
	args = _parse_args()
	outputs_dir = Path(args.outputs_dir)
	pic_dir = Path(args.pic_dir)
	pic_dir.mkdir(parents=True, exist_ok=True)

	train_runs, eval_runs = _collect_runs(outputs_dir)

	_plot_train_loss_curves(train_runs, pic_dir)
	_plot_lr_curves(train_runs, pic_dir)
	_plot_epoch_time(train_runs, pic_dir)
	_plot_absrel_comparison(eval_runs, pic_dir)
	_plot_data_scale_effect(train_runs, eval_runs, pic_dir)
	_plot_eval_batch_curves(eval_runs, pic_dir, max_eval_points=args.max_eval_points)
	_plot_eval_hist(eval_runs, pic_dir)
	_write_summary(train_runs, eval_runs, pic_dir)

	print(f"Collected {len(train_runs)} train runs and {len(eval_runs)} eval runs.")
	print(f"Saved figures and summary into: {pic_dir}")


if __name__ == "__main__":
	main()

