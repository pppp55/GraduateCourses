#!/usr/bin/env python
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from depth_model import ResNet50DepthModel
from metrics import abs_rel_metric, solve_scale_shift
from scannet_dataset import ScanNetDepthDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ScanNet depth evaluation (aligned AbsRel only).")
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True, help="Scenes to evaluate. scannetv2_val.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "foundation_vggt", "foundation_da3"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Required for baseline mode.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_jsonl", type=str, default=None, help="Optional per-batch abs_rel log.")
    parser.add_argument("--hf_home", type=str, default=None)
    parser.add_argument("--hf_hub_cache", type=str, default=None)
    parser.add_argument(
        "--vggt_model_dir",
        type=str,
        default=None,
        help="Local VGGT pretrained directory or HF local snapshot path.",
    )
    parser.add_argument(
        "--vggt_resolution",
        type=int,
        default=518,
        help="Square resolution used by VGGT forward path (must be divisible by 14).",
    )
    parser.add_argument(
        "--da3_model_dir",
        type=str,
        default=None,
        help="Local DA3 pretrained directory or HF repo id.",
    )
    parser.add_argument(
        "--da3_process_res",
        type=int,
        default=504,
        help="DA3 inference process resolution.",
    )
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        help="DA3 resize strategy passed to inference().",
    )
    return parser.parse_args()


class FoundationVGGTProxy(nn.Module):
    """
    Placeholder foundation-style depth model for evaluation protocol integration.
    If no external foundation checkpoint is provided, this runs an ImageNet-pretrained
    ResNet50 depth backbone to keep the pipeline executable.
    """

    def __init__(self, model_dir: str, resolution: int):
        super().__init__()
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt"))
        from vggt.models.vggt import VGGT

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        if resolution % 14 != 0:
            raise ValueError(f"--vggt_resolution must be divisible by 14, got {resolution}")
        self.resolution = resolution

        self.model = VGGT.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VGGT expects square inputs and internally assumes patch-size compatibility.
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

        if self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=self.dtype):
                images = x.unsqueeze(0)  # [1, B, 3, H, W]
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
                depth_map, _ = self.model.depth_head(aggregated_tokens_list, images, patch_start_idx)
        else:
            images = x.unsqueeze(0)
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
            depth_map, _ = self.model.depth_head(aggregated_tokens_list, images, patch_start_idx)

        # depth_map can be [B, S, 1, H, W] or [B, S, H, W, 1].
        if depth_map.ndim != 5:
            raise RuntimeError(f"Unexpected VGGT depth_map shape: {tuple(depth_map.shape)}")

        if depth_map.shape[2] == 1:
            pred = depth_map[:, :, 0, :, :]
        elif depth_map.shape[-1] == 1:
            pred = depth_map[..., 0]
        else:
            raise RuntimeError(f"Cannot infer channel dim for VGGT depth_map: {tuple(depth_map.shape)}")

        # Merge VGGT batch/sequence axes into evaluation batch axis.
        pred = pred.reshape(-1, 1, pred.shape[-2], pred.shape[-1])
        if pred.shape[-2:] != (orig_h, orig_w):
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return torch.clamp(pred, min=1e-3)


class FoundationDA3Proxy(nn.Module):
    """
    Another placeholder foundation-style model.
    Uses the same architecture but disables pretrained backbone to provide a distinct
    comparison point under the same evaluation protocol.
    """

    def __init__(self, model_dir: str, process_res: int = 504, process_res_method: str = "upper_bound_resize"):
        super().__init__()
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Depth-Anything-3"))
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(model_dir)
        self.model = self.model.to(device=self.device)
        self.model.eval()

        self.process_res = int(process_res)
        self.process_res_method = process_res_method

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def _tensor_batch_to_rgb_list(self, x: torch.Tensor) -> List[object]:
        # Convert normalized tensor batch back to uint8 RGB images for DA3 API.
        x_cpu = x.detach().float().cpu()
        x_cpu = torch.clamp(x_cpu * self._std + self._mean, 0.0, 1.0)
        imgs: List[object] = []
        for i in range(x_cpu.shape[0]):
            arr = (x_cpu[i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            imgs.append(arr)
        return imgs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        image_list = self._tensor_batch_to_rgb_list(x)

        prediction = self.model.inference(
            image_list,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )
        depth = prediction.depth

        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth)
        depth = depth.float()

        if depth.ndim != 3:
            raise RuntimeError(f"Unexpected DA3 depth output shape: {tuple(depth.shape)}")

        pred = depth.unsqueeze(1).to(x.device)
        if pred.shape[-2:] != (orig_h, orig_w):
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return torch.clamp(pred, min=1e-3)

def _read_scene_file(path: str) -> List[str]:
    scenes: List[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line)
    return scenes


def _build_eval_dataset(args: argparse.Namespace) -> ScanNetDepthDataset:
    scenes = _read_scene_file(args.split_file)
    print(f"Evaluation scenes: {len(scenes)}")
    return ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=scenes,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
        max_samples=args.max_samples,
    )


@torch.no_grad()
def _evaluate(
    args: argparse.Namespace,
    dataset: ScanNetDepthDataset,
    device: torch.device,
) -> Dict[str, object]:
    if args.mode == "baseline":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode baseline")
        model: nn.Module = ResNet50DepthModel(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            pretrained_backbone=False,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        ckpt_path = str(args.checkpoint)
        train_meta = {
            "trained_in_this_repo": True,
            "training_type": "supervised_baseline",
            "checkpoint": ckpt_path,
        }
    elif args.mode == "foundation_vggt":
        model = FoundationVGGTProxy(
            model_dir=args.vggt_model_dir,
            resolution=args.vggt_resolution,
        ).to(device)
        ckpt_path = None
        train_meta = {
            "trained_in_this_repo": False,
            "training_type": "foundation_pretrained_or_proxy",
            "checkpoint": ckpt_path,
        }
    elif args.mode == "foundation_da3":
        model = FoundationDA3Proxy(
            model_dir=args.da3_model_dir,
            process_res=args.da3_process_res,
            process_res_method=args.da3_process_res_method,
        ).to(device)
        ckpt_path = None
        train_meta = {
            "trained_in_this_repo": False,
            "training_type": "foundation_pretrained_or_proxy",
            "checkpoint": ckpt_path,
            "model_dir": args.da3_model_dir,
        }
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_abs_rel = 0.0
    count = 0
    per_batch: List[Dict[str, object]] = []

    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        pred = model(image)
        pred = solve_scale_shift(pred, depth, valid_mask)

        abs_rel = abs_rel_metric(pred, depth, valid_mask)
        if math.isnan(abs_rel):
            continue
        total_abs_rel += abs_rel
        count += 1
        per_batch.append(
            {
                "step": int(i),
                "abs_rel": float(abs_rel),
                "batch_size": int(image.shape[0]),
                "valid_pixels": int(valid_mask.sum().item()),
            }
        )

        if i % 20 == 0:
            print(f"[{args.mode}] step {i}/{len(loader)} abs_rel={abs_rel:.4f}")

    mean_abs_rel = float("inf") if count == 0 else total_abs_rel / count

    return {
        "mode": args.mode,
        "checkpoint": ckpt_path,
        "training": train_meta,
        "num_batches": int(len(loader)),
        "valid_batches": int(count),
        "abs_rel": float(mean_abs_rel),
        "per_batch": per_batch,
    }


def main() -> None:
    args = _parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_hub_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_hub_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _build_eval_dataset(args)
    metrics = _evaluate(args, dataset, device)
    abs_rel = float(metrics["abs_rel"])


    print("\n=== Evaluation Results ===")
    print(f"{'mode':>8s}: {args.mode}")
    print(f"{'abs_rel':>8s}: {abs_rel:.6f}")
    print(f"{'valid_batches':>12s}: {metrics['valid_batches']}")

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nSaved metrics to: {save_path}")

    if args.save_jsonl is not None:
        save_jsonl = Path(args.save_jsonl)
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with save_jsonl.open("w", encoding="utf-8") as f:
            for row in metrics["per_batch"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved per-batch log to: {save_jsonl}")


if __name__ == "__main__":
    main()
