#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_FLAT_IMAGE_EXTS = (".jpg", ".jpeg")
_DEPTH_EXTS = (".png", ".npy", ".tiff", ".tif")


@dataclass
class _Sample:
    image_path: Path
    depth_path: Path
    scene: str
    frame_id: str


def _read_split_file(split_file: Optional[str]) -> Optional[List[str]]:
    if split_file is None:
        return None
    scenes: List[str] = []
    for raw in Path(split_file).read_text().splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        scenes.append(item)
    return scenes


def list_scannet_scenes(scannet_root: str) -> List[str]:
    root = Path(scannet_root)
    scene_parent = _resolve_scene_parent(root)
    return [p.name for p in _list_scene_dirs(scene_parent)]


def build_train_val_scenes(
    scannet_root: str,
    split_file: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Return (train_scenes, val_scenes).
    If split files are provided, they take precedence. Otherwise scenes are split
    automatically by val_ratio.
    """
    train_scenes = _read_split_file(split_file) or []
    return train_scenes


def _collect_files(folder: Path, exts: Sequence[str]) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if not folder.is_dir():
        return files
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue
        files[file_path.stem] = file_path
    return files


def _sorted_intersection_ids(a: Dict[str, Path], b: Dict[str, Path]) -> List[str]:
    common = set(a.keys()) & set(b.keys())
    return sorted(common, key=lambda x: int(x) if x.isdigit() else x)


def _guess_scene_dirs(scannet_root: Path, scenes: Optional[Sequence[str]]) -> List[Path]:
    scene_parent = _resolve_scene_parent(scannet_root)

    if scenes is None:
        return _list_scene_dirs(scene_parent)

    scene_dirs: List[Path] = []
    for scene in scenes:
        scene_dir = scene_parent / scene
        if scene_dir.is_dir():
            scene_dirs.append(scene_dir)
    return scene_dirs


def _list_scene_dirs(scene_parent: Path) -> List[Path]:
    scene_dirs = sorted(
        [p for p in scene_parent.iterdir() if p.is_dir() and p.name.startswith("scene")]
    )
    if scene_dirs:
        return scene_dirs
    return sorted([p for p in scene_parent.iterdir() if p.is_dir()])


def _resolve_scene_parent(scannet_root: Path) -> Path:
    if not scannet_root.is_dir():
        raise FileNotFoundError(f"Cannot find scannet_root: {scannet_root}")

    scans_dir = scannet_root / "scans"
    if scans_dir.is_dir():
        return scans_dir

    posed_images_dir = scannet_root / "posed_images"
    if posed_images_dir.is_dir():
        return posed_images_dir

    if any(p.is_dir() and p.name.startswith("scene") for p in scannet_root.iterdir()):
        return scannet_root

    raise FileNotFoundError(
        "Cannot find ScanNet scenes. Tried these layouts:\n"
        f"  1) {scannet_root / 'scans'}/sceneXXXX_XX/...\n"
        f"  2) {scannet_root / 'posed_images'}/sceneXXXX_XX/...\n"
        f"  3) {scannet_root}/sceneXXXX_XX/..."
    )


def _load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        if cv2 is not None:
            flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            depth = cv2.imread(str(depth_path), flags)
            if depth is None:
                raise RuntimeError(f"Failed to load depth map: {depth_path}")
            depth = depth.astype(np.float32)
        else:
            depth = imageio.imread(depth_path).astype(np.float32)

        # ScanNet PNG depth is usually in millimeters.
        depth = depth / 1000.0

    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


class ScanNetDepthDataset(Dataset):
    """
    Monocular depth dataset loader for ScanNet.

    Supported layouts:
      <scannet_root>/scans/sceneXXXX_XX/color/*.jpg
      <scannet_root>/scans/sceneXXXX_XX/depth/*.png

    or
      <scannet_root>/posed_images/sceneXXXX_XX/*.jpg
      <scannet_root>/posed_images/sceneXXXX_XX/*.png
    """

    def __init__(
        self,
        scannet_root: str,
        split_file: Optional[str] = None,
        scenes: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (240, 320),
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.scannet_root = Path(scannet_root)
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment = augment
        self.max_samples = max_samples
        self._color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05)

        if split_file is not None and scenes is not None:
            raise ValueError("Only one of `split_file` or `scenes` can be provided.")

        scene_names = list(scenes) if scenes is not None else _read_split_file(split_file)
        self.samples = self._build_samples(scene_names)
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid ScanNet samples were found. "
                "Please verify root/split paths and folder structure."
            )

        print(f"Loaded ScanNetDepthDataset: {len(self.samples)} samples.")

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _build_samples(self, scenes: Optional[Sequence[str]]) -> List[_Sample]:
        samples: List[_Sample] = []
        scene_dirs = _guess_scene_dirs(self.scannet_root, scenes)

        for scene_dir in scene_dirs:
            scene = scene_dir.name
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            if color_dir.is_dir() and depth_dir.is_dir():
                color_map = _collect_files(color_dir, _IMAGE_EXTS)
                depth_map = _collect_files(depth_dir, _DEPTH_EXTS)
            else:
                # Flat layout: sceneXXXX_XX/{00000.jpg,00000.png,00000.txt,...}
                color_map = _collect_files(scene_dir, _FLAT_IMAGE_EXTS)
                depth_map = _collect_files(scene_dir, _DEPTH_EXTS)

            if not color_map or not depth_map:
                continue

            frame_ids = _sorted_intersection_ids(color_map, depth_map)
            for frame_id in frame_ids:
                samples.append(
                    _Sample(
                        image_path=color_map[frame_id],
                        depth_path=depth_map[frame_id],
                        scene=scene,
                        frame_id=frame_id,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        if self.augment:
            image = self._color_jitter(image)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        depth_np = _load_depth(sample.depth_path)

        if self.augment and np.random.rand() < 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            depth_np = np.flip(depth_np, axis=1).copy()

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()

        target_h, target_w = self.image_size
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0)

        valid_mask = (depth_tensor >= self.min_depth) & (depth_tensor <= self.max_depth)
        valid_mask = valid_mask & torch.isfinite(depth_tensor)
        depth_tensor = torch.clamp(depth_tensor, min=0.0, max=self.max_depth)

        image_tensor = (image_tensor - self._mean) / self._std

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "scene": sample.scene,
            "frame_id": sample.frame_id,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
        }
