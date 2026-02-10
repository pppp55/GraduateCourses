import argparse
import os
import sys

sys.path.append(os.path.join(".."))
from pathlib import Path
import importlib
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

import torch
from typing import cast

from dataset import EG1800Dataset, FacePreprocessConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Task3: SAM2 zero-shot portrait segmentation demo"
    )
    p.add_argument("--data-root", type=str, default="../EG1800")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--num-examples", type=int, default=5)
    p.add_argument("--img-size", type=int, default=512)

    p.add_argument(
        "--sam2-config", type=str, required=True, help="SAM2 model config YAML path"
    )
    p.add_argument(
        "--sam2-checkpoint", type=str, required=True, help="SAM2 checkpoint (.pt) path"
    )

    p.add_argument(
        "--use-face-box-prompt",
        action="store_true",
        help="Use face detector to create a box prompt",
    )
    p.add_argument("--face-margin", type=float, default=0.45)
    return p.parse_args()


@contextmanager
def _temp_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _try_import_sam2():
    build_mod = importlib.import_module("sam2.build_sam")
    pred_mod = importlib.import_module("sam2.sam2_image_predictor")
    return getattr(build_mod, "build_sam2"), getattr(pred_mod, "SAM2ImagePredictor")


def _import_sam2(local_sam2_repo_root: Path | None):
    # SAM2 repo: https://github.com/facebookresearch/sam2
    # The official API may change; this function keeps imports isolated.
    try:
        return _try_import_sam2()
    except Exception as e:  # pragma: no cover
        msg = str(e)
        if (
            isinstance(e, RuntimeError)
            and "parent directory of the sam2 repository" in msg
            and local_sam2_repo_root is not None
            and (local_sam2_repo_root / "sam2").is_dir()
        ):
            # If SAM2 is cloned into ./sam2, importing from the parent directory triggers
            # an upstream RuntimeError. Temporarily import from the repo root instead.
            try:
                for name in list(sys.modules.keys()):
                    if name == "sam2" or name.startswith("sam2."):
                        sys.modules.pop(name, None)
                with _temp_cwd(local_sam2_repo_root):
                    return _try_import_sam2()
            except Exception as e2:
                e = e2

        raise ImportError(
            "SAM2 is not installed or the API is unavailable.\n"
            "Install per the official repo and ensure it is importable in this environment.\n"
            f"Original error: {type(e).__name__}: {e}"
        )


def _resolve_path(path_str: str, local_sam2_repo_root: Path | None) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    if local_sam2_repo_root is not None:
        candidate = local_sam2_repo_root / p
        if candidate.exists():
            return candidate.resolve()
    return p.resolve()


def main():
    args = parse_args()

    local_sam2_repo_root = Path(__file__).resolve().parent / "sam2"
    if not local_sam2_repo_root.exists():
        local_sam2_repo_root = None

    build_sam2, SAM2ImagePredictor = _import_sam2(local_sam2_repo_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_config_path = _resolve_path(args.sam2_config, local_sam2_repo_root)
    sam2_ckpt_path = _resolve_path(args.sam2_checkpoint, local_sam2_repo_root)

    sam2_model = build_sam2(str(sam2_config_path), str(sam2_ckpt_path), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Use dataset just to get file listing & ground-truth mask (for visualization only)
    face_cfg = FacePreprocessConfig(enabled=False)
    dataset = EG1800Dataset(
        args.data_root,
        split=args.split,
        img_size=args.img_size,
        augment=False,
        face_preprocess=face_cfg,
    )

    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(args.num_examples, 4, figsize=(16, 4 * args.num_examples))
    if args.num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(args.num_examples):
        img_tensor, mask_tensor = dataset[i]
        img_tensor = cast(torch.Tensor, img_tensor)
        mask_tensor = cast(torch.Tensor, mask_tensor)

        # Reconstruct uint8 RGB for SAM2
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        predictor.set_image(img_np)

        box = None
        if args.use_face_box_prompt:
            # Reuse OpenCV via FacePreprocessConfig logic without cropping the model input.
            # Here we just detect face box on the original (denormalized) image.
            try:
                import cv2

                cv2_data = getattr(cv2, "data", None)
                haar_base = (
                    getattr(cv2_data, "haarcascades", "")
                    if cv2_data is not None
                    else ""
                )
                cascade_path = os.path.join(
                    haar_base, "haarcascade_frontalface_default.xml"
                )
                detector = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if faces is not None and len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    # expand box a bit
                    margin = args.face_margin
                    cx, cy = x + w / 2.0, y + h / 2.0
                    side = max(w, h) * (1.0 + margin)
                    x0 = max(0, int(round(cx - side / 2.0)))
                    y0 = max(0, int(round(cy - side / 2.0)))
                    x1 = min(img_np.shape[1], int(round(cx + side / 2.0)))
                    y1 = min(img_np.shape[0], int(round(cy + side / 2.0)))
                    box = np.array([x0, y0, x1, y1], dtype=np.float32)
            except Exception:
                box = None

        if box is None:
            # fallback prompt: center box
            h, w = img_np.shape[:2]
            x0, y0 = int(0.25 * w), int(0.2 * h)
            x1, y1 = int(0.75 * w), int(0.9 * h)
            box = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        best_idx = int(np.argmax(scores))
        pred_mask = masks[best_idx].astype(np.uint8)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_np)
        axes[i, 1].add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                color="lime",
                linewidth=2,
            )
        )
        axes[i, 1].set_title("Prompt (Box)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_tensor.cpu().numpy(), cmap="gray")
        axes[i, 2].set_title("GT Mask")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_mask, cmap="gray")
        axes[i, 3].set_title(f"SAM2 Mask (score={scores[best_idx]:.3f})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = Path("results") / "sam2_examples.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
