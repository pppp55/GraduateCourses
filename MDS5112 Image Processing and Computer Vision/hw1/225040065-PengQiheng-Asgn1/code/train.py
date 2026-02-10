import argparse
import json
import os
from dataclasses import asdict
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EG1800Dataset, FacePreprocessConfig
from model import build_segmentation_model


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs["out"]

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

    avg_loss = total_loss / len(dataloader)
    pixel_acc = correct_pixels / total_pixels

    return avg_loss, pixel_acc


def dice_loss_from_logits(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # logits: [B, C, H, W], target: [B, H, W] with values 0..C-1
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    union = torch.sum(probs + target_1h, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def build_criterion(
    name: str, ce_weight: float = 1.0, dice_weight: float = 0.5
) -> nn.Module:
    name = name.lower()
    ce = nn.CrossEntropyLoss()
    if name == "ce":
        return ce  # nn.Module is also Callable
    if name in {"ce_dice", "cedice", "ce+dice"}:

        def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return ce_weight * ce(logits, target) + dice_weight * dice_loss_from_logits(
                logits, target
            )

        return _loss  # type: ignore[return-value]
    raise ValueError("loss must be one of: ce, ce_dice")


def plot_curves(metrics: dict, save_path: str):
    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, metrics["train_loss"], label="train")
    axes[0].plot(epochs, metrics["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, metrics["val_loss"], label="val loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].plot(epochs, metrics["val_pixel_acc"], label="val pixel acc")
    axes[2].set_title("Validation Pixel Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task3: Portrait Segmentation Training"
    )
    parser.add_argument("--data-root", type=str, default="./EG1800")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--arch",
        type=str,
        default="unet",
        choices=["unet", "fcn_resnet50", "deeplabv3_resnet50"],
    )
    parser.add_argument("--pretrained-backbone", action="store_true")

    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "ce_dice"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--scheduler", type=str, default="none", choices=["none", "plateau", "cosine"]
    )

    parser.add_argument("--face-preprocess", action="store_true")
    parser.add_argument("--face-margin", type=float, default=0.45)

    parser.add_argument(
        "--compare-face-preprocess",
        action="store_true",
        help="Evaluate on shifted images before/after face-preprocess",
    )
    parser.add_argument(
        "--noncentered-shift-eval",
        action="store_true",
        help="Evaluate on synthetic non-centered images",
    )

    parser.add_argument("--run-name", type=str, default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    face_cfg = FacePreprocessConfig(
        enabled=args.face_preprocess, margin=args.face_margin
    )

    train_dataset = EG1800Dataset(
        root_dir=args.data_root,
        split="train",
        img_size=args.img_size,
        augment=args.augment,
        face_preprocess=face_cfg,
        simulate_noncentered=False,
        seed=args.seed,
    )

    test_dataset = EG1800Dataset(
        root_dir=args.data_root,
        split="test",
        img_size=args.img_size,
        augment=False,
        face_preprocess=face_cfg,
        simulate_noncentered=args.noncentered_shift_eval,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    model = build_segmentation_model(
        arch=args.arch,
        num_classes=num_classes,
        in_channels=3,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    print(f"Arch: {args.arch} | Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = build_criterion(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    plateau_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    epoch_scheduler = None
    if args.scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    elif args.scheduler == "cosine":
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_pixel_acc": [],
        "args": vars(args),
        "face_preprocess": asdict(face_cfg),
    }

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join("checkpoints", f"best_{args.run_name}.pth")
    last_ckpt_path = os.path.join("checkpoints", f"last_{args.run_name}.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        metrics["train_loss"].append(float(train_loss))
        metrics["val_loss"].append(float(val_loss))
        metrics["val_pixel_acc"].append(float(val_acc))

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_acc:.4f} | LR: {lr_now:.2e}"
        )

        if plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        # save last
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics},
            last_ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics},
                best_ckpt_path,
            )
            print(f"Saved best checkpoint: {best_ckpt_path}")

    metrics_path = os.path.join("results", f"metrics_{args.run_name}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    curves_path = os.path.join("results", f"curves_{args.run_name}.png")
    plot_curves(metrics, curves_path)
    print(f"Saved curves: {curves_path}")

    if args.compare_face_preprocess:
        print("\n[Compare] Synthetic non-centered evaluation:")

        shifted_dataset_no_face = EG1800Dataset(
            root_dir=args.data_root,
            split="test",
            img_size=args.img_size,
            augment=False,
            face_preprocess=FacePreprocessConfig(enabled=False),
            simulate_noncentered=True,
            seed=args.seed,
        )
        shifted_dataset_face = EG1800Dataset(
            root_dir=args.data_root,
            split="test",
            img_size=args.img_size,
            augment=False,
            face_preprocess=FacePreprocessConfig(enabled=True, margin=args.face_margin),
            simulate_noncentered=True,
            seed=args.seed,
        )

        shifted_loader_no_face = DataLoader(
            shifted_dataset_no_face,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        shifted_loader_face = DataLoader(
            shifted_dataset_face,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # evaluate current model with/without preprocessing
        # (note: model expects normalized tensor, preprocessing is inside dataset)
        loss_nf, acc_nf = validate(model, shifted_loader_no_face, criterion, device)
        loss_f, acc_f = validate(model, shifted_loader_face, criterion, device)
        print(f"No face-preprocess: loss={loss_nf:.4f}, pixel_acc={acc_nf:.4f}")
        print(f"With face-preprocess: loss={loss_f:.4f}, pixel_acc={acc_f:.4f}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
