import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import EG1800Dataset, FacePreprocessConfig
from model import build_segmentation_model


def predict_and_visualize(model, dataset, device, num_samples=4):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]

            # 预测
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            if isinstance(output, dict):
                output = output["out"]
            pred = torch.argmax(output, dim=1).squeeze().cpu()

            # 反归一化显示图像
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            image_np = np.clip(image_np, 0, 1)

            # 显示
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 1].set_title("Label Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "predictions.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize segmentation predictions")
    p.add_argument("--data-root", type=str, default="./EG1800")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument(
        "--arch",
        type=str,
        default="unet",
        choices=["unet", "fcn_resnet50", "deeplabv3_resnet50"],
    )
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_baseline.pth")
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--face-preprocess", action="store_true")
    p.add_argument("--face-margin", type=float, default=0.45)
    p.add_argument("--noncentered-shift", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    face_cfg = FacePreprocessConfig(
        enabled=args.face_preprocess, margin=args.face_margin
    )
    dataset = EG1800Dataset(
        root_dir=args.data_root,
        split=args.split,
        img_size=args.img_size,
        augment=False,
        face_preprocess=face_cfg,
        simulate_noncentered=args.noncentered_shift,
    )

    model = build_segmentation_model(args.arch, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    predict_and_visualize(model, dataset, device, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
