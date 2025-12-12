import argparse
import os
from typing import List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from diffusers import StableDiffusionInstructPix2PixPipeline

from dataset import VideoFrameEvalDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LoRA finetuned InstructPix2Pix on held-out frames')
    parser.add_argument('--data-root', type=str, default='../data/processed_dataset')
    parser.add_argument('--model-id', type=str, default='timbrooks/instruct-pix2pix')
    parser.add_argument('--lora-weights', type=str, default='', help='Path to saved LoRA adapter directory')
    parser.add_argument('--lora-weight-name', type=str, default='pytorch_lora_weights.safetensors', help='Filename of the LoRA weights inside the adapter directory')
    parser.add_argument('--lora-scale', type=float, default=1.0, help='Scaling factor applied to LoRA weights')
    parser.add_argument('--image-size', type=int, default=128, help='Resolution for dataset frames')
    parser.add_argument('--pipe-resolution', type=int, default=512, help='Resolution passed to pix2pix pipeline input image')
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    parser.add_argument('--vae-fp32', action='store_true', help='Keep VAE in float32 for numerical stability')
    parser.add_argument('--use-lora', action=argparse.BooleanOptionalAction, default=True, help='Enable LoRA adapters during evaluation (use --no-use-lora to disable)')
    parser.add_argument('--prompt-suffix', type=str, default='Predict the immediate next frame.')
    parser.add_argument('--num-inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--max-samples', type=int, default=-1, help='Limit number of evaluation samples (use -1 for all)')
    parser.add_argument('--eval-batch-size', type=int, default=4, help='Batch size for evaluation dataloader')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of dataloader worker processes')
    parser.add_argument('--output-dir', type=str, default='./eval_outputs')
    parser.add_argument('--save-images', action='store_true', help='Persist input/prediction/target triplets for inspection')
    return parser.parse_args()


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def format_prompt(sample: Dict, suffix: str) -> str:
    instruction = sample['instruction']
    return f"This is a current frame of a 12 fps video of {instruction}, and please predict the next frame."


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).cpu().numpy()
    return (array * 255.0).astype(np.uint8)


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(gt, -1, window)
    mu2 = cv2.filter2D(pred, -1, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(gt * gt, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(pred * pred, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(gt * pred, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def save_triplet(output_dir: str, base_name: str, input_img: Image.Image, pred_img: Image.Image, target_img: Image.Image):
    os.makedirs(output_dir, exist_ok=True)
    input_img.save(os.path.join(output_dir, f'{base_name}_input.jpg'))
    pred_img.save(os.path.join(output_dir, f'{base_name}_pred.jpg'))
    target_img.save(os.path.join(output_dir, f'{base_name}_target.jpg'))


def save_ssim_psnr_scatter(ssim_scores: List[float], psnr_scores: List[float], output_path: str):
    if not ssim_scores or not psnr_scores:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(ssim_scores, psnr_scores, alpha=0.6, edgecolors='none')
    plt.xlabel('SSIM')
    plt.ylabel('PSNR (dB)')
    plt.title('SSIM vs PSNR')
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    weight_dtype = dtype_map[args.precision]

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    vae_dtype = torch.float32 if args.vae_fp32 else weight_dtype
    pipe.vae = pipe.vae.to(dtype=vae_dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    if args.use_lora:
        if not args.lora_weights:
            raise ValueError('LoRA weights path is required when --use-lora is enabled.')
        weight_path = os.path.join(args.lora_weights, args.lora_weight_name)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'LoRA weight file not found: {weight_path}')
        state_dict = load_file(weight_path)
        # Preface keys with 'unet.' to match diffusers expectations for adapter loading
        state_dict = {f'unet.{k}': v for k, v in state_dict.items()}
        adapter_name = 'eval_lora'
        pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
        if hasattr(pipe, 'set_adapters'):
            pipe.set_adapters([adapter_name])
        if hasattr(pipe, 'set_lora_scale'):
            pipe.set_lora_scale(args.lora_scale)
        elif hasattr(pipe.unet, 'set_lora_scale'):
            pipe.unet.set_lora_scale(args.lora_scale)
    else:
        if args.lora_weights:
            print('[Eval] use_lora is False; skipping provided LoRA weights and evaluating the base model.')

    transform = build_transform(args.image_size)
    dataset = VideoFrameEvalDataset(args.data_root, transform=transform, return_tensor=True)
    if len(dataset) == 0:
        raise RuntimeError('Evaluation dataset is empty. Please run preprocessing first.')

    to_pil = transforms.ToPILImage()
    total_samples = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    ssim_scores = []
    psnr_scores = []
    image_output_dir = os.path.join(args.output_dir, 'use_lora' if args.use_lora else 'origin')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.join(image_output_dir, 'pic'), exist_ok=True)

    progress = tqdm(total=total_samples, desc='Evaluating', ncols=120)
    processed = 0
    for batch in dataloader:
        if processed >= total_samples:
            break

        batch_size = batch['input'].size(0)
        remaining = total_samples - processed
        if batch_size > remaining:
            for key in batch:
                if isinstance(batch[key], list):
                    batch[key] = batch[key][:remaining]
                else:
                    batch[key] = batch[key][:remaining]
            batch_size = remaining

        prompts = [
            format_prompt({'instruction': inst, 'template': tmpl}, args.prompt_suffix)
            for inst, tmpl in zip(batch['instruction'], batch['template'])
        ]

        input_images = [
            to_pil(tensor).resize((args.pipe_resolution, args.pipe_resolution), Image.BICUBIC)
            for tensor in batch['input']
        ]
        target_pils = [to_pil(tensor) for tensor in batch['target']]

        with torch.inference_mode():
            result = pipe(
                prompt=prompts,
                image=input_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        pred_images = result.images

        for i in range(batch_size):
            target_tensor = batch['target'][i]
            target_np = tensor_to_uint8(target_tensor)
            resized_pred = pred_images[i].resize(target_pils[i].size, Image.BICUBIC)
            pred_np = np.array(resized_pred)

            ssim_scores.append(compute_ssim(target_np, pred_np))
            psnr_scores.append(compute_psnr(target_np, pred_np))

            if args.save_images:
                base_name = f"{batch['folder'][i]}_{batch['video_id'][i]}_idx{processed + i:04d}"
                save_triplet(
                    os.path.join(image_output_dir, 'pic'),
                    base_name,
                    input_images[i].resize(target_pils[i].size, Image.BICUBIC),
                    pred_images[i],
                    target_pils[i],
                )

        processed += batch_size
        progress.update(batch_size)
        progress.set_postfix(done=f'{processed}/{total_samples}')

    avg_ssim = float(np.mean(ssim_scores))
    avg_psnr = float(np.mean(psnr_scores))

    scatter_path = os.path.join(image_output_dir, 'ssim_psnr_scatter.png')
    save_ssim_psnr_scatter(ssim_scores, psnr_scores, scatter_path)

    print('=== Evaluation Summary ===')
    print(f'Samples evaluated : {total_samples}')
    print(f'Average SSIM      : {avg_ssim:.4f}')
    print(f'Average PSNR (dB) : {avg_psnr:.2f}')


if __name__ == '__main__':
    main()
