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
    parser.add_argument('--prompt-suffix', type=str, default='Predict the immediate next frame.')
    parser.add_argument('--num-inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--max-samples', type=int, default=-1, help='Limit number of evaluation samples (use -1 for all)')
    parser.add_argument('--output-dir', type=str, default='./eval_outputs')
    parser.add_argument('--save-images', action='store_true', help='Persist input/prediction/target triplets for inspection')
    return parser.parse_args()


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def format_prompt(sample: Dict, suffix: str) -> str:
    parts: List[str] = [
        f"Instruction: {sample['instruction']}",
        f"Template: {sample['template']}",
        suffix.strip(),
    ]
    return ' '.join(part for part in parts if part)


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

    if args.lora_weights:
        weight_path = os.path.join(args.lora_weights, args.lora_weight_name)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'LoRA weight file not found: {weight_path}')
        state_dict = load_file(weight_path)
        state_dict = {f'unet.{k}': v for k, v in state_dict.items()}
        adapter_name = 'eval_lora'
        pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
        if hasattr(pipe, 'set_adapters'):
            pipe.set_adapters([adapter_name])
        if hasattr(pipe, 'set_lora_scale'):
            pipe.set_lora_scale(args.lora_scale)
        elif hasattr(pipe.unet, 'set_lora_scale'):
            pipe.unet.set_lora_scale(args.lora_scale)

    transform = build_transform(args.image_size)
    dataset = VideoFrameEvalDataset(args.data_root, transform=transform, return_tensor=True)
    if len(dataset) == 0:
        raise RuntimeError('Evaluation dataset is empty. Please run preprocessing first.')

    to_pil = transforms.ToPILImage()
    total_samples = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)
    metrics = []

    progress = tqdm(range(total_samples), desc='Evaluating', ncols=120)
    for idx in progress:
        sample = dataset[idx]
        prompt = format_prompt(sample, args.prompt_suffix)

        input_tensor = sample['input']
        target_tensor = sample['target']

        input_image = to_pil(input_tensor).resize((args.pipe_resolution, args.pipe_resolution), Image.BICUBIC)
        target_image = to_pil(target_tensor)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=input_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        pred_image = result.images[0].resize(target_image.size, Image.BICUBIC)

        target_np = tensor_to_uint8(target_tensor)
        pred_np = np.array(pred_image)

        ssim_value = compute_ssim(target_np, pred_np)
        psnr_value = compute_psnr(target_np, pred_np)
        metrics.append((ssim_value, psnr_value))

        progress.set_postfix(ssim=f'{ssim_value:.4f}', psnr=f'{psnr_value:.2f}')

        if args.save_images:
            base_name = f"{sample['folder']}_{sample['video_id']}_idx{idx:04d}"
            save_triplet(args.output_dir, base_name, input_image.resize(target_image.size, Image.BICUBIC), pred_image, target_image)

    ssim_scores = [m[0] for m in metrics]
    psnr_scores = [m[1] for m in metrics]
    avg_ssim = float(np.mean(ssim_scores))
    avg_psnr = float(np.mean(psnr_scores))

    print('=== Evaluation Summary ===')
    print(f'Samples evaluated : {total_samples}')
    print(f'Average SSIM      : {avg_ssim:.4f}')
    print(f'Average PSNR (dB) : {avg_psnr:.2f}')


if __name__ == '__main__':
    main()
