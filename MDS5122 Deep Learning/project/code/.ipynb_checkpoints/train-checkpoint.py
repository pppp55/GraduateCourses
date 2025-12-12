import argparse
import os
from contextlib import nullcontext
from itertools import cycle
from typing import List

import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline

from dataset import VideoFrameTrainDataset


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for InstructPix2Pix')
    parser.add_argument('--data-root', type=str, default='../data/processed_dataset')
    parser.add_argument('--output-dir', type=str, default='./pix2pix-lora')
    parser.add_argument('--model-id', type=str, default='timbrooks/instruct-pix2pix')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-train-steps', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lora-rank', type=int, default=4)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    parser.add_argument('--prompt-suffix', type=str, default='Predict the immediate next frame.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def format_prompts(batch, suffix: str) -> List[str]:
    prompts = []
    for inst, tmpl in zip(batch['instruction'], batch['template']):
        prompt = f"Instruction: {inst}. Template: {tmpl}. {suffix}".strip()
        prompts.append(prompt)
    return prompts


def encode_prompts(pipe, prompts, device, dtype):
    text_inputs = pipe.tokenizer(
        prompts,
        padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device) if hasattr(pipe.text_encoder.config, 'use_attention_mask') and pipe.text_encoder.config.use_attention_mask else None
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
    return text_embeddings.to(device=device, dtype=dtype)


def encode_latents(vae, images):
    latents = vae.encode(images * 2.0 - 1.0).latent_dist.sample()
    return latents * vae.config.scaling_factor


def setup_lora(unet, rank, alpha, dropout):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError('Please install `peft` to run LoRA fine-tuning: pip install peft') from exc

    lora_kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias='none',
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
    )

    if hasattr(TaskType, 'UNET'):
        lora_kwargs['task_type'] = TaskType.UNET

    lora_config = LoraConfig(**lora_kwargs)

    unet.requires_grad_(False)
    unet.add_adapter(lora_config)
    unet.enable_lora()

    return [p for p in unet.parameters() if p.requires_grad]


def save_lora(pipe, output_dir, step):
    save_path = os.path.join(output_dir, f'lora-step-{step}')
    pipe.unet.save_lora_adapter(save_path)
    print(f'[Checkpoint] Saved LoRA weights to {save_path}')


def save_loss_curve(steps, losses, output_dir):
    if not steps:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.title('LoRA Training Loss over Steps')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    output_path = os.path.join(output_dir, 'training_loss.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'[Plot] Saved loss curve to {output_path}')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    weight_dtype = dtype_map[args.precision]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id,
        # torch_dtype=weight_dtype,
        dtype=weight_dtype,
        safety_checker=None,
    ).to(device)

    pipe.vae = pipe.vae.to(dtype=torch.float32)  # keep VAE in fp32 to avoid latent NaNs

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    trainable_params = setup_lora(pipe.unet, args.lora_rank, args.lora_alpha, args.lora_dropout)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_train_steps)

    transform = build_transform(args.image_size)
    dataset = VideoFrameTrainDataset(args.data_root, transform=transform, return_tensor=True)
    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty. Please run preprocessing first.')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    data_iter = cycle(dataloader)

    use_autocast = device.type == 'cuda' and args.precision in {'fp16', 'bf16'}
    autocast_dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16

    pipe.unet.train()
    progress = tqdm(range(1, args.max_train_steps + 1), desc='Training', ncols=120)
    step_history = []
    loss_history = []

    for step in progress:
        batch = next(data_iter)

        prompts = format_prompts(batch, args.prompt_suffix)
        prompt_embeds = encode_prompts(pipe, prompts, device, pipe.unet.dtype)

        input_images = batch['input'].to(device=device, dtype=pipe.vae.dtype)
        target_images = batch['target'].to(device=device, dtype=pipe.vae.dtype)

        with torch.no_grad():
            input_latents = encode_latents(pipe.vae, input_images)
            target_latents = encode_latents(pipe.vae, target_images)

        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noise_scheduler.scale_model_input(noisy_latents, timesteps), input_latents], dim=1)

        with (amp.autocast('cuda', dtype=autocast_dtype) if use_autocast else nullcontext()):
            noise_pred = pipe.unet(model_input, timesteps, encoder_hidden_states=prompt_embeds).sample
            loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        progress.set_postfix(loss=f'{loss.item():.4f}')
        step_history.append(step)
        loss_history.append(loss.item())

        if args.save_every > 0 and step % args.save_every == 0:
            save_lora(pipe, args.output_dir, step)

    save_lora(pipe, args.output_dir, args.max_train_steps)
    save_loss_curve(step_history, loss_history, script_dir)


if __name__ == '__main__':
    main()