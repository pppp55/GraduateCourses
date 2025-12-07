import os
import pickle
import json
import time
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")

    # Model and Data paths
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base', help='The name of the pretrained model to use.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored.')
    parser.add_argument('--output_dir', type=str, default='out-instruction-tuning', help='Directory to save the fine-tuned model.')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW optimizer beta2.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation.')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients.')

    # Logging and Evaluation
    parser.add_argument('--log_interval', type=int, default=10, help='Log training loss every N steps.')
    # parser.add_argument('--valid_interval', type=int, default=10, help='Run validation every N steps.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Run validation every N steps.')

    # Optimization method
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd', 'lora'], help='Optimization method to use.')

    parser.add_argument('--lora_rank', type=int, default=8, help='The rank of the LoRA matrices.')

    return parser.parse_args()

class TokenizedDataset(Dataset):
    """A simple dataset class to load tokenized IDs from a pickle file."""
    def __init__(self, pickle_file_path):
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(
                f"Pickle file not found at {pickle_file_path}. "
                "Please run the data preparation script first."
            )
        with open(pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} examples from {pickle_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmartDataCollator:
    """
    Pads sequences to the max length in a batch and creates labels.
    Labels are -100 for pad tokens.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }

def main():
    start_time = time.time()
    args = get_args()

    # Derived paths
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'val.pkl')
    output_dir = os.path.join(script_dir, args.output_dir)


    print(f"Loading model and tokenizer from {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)

    # Memory usage before model loading
    if device == "cuda":
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1024**3:.2f} GB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad_token to eos_token")

    collate_fn = SmartDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = TokenizedDataset(train_data_path)
    val_dataset = TokenizedDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Setting up optimizer: {args.optimization_method}")

    # TODO: Apply different optimizer
    if args.optimization_method == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimization_method == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimization_method == "lora":
        print(f"Setting up LoRA with rank={args.lora_rank}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Apply Lora to all possible modules
        )
        model = get_peft_model(model, lora_config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:
        raise ValueError(f"Unknown optimization_method: {args.optimization_method}")

    print("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lists to store loss values for plotting
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accumulation_steps
            loss.backward()
            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    train_loss_value = loss.item() * args.grad_accumulation_steps
                    print(f"Step {global_step}: Train Loss = {train_loss_value:.4f}")
                if global_step % args.eval_interval == 0:
                    train_steps.append(global_step)
                    train_losses.append(loss.item())
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            with torch.autocast(device_type=device, dtype=dtype):
                                val_outputs = model(**val_batch)
                                val_loss = val_outputs.loss
                            total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_steps.append(global_step)
                    val_losses.append(avg_val_loss)
                    print(f"Step {global_step}: Validation Loss = {avg_val_loss:.4f}")
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"  -> New best validation loss! Saving model to {output_dir}")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    model.train()

    print("\nTraining finished. Running one final evaluation...")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Final Validation"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Final Validation Loss = {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        print(f"  -> Final model was the best! Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print(f"  -> An earlier checkpoint was better (Val Loss: {best_val_loss:.4f}). Final model not saved.")

    # Save training history for later aggregation
    plot_data = {
        'optimization_method': args.optimization_method,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'lora_rank': args.lora_rank if args.optimization_method == 'lora' else None,
        'train_steps': train_steps,
        'train_losses': train_losses,
        'val_steps': val_steps,
        'val_losses': val_losses
    }
    
    # Save plot data
    exp_output_dir = os.path.join(output_dir, 'plot')
    os.makedirs(exp_output_dir, exist_ok=True)
    exp_filename = f"{args.optimization_method}_epochs{args.num_epochs}_lr{args.learning_rate}_rank{args.lora_rank}.json"
    exp_path = os.path.join(exp_output_dir, exp_filename)
    
    with open(exp_path, 'w') as f:
        json.dump(plot_data, f)
    
    print(f"Plot data saved to {exp_path}")

    # Memory usage after training
    if device == "cuda":
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Final GPU memory: {final_memory / 1024**3:.2f} GB")
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")
        print(f"Memory used during training: {(peak_memory - initial_memory) / 1024**3:.2f} GB")

    # Calculate and print training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    print(f"\nProcess complete. Best model is saved in {output_dir}")

if __name__ == '__main__':
    main()