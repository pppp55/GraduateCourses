import json
import os
import matplotlib.pyplot as plt
import glob

def plot_combined_losses():
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    plot_dir = os.path.join(script_dir, "saves", "lora-tuned", "plot")
    json_files = glob.glob(os.path.join(plot_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {plot_dir}")
        return
    
    # Load all experiment data
    experiments = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            experiments.append(data)
    
    # Create training loss plot
    plt.figure(figsize=(12, 8))
    for method in ['lora', 'sgd', 'adam']:
        plt.close()
        for exp in experiments:
            if exp['optimization_method'] != method:
                continue
            lr = exp['learning_rate']
            rank = exp['lora_rank'] if exp['lora_rank'] is not None else 'N/A'

            # Create label with method, learning rate and LoRA rank if applicable
            if method == 'lora':
                label = f"{method.upper()} (lr={lr}, rank={rank})"
            else:
                label = f"{method.upper()} (lr={lr})"

            plt.plot(exp['train_steps'], exp['train_losses'], label=label)

        plt.xlabel('Training Steps')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Steps for Different Configurations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{method}_training_loss_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create validation loss plot
    plt.figure(figsize=(12, 8))
    for exp in experiments:
        method = exp['optimization_method']
        lr = exp['learning_rate']
        rank = exp['lora_rank'] if exp['lora_rank'] is not None else 'N/A'
        
        # Create label with method, learning rate and LoRA rank if applicable
        if method == 'lora':
            label = f"{method.upper()} (lr={lr}, rank={rank})"
        else:
            label = f"{method.upper()} (lr={lr})"
            
        plt.plot(exp['val_steps'], exp['val_losses'], label=label)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Steps for Different Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'validation_loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_combined_losses()