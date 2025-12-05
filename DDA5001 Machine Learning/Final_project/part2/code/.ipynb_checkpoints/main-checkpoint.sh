cd /root/autodl-tmp/DDA5001-25Fall-main/p2/src
ls

export HF_ENDPOINT=https://hf-mirror.com

# python prepare.py

# lora
# python finetune.py --optimization_method "lora" --num_epochs 1 --learning_rate 2e-5 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python finetune.py --optimization_method "lora" --num_epochs 1 --learning_rate 2e-5 --lora_rank 16 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

python finetune.py --optimization_method "lora" --num_epochs 1 --learning_rate 2e-5 --lora_rank 4 --output_dir "saves/lora-tuned"
python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# sgd
python finetune.py --optimization_method "sgd" --num_epochs 1 --learning_rate 1e-3 --lora_rank 8 --output_dir "saves/lora-tuned"
python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python finetune.py --optimization_method "sgd" --num_epochs 1 --learning_rate 5e-3 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python finetune.py --optimization_method "sgd" --num_epochs 1 --learning_rate 3e-4 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# # adam
# python finetune.py --optimization_method "adam" --num_epochs 1 --learning_rate 2e-5 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python finetune.py --optimization_method "adam" --num_epochs 1 --learning_rate 1e-5 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python finetune.py --optimization_method "adam" --num_epochs 1 --learning_rate 5e-6 --lora_rank 8 --output_dir "saves/lora-tuned"
# python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"
# python evaluate.py --input_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug_evaled.jsonl"

# python /root/autodl-tmp/DDA5001-25Fall-main/p2/src/plot.py