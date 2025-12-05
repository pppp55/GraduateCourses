### 1. Download the requirements
At `p2/` folder: \
`pip install -r requirements.txt`

### 2. Run the notebook
Run the notebook `main.ipynb`

### 3. Change hyperparameters
In `main.ipynb`, modify the param X of the code: \
`!python finetune.py --optimization_method "XXX" --num_epochs X --learning_rate XX --lora_rank X --output_dir "saves/lora-tuned"` \
And change the rollout settings: \
If using lora, run: `!python rollout.py --model "Qwen/Qwen3-0.6B-Base" --lora_path "saves/lora-tuned" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"` \
If not using lora, run: `!python rollout.py --model "/root/autodl-tmp/DDA5001-25Fall-main/p2/src/saves/lora-tuned/" --output_file "output/qwen3_0.6b_base_nosys_it_lora_debug.jsonl"` \
(Please remind changing the parm of model path)
### 4. Check the test accuracy
The last output of the notebook `main.ipynb` has an accuracy of the model you have trained

### 5. Plot the loss figures
At `p2/` folder: \
`python ./src/plot.py`

### 6. Check the figures
The figures are all in the path `p2/src/saves/lora-tuned\plot\`
