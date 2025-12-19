#!/usr/bin/env bash

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/model_cache/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

SUMMARY_FILE="../grid_summary.txt"
: > "${SUMMARY_FILE}"

# echo "Starting grid run..." | tee -a "${SUMMARY_FILE}"

models=("Qwen/Qwen2.5-Math-1.5B" "Qwen/Qwen2.5-Math-1.5B-Instruct")
temperatures=(0.6 1.0 1.2)
datasets=("math" "amc" "aime")

for model in "${models[@]}"; do
  model_slug="${model##*/}"
  for temp in "${temperatures[@]}"; do
    for dataset in "${datasets[@]}"; do
      if [[ "${dataset}" == "math" ]]; then
        rollout_n=16
      else
        rollout_n=64
      fi

      infer_out="outputs/${dataset}_${model_slug}_t${temp}.jsonl"
      eval_out="outputs/${dataset}_${model_slug}_t${temp}_eval.jsonl"

      echo "\n==== Running ${model} | dataset=${dataset} | temperature=${temp} ====" | tee -a "${SUMMARY_FILE}"

      python inference.py \
        --model "${model}" \
        --dataset "${dataset}" \
        --dp-size 1 \
        --batch-size 16 \
        --rollout-n "${rollout_n}" \
        --temperature "${temp}" \
        --top-p 0.9 \
        --output_file "${infer_out}"

    #   echo "-- Evaluation output: ${eval_out}" | tee -a "${SUMMARY_FILE}"
      eval_log=$(python evaluate.py --input_file "${infer_out}" --output_file "${eval_out}")
      echo "${eval_log}" | tee -a "${SUMMARY_FILE}"
    done
  done
done

# echo "\nAll runs complete. Summary saved to ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
