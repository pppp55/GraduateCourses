#!/usr/bin/env bash
set -euo pipefail

# Always operate from the deepseek_pipeline directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration (override via env vars if needed)
DATASET_PATH=${DATASET_PATH:-"selected_paths.json"}
RESPONSES_PATH=${RESPONSES_PATH:-"sample_responses.json"}
LAWS_PATH=${LAWS_PATH:-"法律法规.jsonl"}
API_KEY=${DEEPSEEK_API_KEY:-"sk-53aca6f952f34690855460c4537f9623"}
DATASET_LIMIT=${DATASET_LIMIT:-10000}

if [[ -z "$API_KEY" ]]; then
  echo "DEEPSEEK_API_KEY environment variable is required" >&2
  exit 1
fi

echo "[1/3] Running DeepSeek batch generator..."
python deepseek_batch_runner.py \
  --dataset "$DATASET_PATH" \
  --output "$RESPONSES_PATH" \
  --limit "$DATASET_LIMIT" \
  --concurrency 20 \
  --api_key "$API_KEY"

echo "[2/3] Mapping legal citations to law IDs..."
python legal_text_mapper.py \
  --responses "$RESPONSES_PATH" \
  --laws "$LAWS_PATH"

echo "[3/3] Validating responses..."
python validate_responses.py \
  --responses "$RESPONSES_PATH" \
  --laws "$LAWS_PATH"

echo "Pipeline completed successfully."
