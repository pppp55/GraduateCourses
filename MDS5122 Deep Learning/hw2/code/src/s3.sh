#!/usr/bin/env bash

LIBRISPEECH_DIR=$(readlink -f "../data/resampled_audio")
OUT_DIR="../data"
ONNX_PATH="../repo/CosyVoice/pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx"

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX model not found at $ONNX_PATH"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Generating wav.scp..."

find "$LIBRISPEECH_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" \) | sort | while read -r f; do
  fname=$(basename "$f")
  id="${fname%.*}"  
  echo "$id $f" 
done > "$OUT_DIR/wav.scp"

dup_id=$(awk '{print $1}' "$OUT_DIR/wav.scp" | sort | uniq -d | head -n 1)
if [ -n "$dup_id" ]; then
  echo "Error: duplicate utterance id detected in wav.scp -> $dup_id"
  echo "Please ensure resampled audio filenames are unique before continuing."
  exit 1
fi

echo "Running extract_speech_token.py..."

python3 "../repo/CosyVoice/tools/extract_speech_token.py" \
  --dir "$OUT_DIR" \
  --onnx_path "$ONNX_PATH" \
  --num_thread 4  

if [ -f "$OUT_DIR/utt2speech_token.pt" ]; then
    mv "$OUT_DIR/utt2speech_token.pt" "$OUT_DIR/s3_tokens.pt"
    echo "S3 tokens saved to $OUT_DIR/s3_tokens.pt"
else
    echo "Error: Extraction failed, output file not found."
fi