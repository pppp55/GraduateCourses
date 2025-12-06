# SPEECH-TOKEN ALIGNMENT MODEL

## Project Objective
This project implements an end-to-end pipeline for training a speech-text alignment model that predicts discrete speech tokens from textual input and acoustic features. The system leverages CosyVoice's language model capabilities combined with Whisper-derived acoustic representations to generate high-quality speech token sequences that can be used for controllable text-to-speech synthesis and speech generation research.

## Overview
This repository provides a complete workflow for:
1. Preprocessing LibriSpeech audio/text pairs
2. Extracting text embeddings and acoustic features
3. Converting audio to discrete speech tokens using SpeechTokenizer
4. Training a lightweight alignment model on top of frozen CosyVoice LLM

The trained model learns to predict speech tokens conditioned on both text semantics and acoustic context, enabling applications in speech synthesis, voice conversion, and controllable speech generation.

## Hardware & Software Environment
- **CPU**: 16 vCPU Intel(R) Xeon(R) Gold 6459C
- **GPU**: RTX 5090
- **Memory**: 90 GB
- **Python**: 3.10

## 1. Environment Setup

### Create and activate conda environment:
```bash
conda create -n speech-token python=3.10 -y
conda activate speech-token
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Required external models:
```bash
# Download CosyVoice model
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')"

# Download SpeechTokenizer ONNX model
# (Ensure speech_tokenizer_v1.onnx is available in CosyVoice-300M directory)
```

## 2. Data Preparation Pipeline

### Step 1: Preprocess LibriSpeech dataset
```bash
python scripts/preprocess_librispeech.py \
  --librispeech_dir ./data/LibriSpeech \
  --output_dir ./data \
  --target_sr 16000 \
  --max_samples 5000
```
This script:
- Resamples all audio to 16kHz mono WAV format
- Generates JSONL manifests with utterance IDs, text, and audio paths
- Limits to 5,000 samples per subset by default (adjustable)

### Step 2: Extract text embeddings and Whisper features
```bash
python scripts/utt2text_and_feature.py \
  --jsonl ./data/train-clean-100.jsonl \
  --model_dir ./pretrained_models/CosyVoice-300M \
  --output_text ./data/text_emb.pt \
  --output_whisper ./data/speech_feat.pt \
  --max_duration 30.0
```
This script:
- Uses CosyVoice to extract text embeddings from transcripts
- Uses Whisper-large-v3 to extract mid-layer and final-layer encoder features
- Saves features as PyTorch tensors for efficient loading during training

### Step 3: Extract speech tokens (S3 tokens)
```bash
bash s3.sh
```
This script:
- Creates `wav.scp` listing all audio files
- Runs SpeechTokenizer to convert audio to discrete tokens
- Saves token sequences as `s3_tokens.pt`

## 3. Training & Evaluation

### Train the alignment model
```bash
python src/example_code.py
```

## Customization Options
- **Dataset size**: Adjust `--max_samples` in preprocessing scripts
- **Audio duration**: Change `--max_duration` to process longer audio clips
- **Model architecture**: Modify `SimpleTextSpeechAggregator` in `example_code.py`
- **Training hyperparameters**: Edit constants at the top of `example_code.py`
- **Feature selection**: Choose different Whisper layers by modifying `utt2text_and_feature.py`
