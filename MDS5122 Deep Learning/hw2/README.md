# TASTE Project

## Project Objective
The project demonstrates an end-to-end recipe for transforming LibriSpeech audio/text pairs into CosyVoice-compatible speech tokens and training a speech-text alignment model that can predict tokenized speech sequences from textual cues and Whisper-derived acoustic summaries. This pipeline provides a reproducible foundation for research on speech token modeling, controllable TTS, or downstream speech generation tasks that rely on CosyVoice and SpeechTokenizer representations.

## Overview
This repository trains a CosyVoice-based large language model (LLM) to predict discrete speech tokens ("S3 tokens") from paired text and Whisper encoder features derived from LibriSpeech. The pipeline resamples LibriSpeech audio, extracts CosyVoice text embeddings together with Whisper middle/final-layer representations, converts audio to speech-token sequences via the CosyVoice SpeechTokenizer, and fine-tunes a lightweight aggregation layer on top of the frozen CosyVoice LLM to model the relationship between text and speech tokens. The end goal is to produce high-quality speech token predictions that can be reused for downstream TTS or speech generation research.

## Hardware & Software Environment
- CPU: Intel(R) Xeon(R) Platinum 8470Q
- GPU: NVIDIA RTX 5090 (CUDA 13.0)
- Memory: 90 GB RAM
- Recommended OS: Ubuntu 22.04
- Python: 3.10 (created via Miniconda)

## Repository Layout
```
TASTE_Project
├── checkpoints/                 # Trained model checkpoints (created during training)
├── data/                        # LibriSpeech subsets, resampled audio, intermediate tensors
│   ├── LibriSpeech/
│   ├── resampled_audio/
│   ├── *.jsonl / *.pt artifacts (generated)
├── repo/
│   └── CosyVoice/               # Upstream CosyVoice repo (with submodules + pretrained models)
├── results/                     # Evaluation metrics and sampled predictions
└── src/
    ├── requirement.txt          # Python dependencies for the virtual environment
    ├── preprocess_librispeech.py# Resamples LibriSpeech and writes jsonl manifests
    ├── utt2text_and_feature.py  # Extracts text embeddings + Whisper features
    ├── s3.sh                    # Runs CosyVoice SpeechTokenizer to build S3 tokens
    ├── preprocess scripts outputs (text_emb.pt, speech_feat.pt, s3_tokens.pt)
    └── example_code.py          # Training & evaluation entry point
```

## 1. Environment Setup
1. Install Miniconda if it is not already available.
2. Create and activate a Python 3.10 environment:
   ```bash
   conda create -n taste python=3.10 -y
   conda activate taste
   ```
3. Install dependencies from `src/requirements.txt`:
   ```bash
   pip install -r src/requirements.txt
   ```
   The list includes Torch 2.9.1, torchaudio 2.9.1, CosyVoice dependencies (hydra-core, modelscope, einx, etc.), Whisper, and SpeechTokenizer tooling.

## 2. External Assets
1. **LibriSpeech**: download the required subsets into `data/LibriSpeech`. Only `train-clean-100` and `test-clean` are needed for the provided scripts.
2. **CosyVoice repository**: clone into `repo/` and initialize submodules.
   ```bash
   cd repo
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   git submodule update --init --recursive
   ```
3. **CosyVoice pretrained models**: download the LLM/SpeechTokenizer checkpoints via ModelScope.
   ```bash
   python -c "from modelscope import snapshot_download; \
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M'); \
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')"
   cd pretrained_models/CosyVoice-ttsfrd/
   unzip resource.zip -d .
   pip install ttsfrd_dependency-0.1-py3-none-any.whl
   pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
   ```
   These assets provide the frozen CosyVoice LLM, text embedding tables, and the ONNX SpeechTokenizer used later in `s3.sh`.

## 3. Data Preparation Pipeline
### Step 1: Resample LibriSpeech (`preprocess_librispeech.py`)
Purpose: normalize LibriSpeech audio to mono 16 kHz WAV, cap to 5,000 utterances per subset, and emit JSONL manifests with utterance IDs, transcripts, and resampled paths.
Commands:
```bash
cd src
python preprocess_librispeech.py \
  --librispeech_dir ../data/LibriSpeech \
  --output_dir ../data \
  --target_sr 16000 \
  --max_samples 5000
```
Outputs:
- `data/resampled_audio/{train-clean-100,test-clean}/**/*.wav`
- `data/{train-clean-100,test-clean}.jsonl`

### Step 2: Text embeddings & Whisper features (`utt2text_and_feature.py`)
Purpose: for each utterance listed in the JSONL manifest, the script:
- Uses CosyVoice front-end to tokenize transcripts and extract text embeddings.
- Runs Whisper Large-V3 to derive mid-layer and final-layer encoder representations for the 16 kHz audio.
- Stores the tensors in `text_emb.pt` and `speech_feat.pt` (with `mid`/`final` keys).
Command:
```bash
python utt2text_and_feature.py \
  --jsonl ../data/train-clean-100.jsonl \
  --model_dir ../repo/CosyVoice/pretrained_models/CosyVoice-300M \
  --output_text ../data/text_emb.pt \
  --output_whisper ../data/speech_feat.pt
```
Environment variables inside the script redirect Hugging Face/ModelScope caches for efficiency.

### Step 3: Speech tokens via SpeechTokenizer (`s3.sh`)
Purpose: run the CosyVoice SpeechTokenizer ONNX model to convert resampled waveforms into discrete token sequences (`s3_tokens.pt`).
Command:
```bash
bash src/s3.sh
```
What happens:
- Builds `data/wav.scp` listing utterance IDs to waveform paths and checks for duplicate IDs.
- Invokes `repo/CosyVoice/tools/extract_speech_token.py` with `speech_tokenizer_v1.onnx`.
- Renames `utt2speech_token.pt` to `s3_tokens.pt` for consistency with `example_code.py`.

## 4. Training & Evaluation (`example_code.py`)
Purpose: fine-tune a SimpleTextSpeechAggregator + CosyVoice LLM wrapper that fuses CosyVoice text embeddings with Whisper features to predict S3 tokens.
Key elements:
- `SimpleTextSpeechAggregator`: projects text embeddings, Whisper final-layer (keys), and mid-layer (values) into a shared hidden space and applies scaled dot-product attention.
- `CosyVoiceS3Model`: freezes the backbone LLM, learns lightweight fusion (`fuse_alpha`) + input/output projections, and projects hidden states to `4096 + 1` S3 vocabulary.
- Training loop (`train_one_epoch` / `eval_one_epoch`): supports gradient accumulation, token-level loss tracking, and Cosine LR scheduling.
- `predict_s3`: autoregressive decoder for qualitative inspection.

Run training after data artifacts are ready:
```bash
python src/example_code.py
```
Artifacts:
- Best checkpoint saved to `checkpoints/model_best.pt`.
- Validation/test metrics and sampled predictions stored under `results/` (timestamped JSON/JSONL files).
- Console prints include epoch losses, learning rate, and example prediction lengths.

