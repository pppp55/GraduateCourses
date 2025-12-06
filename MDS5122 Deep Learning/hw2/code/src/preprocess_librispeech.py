#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import torchaudio

def find_audio_transcript_pairs(root_dir):
    """search LibriSpeech pairs"""
    pairs = []

    for speaker_dir in Path(root_dir).glob("*/"):
        if not speaker_dir.is_dir():
            continue
            
        for chapter_dir in speaker_dir.glob("*/"):
            if not chapter_dir.is_dir():
                continue
                
            # read trans.txt
            transcript_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if not transcript_file.exists():
                continue
                
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # form: utterance_id TRANSCRIPT_TEXT
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    
                    utterance_id, text = parts[0], parts[1]
                    audio_file = chapter_dir / f"{utterance_id}.flac"
                    
                    if audio_file.exists():
                        pairs.append({
                            "audio_path": str(audio_file),
                            "text": text,
                            "utterance_id": utterance_id
                        })
    
    return pairs

def resample_audio(input_path, output_path, target_sr=16000):
    """Resample all audio to 16 kHz"""
    input_path_str = str(input_path)
    output_path_str = str(output_path)
    waveform, orig_sr = torchaudio.load(input_path_str)
    
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    
    torchaudio.save(output_path_str, waveform, target_sr)
    return output_path_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_dir", type=str, default='../data/LibriSpeech', 
                       help="LibriSpeech path")
    parser.add_argument("--output_dir", type=str, default='../data/',
                       help="output path")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="target sampling rate")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="max samples (for testing)")
    args = parser.parse_args()
    
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # find all subsets
    subsets = ["train-clean-100", "test-clean"]
    
    seen_utt_ids = set()

    for subset in subsets:
        subset_dir = Path(args.librispeech_dir) / subset
        if not subset_dir.exists():
            print(f"warning: {subset_dir} doesn't exist, skip")
            continue
            
        print(f"processing {subset}...")
        
        # find record-text pairs
        pairs = find_audio_transcript_pairs(subset_dir)
        print(f"find {len(pairs)} samples")
        
        # limit number of sample (for test)
        if args.max_samples and len(pairs) > args.max_samples:
            pairs = pairs[:args.max_samples]
            print(f"limit to {args.max_samples} samples")
        
        # make resample dir
        resampled_audio_dir = Path(args.output_dir) / "resampled_audio" / subset
        os.makedirs(resampled_audio_dir, exist_ok=True)
        
        # process every sample
        output_pairs = []
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                print(f"processing: {i}/{len(pairs)}")
            
            # resample
            input_audio = Path(pair["audio_path"])
            output_audio = resampled_audio_dir / f"{pair['utterance_id']}.wav"

            utt_id = pair["utterance_id"]
            if utt_id in seen_utt_ids:
                print(f"duplicate utterance id detected: {utt_id}, skip")
                continue
            
            try:
                resampled_path = resample_audio(input_audio, output_audio, args.target_sr)
                
                output_pairs.append({
                    "utt_id": utt_id,
                    "audio_path": str(resampled_path),
                    "text": pair["text"],
                    "subset": subset
                })
                seen_utt_ids.add(utt_id)
            except Exception as e:
                print(f"exception while processing {input_audio}: {e}")
                continue
        
        # save as jsonl
        output_file = Path(args.output_dir) / f"{subset}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in output_pairs:
                f.write(json.dumps(pair) + '\n')
        
        print(f"save {len(output_pairs)} samples to {output_file}")

if __name__ == "__main__":
    main()