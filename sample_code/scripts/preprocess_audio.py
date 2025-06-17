#!/usr/bin/env python3
"""
Utility script to preprocess audio files and update JSONL files with audio data.
This helps avoid loading audio files one by one during training.
"""

import os
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_audio_file(args: tuple) -> Optional[Dict]:
    """Process a single audio file"""
    item, audio_base_path, target_sr, max_duration = args

    try:
        # Construct full audio path
        audio_path = os.path.join(audio_base_path, item['audio_filepath'])

        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            return None

        # Load and process audio
        speech, sr = librosa.load(audio_path, sr=target_sr)

        # Check duration
        duration = len(speech) / target_sr
        if max_duration is not None and duration > max_duration:
            print(
                f"Warning: Audio duration {duration:.2f}s exceeds max duration {max_duration}s: {audio_path}")
            return None

        # Convert to float32 and normalize
        speech = speech.astype(np.float32)
        speech = librosa.util.normalize(speech)

        # Update item with audio data
        # Convert to list for JSON serialization
        item['audio_data'] = speech.tolist()
        item['audio_sr'] = target_sr
        item['audio_duration'] = duration


        return item

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def preprocess_dataset(
    input_jsonl: str,
    output_jsonl: str,
    audio_base_path: str,
    target_sr: int = 16000,
    max_duration: Optional[float] = None,
    num_workers: Optional[int] = None
):
    """
    Preprocess audio files and update JSONL file with audio data.

    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file
        audio_base_path: Base path for audio files
        target_sr: Target sampling rate
        max_duration: Maximum audio duration in seconds
        num_workers: Number of worker processes (default: CPU count)
    """
    print(f"\n{'='*50}")
    print(f"🚀 Starting audio preprocessing")
    print(f"{'='*50}")

    # Load data
    print(f"\n📂 Loading data from {input_jsonl}")
    data = load_jsonl(input_jsonl)
    print(f"   Found {len(data)} samples")

    # Prepare arguments for parallel processing
    args_list = [(item, audio_base_path, target_sr, max_duration)
                 for item in data]

    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print(f"\n⚙️ Using {num_workers} worker processes")

    # Process audio files in parallel
    print("\n🎵 Processing audio files...")
    processed_data = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_audio_file, args_list), total=len(args_list)):
            if result is not None:
                processed_data.append(result)

    # Save processed data
    print(f"\n💾 Saving processed data to {output_jsonl}")
    save_jsonl(processed_data, output_jsonl)

    # Print statistics
    print(f"\n📊 Processing Statistics:")
    print(f"   Total samples: {len(data)}")
    print(f"   Processed samples: {len(processed_data)}")
    print(f"   Skipped samples: {len(data) - len(processed_data)}")

    print(f"\n{'='*50}")
    print(f"✅ Preprocessing completed!")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio files and update JSONL files")
    parser.add_argument("--input_jsonl", required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--audio_base_path", required=True,
                        help="Base path for audio files")
    parser.add_argument("--target_sr", type=int,
                        default=16000, help="Target sampling rate")
    parser.add_argument("--max_duration", type=float,
                        default=None, help="Maximum audio duration in seconds")
    parser.add_argument("--num_workers", type=int,
                        default=None, help="Number of worker processes")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    preprocess_dataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        audio_base_path=args.audio_base_path,
        target_sr=args.target_sr,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
