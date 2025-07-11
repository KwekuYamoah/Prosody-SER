#!/usr/bin/env python3
"""
Utility script to preprocess audio files and update JSONL files with audio data.
This helps avoid loading audio files one by one during training.
"""

import os
import json
import argparse
import re
import numpy as np
import jsonlines

import librosa
import soundfile as sf

from tqdm import tqdm
from typing import Dict, List, Optional

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    return data



def remove_special_chars_and_numbers(input_string, allowed_chars=""):
    '''
    Removes all of the numbers and special characters from the string

    Params:
        input_string (str): Input containing the words, special characters and numbers
    
    Returns:
        str : Input string without special characters and numbers
    '''
    # Escape all allowed characters for regex, in case they have special meaning
    escaped_allowed = re.escape(allowed_chars)
    
    # Create a regex pattern that allows letters and specified characters
    pattern = f'[^a-zA-Z{escaped_allowed}]'
    
    return re.sub(pattern, '', input_string)


def get_unique_vocab_chars(data_path):
    '''
    Obtain all of the unique vocabulary characters from the transcriptions

    Params:
        data_path (str): This is the path that contains the dataset.
    
    Returns:
        vocabulary_dict (dict): Creates a dict containing the unique vocabulary characters.
    '''

    with open(data_path, 'r') as input_data:
        data_content = json.load(input_data)

   

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file"""
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     for item in data:
    #         json.dump(item, f, ensure_ascii=False, indent=4)


def process_audio_file(args: tuple) -> Optional[Dict]:
    """Process a single audio file"""
    item, audio_base_path, vocab_json_path, target_sr, max_duration = args

    # new dict to store the processed data
    processed_data = {}

    
    tokenizer = Wav2Vec2CTCTokenizer(
                                    vocab_json_path,
                                    unk_token="[UNK]",
                                    pad_token="[PAD]"
                                )

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

        # extract the prosody labels
        prosody_labels = item['prosody_annotations']
        # join words into a sentence
        sentence = "".join(item['words'])
        # remove the special characters from the sentence
        sentence = remove_special_chars_and_numbers(sentence, allowed_chars="ɛɔ")
        # generate the asr labels for the transcription using the tokenizer 
        extracted_asr_labels = tokenizer(sentence, return_tensors="pt").input_ids.squeeze().tolist()
        # get the emotion label
        extracted_emotion_label = item['emotion']

        # store in our processed data dict for all items
        processed_data['audio'] = speech.tolist()
        processed_data['prosody'] =prosody_labels
        processed_data['asr'] =extracted_asr_labels
        processed_data['emotion'] = extracted_emotion_label

        return processed_data

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def preprocess_dataset(
    input_jsonl: str,
    output_jsonl: str,
    audio_base_path: str,
    vocab_json_path: str,
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
    args_list = [(item, audio_base_path, vocab_json_path, target_sr, max_duration)
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
    parser.add_argument("--vocab_json_path", required=True,
                        help="Path to vocabulary JSON file")
    parser.add_argument("--target_sr", type=int,
                        default=16000, help="Target sampling rate")
    parser.add_argument("--max_duration", type=float,
                        default=600, help="Maximum audio duration in seconds")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="Number of worker processes")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    preprocess_dataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        audio_base_path=args.audio_base_path,
        vocab_json_path=args.vocab_json_path,
        target_sr=args.target_sr,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
