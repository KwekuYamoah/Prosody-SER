import os
import numpy as np
import pandas as pd
import re
import librosa
import json

from pydub import AudioSegment
from emotion_encoder import encode_emotion




def load_and_prepare_data():
    """Read and prepare the data from CSV file.

    This function loads a CSV file containing the Akan Speech Emotion Dataset 
    with consolidated prosody features and converts it into a list of dictionaries.

    Returns:
        list: A list of dictionaries where each dictionary represents a row from the CSV file,
              with column names as keys and corresponding values.
    """
    """Read and prepare the data from CSV file."""

    consolidated_df = pd.read_csv(
        "excel_files/Akan Speech Emotion Dataset with Consolidated Prosody.csv",
        encoding='utf-8',
        delimiter=','
    )

    # convert df to dict
    consolidated_dict = consolidated_df.to_dict(orient='records')


    return consolidated_dict

def slice_audio(audio_file, start_time, end_time, save_folder, is_milliseconds=False, is_wav=False):
    """
    Slice the audio file based on the given start and end times.
    Args:
        audio_file (str): Path to the input audio file (mp3 format)
        start_time (float): Start time in seconds or milliseconds
        end_time (float): End time in seconds or milliseconds
        save_folder (str): Directory path where the sliced audio will be saved
        is_milliseconds (bool): Whether the start_time and end_time are in milliseconds
    Returns:
        str: Path to the saved audio slice file
    Notes:
        - The output filename format is: {original_name}_{start_time}_{end_time}.mp3
        - Start time is clamped to minimum of 0 seconds
        - Times are converted to milliseconds internally if needed
        - Requires pydub.AudioSegment for processing
    
    """
    # Load the audio file using pydub
    audio = AudioSegment.from_mp3(audio_file) 

    # Extract the name of the audio file
    audio_name = os.path.basename(audio_file)
    # Remove the file extension
    audio_name = os.path.splitext(audio_name)[0]
    
    # Convert time to milliseconds for pydub if needed
    if is_milliseconds:
        start_ms = max(0, start_time)
        end_ms = end_time
    else:
        start_ms = max(0, start_time * 1000)
        end_ms = end_time * 1000
    
    # Extract the slice of the audio corresponding to the word
    word_audio = audio[start_ms:end_ms]
    
    # Save to a temporary file
    if is_wav:
        temp_filename = f"{save_folder}/{audio_name}_{start_time}_{end_time}.wav"
        word_audio.export(temp_filename, format="wav")
    else:
        temp_filename = f"{save_folder}/{audio_name}_{start_time}_{end_time}.mp3"
        word_audio.export(temp_filename, format="mp3")
    
    return temp_filename

def extract_and_save_audio_features(data_dict, base_dir="audio/movie_audio_segments_wav", output_file="data/ser_audio_features_mp3.json"):
    """
    Extracts audio features from the data dictionary and saves them to a JSON file incrementally.
    This function processes audio files one by one, loads them using librosa, and immediately
    writes the features to a JSON file to avoid memory issues.
    
    Args:
        data_dict (list): List of dictionaries containing audio metadata
        base_dir (str, optional): Base directory containing audio segments
        output_file (str, optional): Path to the output JSON file
    """
    # Clear the output file by writing an empty string
    with open(output_file, 'w') as f:
        f.write("")

    # Counter for processed files
    processed_count = 0

    audio_feature_dict = {}
    # emotion list 
    emotion_list = []
    # Process each item in the data dictionary
    for idx, item in enumerate(data_dict):
        try:
            # extract movie title, start time and end time from the dictionary
            movie_title = item.get('Movie Title')
            start_time = int(item.get('start_milliseconds'))
            end_time = int(item.get('end_milliseconds'))
            split = item.get('split')
            emotion = item.get('Emotion')
            encoded_emotion = encode_emotion(emotion)
            # add the emotion to the list
            emotion_list.append(emotion)


            # adjusting start and end times to expand context window
            start_time = max(0, start_time - 10) # 10ms before the start time
            end_time = end_time + 1500 # 25ms after the end time 
            
            # create intermediate string for path
            inter_path = f"{movie_title}_{start_time}_{end_time}.mp3"

            # build path to audio file
            relative_path = os.path.join(base_dir, movie_title, inter_path)

            json_path = os.path.join(base_dir.split('/')[1], movie_title, inter_path)

            # check if the file exists try-except block
            try:
                # First try to load the original path
                if os.path.exists(relative_path):
                    audio, sr = librosa.load(relative_path, sr=None, res_type='kaiser_fast')
                else:
                    # If not found, try with adjusted end time
                    # end_time -= 5
                    # inter_path = f"{movie_title}_{start_time}_{end_time}.mp3"
                    # relative_path = os.path.join(base_dir, movie_title, inter_path)
                    
                    if os.path.exists(relative_path):
                        audio, sr = librosa.load(relative_path, sr=None, res_type='kaiser_fast')
                    else:
                        # If still not found, create the segment
                        # Make sure the directory exists
                        save_folder = f"{base_dir}/{movie_title}"
                        os.makedirs(save_folder, exist_ok=True)

                        # readjust the end time
                        # end_time += 5
                        
                        filename = slice_audio(
                            audio_file=f"audio/full_movie_audios/{movie_title}.mp3",
                            start_time=start_time,
                            end_time=end_time,
                            save_folder=save_folder,
                            is_milliseconds=True,
                        )
                        
                        # Load the newly created file
                        audio, sr = librosa.load(filename, sr=None, res_type='kaiser_fast')
                        print(f"Slice for {movie_title} created at {filename}")
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")
                continue

            # extract utterance from the dictionary
            utterance = item.get('Utterance')

            # extract Consolidated Prosody from the dictionary
            consolidated_prosody = item.get('Consolidated')
            
            # initilize first regex pattern
            initial_pattern =  r'([^ (]+)\(\s*(\d+)\s*\)(\S*)'
            tokens = re.findall(initial_pattern, consolidated_prosody)
            
            # Initialize words and prosody_annotations for this item
            words = []
            prosody_annotations = []
            
            # check if the regex pattern matched
            if tokens:
                # results is a list of tuples, now extract first index of each tuple to a list to get the words
                words = [word[0]+word[-1] for word in tokens]
                # send the words to lower case
                words = [word.lower() for word in words]
                
                # extract second index of each tuple to a list to get the prosody
                prosody_annotations = [int(annotation[1]) for annotation in tokens]
                
            # Convert NumPy array to list for JSON serialization
            audio_list = audio.tolist()
            
            # Create the feature dictionary for this item
            # feature_dict = {
            #     'audio_features': audio_list,
            #     'utterance': utterance,
            #     'words': words,
            #     'prosody_annotations': prosody_annotations,
            #     'split': split
            # }

            feature_dict = {
                'words': words,
                'prosody_annotations': prosody_annotations,
                'emotion': encoded_emotion,
                'split': split
            }
            # Add the audio features to the dictionary
            audio_feature_dict[json_path] = feature_dict
            # Append the new feature as a single JSON object on a new line (NDJSON format), using inter_path as the key
            # try:
            #     with open(output_file, 'a', encoding='utf-8') as f:
            #         f.write(json.dumps({inter_path: feature_dict}, ensure_ascii=False))
            #         f.write("\n")
            # except Exception as e:
            #     print(f"Error writing to JSON file: {e}")
            #     continue
            
            # Increment the counter
            processed_count += 1

            # # reset the audio dictionary
            # audio_feature_dict = {}
            
            # Print progress every 10 items
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")
            
            # if processed_count >= 100:
            #     print("Processed 1000 files, stopping for now.")
            #     break
                
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue

    print(f"Unique emotions found: {set(emotion_list)}")
    
    print(f"Completed processing {processed_count} files.")
    # Save the audio features to the JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(audio_feature_dict, f, ensure_ascii=False, indent=2)
    return processed_count

if __name__ == '__main__':
    data_dict = load_and_prepare_data()
    
    # Extract and save audio features incrementally
    processed_count = extract_and_save_audio_features(data_dict)
    
    print(f"Audio features saved to data/ser_audio_features.json")
    print(f"Number of files extracted: {processed_count}")