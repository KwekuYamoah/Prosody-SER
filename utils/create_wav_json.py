import json

def change_mp3_path_to_json(input_file="data/ser_audio_features_mp3.json", output_file="data/ser_audio_features_wav.json"):
    """
    Change the mp3 path to wav path in the json file.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    # clear output file
    with open(output_file, 'w') as f:
        f.write('')

    for key in data.keys():
        # change the mp3 path to wav path for the key
        
        if key.endswith('.mp3'):
            new_key = key.replace('movie_audio_segments_wav', 'movie_audio_segments_mp3').replace('.mp3', '.wav')
            data[new_key] = data.pop(key)
            print(f"Changed {key} to {new_key}")
        

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    change_mp3_path_to_json()
    print("mp3 path changed to wav path in the json file.")
