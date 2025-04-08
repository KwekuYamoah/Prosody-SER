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
    new_data = {}
    for key in data.keys():
        # change the mp3 path to wav path for the key
        new_key = key.replace('.mp3', '.wav').replace("_mp3", "_wav")
        
        # new key gets the value of the old key
        new_data[new_key] = data[key]
        

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    change_mp3_path_to_json()
    print("mp3 path changed to wav path in the json file.")
