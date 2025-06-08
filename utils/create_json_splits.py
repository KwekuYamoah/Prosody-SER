import json
# read the json file
data_path = './data/ser_audio_features_wav.json'
with open(data_path, 'r') as f:
    data = json.load(f)

# create separate lists for each split
splits = {"train": [], "test": [], "val": []}

for audio_path, info in data.items():
    # create a new record
    record ={
        "audio_filepath": audio_path,
        "words": info["words"],
        "prosody_annotations": info["prosody_annotations"],
        "emotion": info["emotion"]
    }

    # determine the split based on the audio path
    split_name = info["split"]
    splits[split_name].append(record)

base_path = "./data/"
# save the splits to separate json files
for split_name, records in splits.items():
  with open(base_path+"ser_audio_features_wav_"+f"{split_name}.jsonl", 'w') as f:
    for rec in records:
      f.write(json.dumps(rec, ensure_ascii=False)+"\n")
