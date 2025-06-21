import json
import re
import librosa
from transformers import Wav2Vec2CTCTokenizer
import math

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
    
    #get all of the transcriptions and then extract unique vocabulary from them
    all_text = " "
    for data_item in data_content:
        transcription = " ".join(data_content[data_item]['words'])
        all_text += remove_special_chars_and_numbers(transcription, allowed_chars="ɛɔ")
    vocabulary = list(set(all_text))

    #remove the special characters and numbers from the vocabulary


    #sort the vocabulary and then assign keys to identify the distinct characters
    vocabulary_dict = {v: k for k, v in enumerate(sorted(vocabulary))}

    #add the [UNK],  |  and [PAD] tokens to the vocabulary dictionary
    vocabulary_dict['[UNK]'] = len(vocabulary_dict)
    vocabulary_dict['[PAD]'] = len(vocabulary_dict)
    vocabulary_dict["|"] = vocabulary_dict[" "]
    del vocabulary_dict[" "]

    #save the vocabulary as a JSON file
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocabulary_dict, vocab_file)

    return vocabulary_dict


def tokenize_text(text):
    '''
    Tokenizes text with the wav2vec2 tokenizer

    Params:
        text (str): This is the input text to be tokenized.
    
    Returns:
        tokenized_text_values (obj): Tensor object filled with the tokenized values.
    '''
    tokenizer = Wav2Vec2CTCTokenizer(
                                    "/home/dasa/ser_project/codebase/new_vocab.json",
                                    unk_token="[UNK]",
                                    pad_token="[PAD]"
                                )
    
    tokenized_text_values = tokenizer(text, return_tensors="pt").input_ids.squeeze().tolist()
    return tokenized_text_values


def tokenize_without_delimiter(text):
    tokenizer = Wav2Vec2CTCTokenizer(
                                    "/home/dasa/ser_project/codebase/vocab.json",
                                    unk_token="[UNK]",
                                    pad_token="[PAD]"
                                )
    
    tokenized_text_values = tokenizer(text, return_tensors="pt").input_ids.squeeze().tolist()
    return tokenized_text_values


'''
  "movie_audio_segments_wav/abrokyire_yonko_part_1_2016/abrokyire_yonko_part_1_2016_1279990_1284500.wav": {
    "words": [
      "aabaeve",
      "adɛn?"
    ],
    "prosody_annotations": [
      0,
      1
    ],
    "emotion": 3,
    "split": "train"
  }
'''
def construct_input_feature_vectors(data_info_json, vocab_json_path, storage_path):
    '''
    Produces a json object containing the extracted input feature vectors for the movie audios, prosody annotations and
    emotion labels. It also categorizes these extracted features into 'train', 'val' and 'test' keys within a dict.

    Params:
        data_info_json (obj): This is a json object that contains information about the data to be extracted.
        vocab_json (str): This is the path to where the vocabulary of the transcriptions are stored. 
        storage_path (str): This is the path to where the extracted input features are stored.
    
    Returns:
        input_feature_json (obj): Json object categorized into 'train', 'test' and 'val', where each section contains
                                    the extracted input features of audios, prosody annotations and emotion labels.

    '''

    '''
    tokenizer = Wav2Vec2CTCTokenizer(
                                    vocab_json_path,
                                    unk_token="[UNK]",
                                    pad_token="[PAD]",
                                    word_delimiter_token="|"
                                )
    '''
    tokenizer = Wav2Vec2CTCTokenizer(
                                        vocab_json_path,
                                        unk_token="[UNK]",
                                        pad_token="[PAD]"
                                    )
    
    input_feature_json = {'train': {'audio':[], 'prosody':[], 'asr':[],'emotion':[]}, 'test': {'audio':[], 'prosody':[], 'asr':[],'emotion':[]}, 'val':{'audio':[], 'prosody':[], 'asr':[],'emotion':[]}}
    # read the contents of data_info_json
    with open(data_info_json, 'r') as json_input:
        data_info = json.load(json_input)

        #initialize a variable to keep track of the changes made to the string as a quick sanity check
        sanity_check_count = 0

        # for each audio in data_info extract its audio features, prosody, asr and emotion labels
        for audio_path in data_info:
            #construct the correct audio path
            constructed_audio_path = '../github_repo/data/' + audio_path

            extracted_audio_features, sr = librosa.load(constructed_audio_path, sr=16000)

            extracted_prosody_labels = data_info[audio_path]['prosody_annotations']

            # join the words into a sentence
            #extracted_words = "|".join(data_info[audio_path]["words"])
            extracted_words = "".join(data_info[audio_path]["words"])

            # remove the special characters from the sentence
            #processed_words = remove_special_chars_and_numbers(extracted_words, allowed_chars="ɛɔ|")
            processed_words = remove_special_chars_and_numbers(extracted_words, allowed_chars="ɛɔ")


            if sanity_check_count <= 5:
                print('processed words: ', processed_words)
                sanity_check_count += 1

            # generate the asr labels for the transcription
            extracted_asr_labels = tokenizer(processed_words, return_tensors="pt").input_ids.squeeze().tolist()

            # get the emotion label
            extracted_emotion_label = data_info[audio_path]["emotion"]

            #place the extracted values into input_feature_json
            input_feature_json[data_info[audio_path]['split']]['audio'].append(extracted_audio_features.tolist())
            input_feature_json[data_info[audio_path]['split']]['prosody'].append(extracted_prosody_labels)
            input_feature_json[data_info[audio_path]['split']]['asr'].append(extracted_asr_labels)
            input_feature_json[data_info[audio_path]['split']]['emotion'].append(extracted_emotion_label)
    
    #write the extracted input features into the storage dorectory
    with open(storage_path, 'w') as output_file:
        json.dump(input_feature_json, output_file, indent=4)
    
    return input_feature_json


def conform_data_structure(data_path, output_path):
    '''
    Transforms the structure of given data into the akan data being used so that the models developed can be tested with that data as well.

    Params:
        data_path (str): This is the path to the data
        output_path (str): This is the path to where the newly processed data should be stored.
    
    Returns:
        input_feature_json (dict): Contains the newly processed data structure for the provided data path
    '''
    new_data_json = []

    sanity_check_count = 0

    #read the data from the data path
    with open(data_path, 'r') as input_file:
        audio_json_dataset = json.load(input_file)

        #determine the number of elements that should be in the train, test and val data groups (use 50-25-25 split)
        num_data_entries = len(audio_json_dataset)

        train_end_index = math.floor(0.50 * num_data_entries)
        train_dataset = audio_json_dataset[:train_end_index]

        val_end_index = train_end_index + (num_data_entries - train_end_index)//2
        val_dataset = audio_json_dataset[train_end_index : val_end_index]

        test_dataset = audio_json_dataset[val_end_index : num_data_entries]

        #place all of the dataset splits into one list and iterate over the entries within each of them
        combined_dataset_splits = {'train':train_dataset, 'val':val_dataset, 'test':test_dataset}
        
        input_feature_json = {'train': {'audio':[], 'prosody':[], 'asr':[],'emotion':[]}, 'test': {'audio':[], 'prosody':[], 'asr':[],'emotion':[]}, 'val':{'audio':[], 'prosody':[], 'asr':[],'emotion':[]}}

        #for each entry, obtain the audio path and its corresponding text
        for dataset_split in combined_dataset_splits:
            for data_entry in combined_dataset_splits[dataset_split]:
                data_path = data_entry['path']
                data_text = data_entry['text']

                #extract the audio features
                extracted_audio_features, sr = librosa.load(data_path, sr=16000)

                #pre-process the text
                lowercase_text = data_text.lower().split(' ')
                joined_lowercase_text = "".join(lowercase_text)
                cleaned_joined_lowercase_text = remove_special_chars_and_numbers(joined_lowercase_text)

                if sanity_check_count <= 5:
                    print('processed words: ', cleaned_joined_lowercase_text)
                    sanity_check_count += 1

                #tokenize the processed text
                tokenized_text_values = tokenize_text(cleaned_joined_lowercase_text)

                #place the data into the input feature json dict
                input_feature_json[dataset_split]['audio'].append(extracted_audio_features.tolist())
                input_feature_json[dataset_split]['prosody'].append([0])
                input_feature_json[dataset_split]['asr'].append(tokenized_text_values)
                input_feature_json[dataset_split]['emotion'].append([0])
    

    with open(output_path , 'w') as output_data:
        json.dump(input_feature_json, output_data, indent=4)

    return input_feature_json


def search_features(extracted_features_path, search_section, search_key):
    '''
    Showcases the requested values within the json that contains the extracted features.
    Params:
        extracted_features_path (str): This is a path to the extracted input features.
        search_section (str): This indicates the section (i.e train, test or val) that we would like to search in.
        search_key (str): This indicates the specific key within the search section that we are interested in.
    
    Returns:
        search_info (str): This is the requested information from the json object that contains the extracted features. 
    '''
    with open(extracted_features_path, 'r') as input_features:
        input_data = json.load(input_features)
        requested_section = input_data[search_section]
        requested_key_info = requested_section[search_key]
        search_info = requested_key_info[0]



    return search_info




if __name__ == '__main__':
    #vocab = get_unique_vocab_chars('/home/dasa/ser_project/github_repo/data/ser_audio_features_wav.json')
    
    '''
    input_features = construct_input_feature_vectors(
                                                '/home/dasa/ser_project/github_repo/data/ser_audio_features_wav.json', 
                                                    'vocab.json',
                                                    'extracted_input_features.json'
                                                    )
    '''
    
    #results = search_features('extracted_input_features.json', 'train', 'asr')
    #print(results)
    returned_data_structure = conform_data_structure(
                                                        '/home/dasa/cross_lingual_prosodic_annotation/Research Code/pretrained_prosody_annotation_models/useful_files/asr/libri_light/librispeech_finetuning/1hour.json',
                                                        'conformed_librispeech_dataset.json'
                                                    )



    #print(returned_data_structure)
    
    
