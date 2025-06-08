import os
import parselmouth
import json
import numpy as np
import evaluate
import datasets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


from transformers import TrainingArguments
from datasets import load_metric, load_dataset, Audio

import torch
from transformers import *
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import argparse
import warnings
#from combined_asr_prosody_model import *
from new_combined_asr_prosody_model import *
from model import *



import jiwer



def prepare_librispeech_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch



def librispeech_load_json_dataset(file_path, data_type):
    data_files = {}
    if file_path is not None:
        data_files[data_type] = file_path

    phrasing_features = datasets.Features({
        'path': datasets.features.Value('string'),
        'text': datasets.features.Value('string')
    })

    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)
    return dataset





def compute_metrics(pred):
    wer = evaluate.load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits[0], axis=-1)
    pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)

    #print the results to see whether the fine-tuning process is working or not

    
    for i in range(min(10, len(pred_str))):
        print(f"Prediction: {pred_str[i]}")
        print(f"Reference: {label_str[i]}")
    


    #remove a reference and its corresponding prediction if the reference is just an empty string
    non_empty_ref = []
    non_empty_preds = []

    for ref_index in range(len(label_str)):
        if label_str[ref_index] != '':
            non_empty_ref.append(label_str[ref_index])
            non_empty_preds.append(pred_str[ref_index])


    wer_result = wer.compute(predictions=non_empty_preds, references=non_empty_ref)
    return {"wer": wer_result}


def compute_loss(model, inputs, return_outputs=False):
    # Your custom loss computation
    outputs = model(**inputs)
    loss = outputs["loss"]
    return (loss, outputs) if return_outputs else loss



def calculate_wer_and_cer(predicted_texts, correct_texts):
    # Calculate Word Error Rate
    wer = jiwer.wer(correct_texts, predicted_texts)

    # Calculate Character Error Rate
    cer = jiwer.cer(correct_texts, predicted_texts)

    print(f"Word Error Rate: {wer}, Character Error Rate: {cer}")

    return wer, cer


def initialize_model_lstm(model_path, model):
    original_model = Wav2Vec2ForAudioFrameClassification_custom.from_pretrained('facebook/wav2vec2-base', num_labels=1)

    #load the stored model initialisation weights
    original_model.load_state_dict(torch.load('./all_burnc_results/combined_normalized_extra_ffn_30_epochs_model_iteration_1.pth'))
   

    #identify the keys of the lstm layer of the model state dictionary
    lstm_keys = {}

    for name, param in original_model.named_parameters():
        if 'lstm' in name.lower():
            if 'weight' in name.lower():
                weight_name = name.split('.')[1]
                lstm_keys[weight_name] = param
            elif 'bias' in name.lower():
                bias_name = name.split('.')[1]
                lstm_keys[bias_name] = param
    
    #place the extracted lstm initializations in the new model
    model.lstm.load_state_dict(lstm_keys)


    return


def inference(dataset, model_dir):
    #load all of the data samples from the dataset
    #iteratively move through the data samples and generate predictions for each data sample
    #store all of the predictions and then compute the wer and cer for the generated predictions to measure model performance.

    predicted_texts = ""
    correct_texts = ""


    #don't forget to add the processor
    transcriber = pipeline("automatic-speech-recognition", model=model_dir, tokenizer=processor)


    for audio_item in dataset:
        audio_path = audio_item['audio']['path']
        audio_transcription = audio_item["text"]

        
        prediction = transcriber(audio_path)
        print('correct text: ', audio_transcription)
        print('predicted text: ', prediction)


        
        predicted_texts = predicted_texts + " " + prediction['text']
        


        correct_texts = correct_texts + " " + audio_transcription
    

    return predicted_texts, correct_texts


class UnfreezeModelCallback(TrainerCallback):
    def __init__(self, unfreeze_step, model):
        self.unfreeze_step = unfreeze_step
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_step:
          print("Unfreezing!!")
          model.unfreeze_base_model()


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, List[float]]]]) -> Dict[str, torch.Tensor]:
        # Prepare the inputs for padding
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
     
        
        

        max_length = max(len(feature["pitch"]) for feature in features)
        pitch_batch = torch.full((len(features), max_length), fill_value=0.0)  # Replace 0.0 with an appropriate padding value for pitch
        for i, feature in enumerate(features):
            pitch_batch[i, :len(feature["pitch"])] = feature["pitch"].float().clone().detach()



        max_label_length = max(len(feature["asr_labels"]) for feature in features)
        asr_labels_batch = torch.full((len(features), max_label_length), fill_value=-100, dtype=torch.long)  # -100 is often used for ignoring in loss computation
        for i, feature in enumerate(features):
            label_len = len(feature["asr_labels"])
            asr_labels_batch[i, :label_len] = feature["asr_labels"].float().clone().detach()
  


        # Return the final batch with all features
        batch["asr_labels"] = asr_labels_batch
        batch["pitch"] = pitch_batch



        if "prosodic_labels" in features[0]:
            max_prosodic_label_length = max(len(feature["prosodic_labels"]) for feature in features)
            prosodic_labels_batch = torch.full((len(features), max_prosodic_label_length), fill_value=0, dtype=torch.float) 
            for i, feature in enumerate(features):
                label_len = len(feature["prosodic_labels"])
                prosodic_labels_batch[i, :label_len] = feature["prosodic_labels"].float().clone().detach()
            
            batch["prosodic_labels"] = prosodic_labels_batch

        return batch






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model for audio frame classification')
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base", help='Path, url or short name of the model')
    parser.add_argument('--file_train', type=str, default='./all_burnc_results/generated_librispeech_labels_semi_supervised_models/librispeech_audio_text_semi_supervised_3_without_weight.json', help='Path to the training dataset (a JSON file)')
    parser.add_argument('--file_valid', type=str, default='./burnc_prosody_transcript_val_asr_without_empty_texts.json', help='Path to the validation dataset (a JSON file)')
    parser.add_argument('--file_eval', type=str, default='./burnc_prosody_transcript_val_asr_without_empty_texts.json', help='Path to the evaluation (test) dataset (a JSON file)')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--file_output', type=str, default='./sequential_combined_asr_prosody_ss3_val.txt', help='Path for the output file (output.txt)')
    parser.add_argument('--model_save_dir', type=str, default='./sequential_combined_asr_prosody_ss3', help='Directory for saving the training log and the finetuned model')
    parser.add_argument('--max_duration', type=float, default=21.0, help='Maximum duration of audio files, default = 21s (must be >= duration of the longest file)')
    parser.add_argument('--mode', type=str, default="both", help='Mode: "train", "eval" or "both" (default is "both")')
    parser.add_argument('--epochs_between_checkpoints', type=int, default=1, help='Number of epochs between saved checkpoints during training. Default is 1 - saving every epoch.')
    parser.add_argument('--lr_init', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--lr_num_warmup_steps', type=int, default=0, help='Number of warmup steps for the learning rate scheduler')
    parser.add_argument('--remove_last_label', type=int, default=1, help='Remove the last value from ref. labels to match the number of predictions? 0 = no, 1 = yes (Default: yes)')
    parser.add_argument('--training_metric', type=str, default='asr', help='provide the metric that would be used to compute the loss')


    #librispeech_dataset = load_dataset("librispeech_asr")

    args = parser.parse_args()

    if args.remove_last_label > 0:
        remove_extra_label = True # in a 20.0 s audio, there will be 1000 labels but only 999 logits -> remove the last label so the numbers match
    else: # if the labels are already fixed elsewhere
        remove_extra_label = False

    do_train = args.mode in ['train','both']
    do_eval = args.mode in ['eval','both']


    if args.epochs_between_checkpoints < 0:
        raise ValueError("''--epochs_between_checkpoints'' must be >= 0")

    if do_train:
        if args.file_train is None:
            raise ValueError("Training requires path to the training dataset (argument '--file_train <path>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.num_epochs is None:
            raise ValueError("For training the model, the number of epochs must be specified (argument '--num_epochs <number>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.model_save_dir is None:
            warnings.warn("argument ''--model_save_dir'' is not set -> the finetuned model will NOT be saved.")
            if args.epochs_between_checkpoints > 0:
                print("Checkpoints during training will also NOT be saved.")

        if args.file_valid is None:
            print("There is no validation set. Loss will be calculated only on the training set.")
            do_validation = False
        else:
            do_validation = True
    else:
        do_validation = False

    if do_eval:
        do_validation = True
        if args.file_eval is None:
            raise ValueError("Evaluation requires path to the evaluation dataset (argument '--file_eval <path>'). "
                             "To disable evaluation and only perform training, use '--mode 'train''")

    if args.model_save_dir is None or args.epochs_between_checkpoints == 0:
        save_checkpoints = False
    else:
        save_checkpoints = True

   

    




 

    metric = load_metric("mse")
    dataset = load_json_dataset(args.file_train,args.file_eval,args.file_valid)
    model = Wav2Vec2CombinedASR.from_pretrained('./sequential_combined_asr_prosody_ss3/checkpoint-30000')

    #load the stored model initialisation weights for the lstm
    initialize_model_lstm('./all_burnc_results/combined_normalized_extra_ffn_30_epochs_model_iteration_1.pth', model)

    model_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=model_tokenizer)
    
    
    #rename the 'path' column to 'audio'
    dataset = dataset.rename_column("path","audio")
    #cast the audio column into the audio data types
    dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))


    def librispeech_preprocess_function(examples):
        '''
        Preprocesses the dataset for the model.
        '''
        if examples is None:
            return None

        pitches = []
        audio_arrays = []
        sampling_rate = 16000
        for x in examples["audio"]:
            audio_arrays.append(x["array"])
            snd = parselmouth.Sound(values = x["array"], sampling_frequency = 16_000)
            pitch = snd.to_pitch()
            pitches.append(list(pitch.selected_array['frequency'])[::2])
            
      
        inputs = processor(audio_arrays, sampling_rate=sampling_rate, text=examples["text"])
    
        inputs['pitch'] = pitches
        
        return inputs
    

    

    def preprocess_function(examples):
        '''
        Preprocesses the dataset for the model.
        '''
        if examples is None:
            return None

        pitches = []
        audio_arrays = []
        sampling_rate = 16000
        for x in examples["audio"]:
            audio_arrays.append(x["array"])
            snd = parselmouth.Sound(values = x["array"], sampling_frequency = 16_000)
            pitch = snd.to_pitch()
            pitches.append(list(pitch.selected_array['frequency'])[::2])
            
        
        
        labels = examples["label"]

        
        labels_rate = 50 # labels per second

        num_padded_labels = round(args.max_duration * labels_rate)

        for label in labels:
            for _ in range(len(label), num_padded_labels):
                label.append(0)
            if remove_extra_label:
                label.pop()

    
        #audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"]
        #inputs = processor(audio_arrays, sampling_rate=sampling_rate, padding='max_length', max_length=int(sampling_rate * args.max_duration), truncation=False, text=examples["text"])
        inputs = processor(audio_arrays, sampling_rate=sampling_rate, text=examples["text"])
        
        inputs['pitch'] = pitches
        inputs['prosodic_labels'] = labels

        print(inputs.keys())
        
        return inputs
        




    #determine whether to use the BURNC dataset or the Librispeech dataset as the validation and test datasets
    use_BURNC = True


    # -------------
    # process the train/val/test data
    # -------------
    if do_train:
        print("Processing training data...")
        processed_dataset_train = dataset["train"].map(preprocess_function, remove_columns=["audio","label", "text"], batched=True)
        processed_dataset_train = processed_dataset_train.rename_column("labels", "asr_labels")
        processed_dataset_train.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])
        train_dataloader = DataLoader(processed_dataset_train, shuffle=True, batch_size=args.batch_size)
        
    else:
        processed_dataset_train = None

    if do_validation:
        print("Processing validation data...")
        if use_BURNC == True:
            processed_dataset_valid = dataset["valid"].map(preprocess_function, remove_columns=["audio","label", "text"], batched=True)
            processed_dataset_valid = processed_dataset_valid.rename_column("labels", "asr_labels")
            processed_dataset_valid.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])

            valid_dataloader = DataLoader(processed_dataset_valid, shuffle=False, batch_size=args.batch_size)
        else:
            validation_dataset = '/home/dasa/cross_lingual_prosodic_annotation/Research Code/pretrained_prosody_annotation_models/useful_files/asr/librispeech_data/dev_clean/dev_data.json'
            validation_dataset = librispeech_load_json_dataset(validation_dataset, "val")
            validation_dataset = validation_dataset["val"]
            validation_dataset = validation_dataset.rename_column("path","audio")
            processed_dataset_valid = validation_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    else:
        processed_dataset_valid = None

    if do_eval:
        print("Processing test data...")
        if use_BURNC == True:
            processed_dataset_test = dataset["eval"].map(preprocess_function, remove_columns=["audio","label", "text"], batched=True)
            processed_dataset_test = processed_dataset_test.rename_column("labels", "asr_labels")
            processed_dataset_test.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])
            eval_dataloader = DataLoader(processed_dataset_test, batch_size=1)
        else:
            test_dataset = '/home/dasa/cross_lingual_prosodic_annotation/Research Code/pretrained_prosody_annotation_models/useful_files/asr/librispeech_data/test_clean/test_data.json'
            test_dataset = librispeech_load_json_dataset(test_dataset, "test")
            test_dataset = test_dataset["test"]
            test_dataset = test_dataset.rename_column("path","audio")
            processed_dataset_test = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    else:
        processed_dataset_test = None

    # ----------
    # Training
    # ----------
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model.to(device)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")


    #freeze pretrained speech model parameters
    model.freeze_base_model_except_head()

    #Randomly initialize the head of the pretrained speech model
    model.randomly_initialize_base_model_head()


    training_args = TrainingArguments(
          output_dir=args.model_save_dir,
          per_device_train_batch_size=8,
          gradient_accumulation_steps=2,
          learning_rate=2e-5,
          warmup_steps=600,
          max_steps=30000,
          gradient_checkpointing=True,
          fp16=False,
          group_by_length=True,
          evaluation_strategy="steps",
          per_device_eval_batch_size=8,
          save_steps=100,
          eval_steps=100,
          logging_steps=25,
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
          push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset_train,
        eval_dataset=processed_dataset_valid,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[UnfreezeModelCallback(unfreeze_step=3000, model=model)]
    )






    if do_train:
        print("Starting training...")
        trainer.train()
        model.config.save_pretrained(args.model_save_dir)

        

    if do_eval:
        print("Starting evaluation...")
        prosody_evaluation = True
        asr_evaluation = False

        

        if prosody_evaluation:
            out_dir = os.path.dirname(args.file_output)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)

            model.eval()
            predictions_all = []
            progress_bar = tqdm(range(len(eval_dataloader)))

            
            for _, batch in enumerate(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                try:
                    with torch.no_grad():
                        outputs = model(**batch)
            
                        logits = outputs["prosody_logits"]
                        predictions = logits
                        

                        labels = batch["prosodic_labels"].reshape(-1)
                        predictions = predictions.reshape(-1)
                        predictions_all.append(predictions.cpu().detach().numpy())
                        metric.add_batch(predictions=predictions, references=labels)
                    progress_bar.update(1)
                except:
                    continue
            
            progress_bar.close()
            
            with open(args.file_output, 'w') as file:
                for ii,prediction in enumerate(predictions_all):
                    file.write(dataset["eval"][ii]["audio"]["path"])
                    file.write(",[")
                    prediction.tofile(file,sep=" ", format="%s")
                    file.write("]\n")
            
        if asr_evaluation:
            if use_BURNC == True:
                pred, actual = inference(dataset["eval"], './sequential_combined_asr_prosody_ss3/checkpoint-30000')
            else:
                print('Providing predictions:')
                pred, actual = inference(processed_dataset_test, './sequential_combined_asr_prosody_ss3/checkpoint-30000')
            calculate_wer_and_cer(pred, actual)
            



        