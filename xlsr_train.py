import sys
import torch
import optuna
from functools import partial
import joblib
import torch.utils.data 
import torch.optim as optim
import torch.nn as nn
import json
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForPreTraining
from models import *
import wandb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from jiwer import wer
from torch.cuda.amp import autocast, GradScaler
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Declare global variables to track the values of max prosody len and max asr len


MAX_PROSODY_LABELS_LEN = 0
MAX_ASR_LABELS_LEN = 0


# Load tokenizer from  vocab.json
'''
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file="vocab.json",  
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
'''
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file="new_vocab.json",  
    unk_token="[UNK]",
    pad_token="[PAD]"
)

# add an explicit CTC blank that never appears in transcripts
#tokenizer.add_tokens(["<ctc_blank>"])

print(tokenizer.get_vocab())  # returns a dict {token_string: token_id}

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(
                        feature_size = 1,
                        sampling_rate = 16000,
                        padding_value=0.0,
                        do_normalize=True,
                        return_attention_mask=True
                    )


processor =  Wav2Vec2Processor(
                            feature_extractor=feature_extractor,
                            tokenizer=tokenizer
                            )


wandb.init(project="multitask-speech-model-xlsr", name="experiment_1")




# set global device
device = "cuda" if torch.cuda.is_available() else "cpu"



class SERDataset(Dataset):
    def __init__(self, data_json):
        '''
        Initialize the class variables of the SERDataset class

        Params:
            train_data_json (obj): JSON object containing the training data.
        
        Returns:
            Batch of input features
        '''
        # unpack the values of the training data
        self.audio_features = data_json['audio']
        self.prosody_features = data_json['prosody']
        self.asr_features = data_json['asr']
        self.emotion_labels = data_json['emotion']
    

    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, index):
        #check to be sure if the asr features are a list and if they are not, make them so before returning them
        if isinstance(self.asr_features[index], (list)):
            return (
                        self.audio_features[index],
                        self.prosody_features[index],
                        self.asr_features[index],
                        self.emotion_labels[index]
                    )
        else:
            return (
                        self.audio_features[index],
                        self.prosody_features[index],
                        [self.asr_features[index]],
                        self.emotion_labels[index]
                    )
    



def get_max_prosody_label_len(train_data, val_data, test_data):
    prosody_vectors = train_data['prosody'] + val_data['prosody'] + test_data['prosody']
    
    prosody_labels_len = []
    for prosody_vector in prosody_vectors:
        prosody_labels_len.append(len(prosody_vector))

    return max(prosody_labels_len)



def get_max_asr_label_len(train_data, val_data, test_data):
    asr_vectors = train_data['asr'] + val_data['asr'] + test_data['asr']

    asr_labels_len = []
    for asr_vector in asr_vectors:
        if isinstance(asr_vector, (list)):
            asr_labels_len.append(len(asr_vector))
        else:
            print(f"Warning: Expected list but got {type(asr_vector)} - value: {asr_vector}")
            asr_labels_len.append(1)


    return max(asr_labels_len)






def collate_fn(batch, max_prosody_len):
    """
    batch: List of tuples (raw_waveform_tensor, prosody_labels, asr_labels, emotion_label)
    """
    # Unpack raw audio and labels
    raw_waveforms   = [item[0] for item in batch]  # List of 1D Tensors
    prosody_labels  = [torch.tensor(item[1]) for item in batch]
    asr_labels      = [torch.tensor(item[2]) for item in batch]
    emotion_labels  = [torch.tensor(item[3]) for item in batch]

    # 1) Use processor to convert raw_waveforms → input_values + attention_mask (batched)
    inputs = processor(
        raw_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,          # pad to the longest waveform in this batch
    )
    input_values   = inputs.input_values        # shape (B, T_max)
    attention_mask = inputs.attention_mask      # shape (B, T_max)

    #get the audio lengths
    audio_lengths = [len(audio_feature) for audio_feature in raw_waveforms]
    audio_lengths = torch.tensor(audio_lengths)


    # 2) Pad ASR labels
    asr_lens   = [lbl.size(0) for lbl in asr_labels]
    asr_labels = pad_sequence(asr_labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    asr_lengths = torch.tensor(asr_lens)

    # 3) Pad prosody labels to max_prosody_len
    prosody_lens   = [lbl.size(0) for lbl in prosody_labels]
    prosody_labels = pad_sequence(prosody_labels, batch_first=True)  # (B, L₁)
    pad_diff       = max_prosody_len - prosody_labels.size(1)
    if pad_diff > 0:
        prosody_labels = F.pad(prosody_labels, (0, pad_diff), value=0)
    prosody_lengths = torch.tensor(prosody_lens)

    # 4) Batch emotion_labels
    emotion_labels = torch.tensor(emotion_labels)

    return (
        input_values,     # (B, T_max)
        attention_mask,   # (B, T_max)
        audio_lengths,    # (B,)
        asr_labels,       # (B, L_asr)
        asr_lengths,      # (B,)
        prosody_labels,   # (B, L_prosody_max)
        prosody_lengths,  # (B,)
        emotion_labels    # (B,)
    )


def compute_wer(pred, true):
    '''
    Computes the word error rate given a list of predicted strings and true label strings.

    Params:
        pred (list): List of predicted strings.
        true (list): List of true strings.
    
    Returns:
        generated_wer (float): Computed word error rate.
    '''
    # get a list containing pairs of predicted and true strings 
    pairs = [(p,t) for p,t in zip(pred, true) if t.strip() != ""]

    if pairs:
        pred_labels, true_labels = zip(*pairs)
        generated_wer = wer(list(true_labels), list(pred_labels))
    else:
        generated_wer = float("nan")

    return generated_wer

def flatten(nested_list):
    '''
    Flattens a nested list.

    Params:
        nested_list (list): List containing lists
    
    Returns:
        flattened_list (list): Flattened list from the nested lists
    '''
    flattened_list = [item for list_item in nested_list for item in list_item]

    return flattened_list


def test(model_checkpoint_path, test_data, batch_size):
    '''
    Tests the trained model.

    Params:
        model_checkpoint_path (str): Path to the trained model checkpoint
        test_data (obj): This is the test data json file
    
    Returns:
        wer,
        prosody_f1,
        prosody_precision, 
        prosody_recall, 
        prosody_accuracy,
        emotion_accuracy,
        emotion_precision,
        emotion_recall,
        emotion_weighted_f1,
        emotion_macro_f1
    '''
    global MAX_PROSODY_LABELS_LEN
    global MAX_ASR_LABELS_LEN
    global device

    wav2vec2_model_base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")

    base_dict = wav2vec2_model_base_config.to_dict()

    # remove any existing entries we intend to override
    for k in ("pad_token_id", "vocab_size"):
        base_dict.pop(k, None)

    # create a new custom config file
    custom_cfg = Wav2Vec2Config(
                                    **base_dict,
                                    pad_token_id = tokenizer.pad_token_id,    
                                    vocab_size = tokenizer.vocab_size
                                )

    #get the configuration for the xlsr model using the new custom config
    w2v_base_model = Wav2Vec2ForCTC.from_pretrained(
                                            "facebook/wav2vec2-xls-r-300m", 
                                            config=custom_cfg
                                        )
    #enable gradient checkpointing on the model
    w2v_base_model.gradient_checkpointing_enable()
    #blank_id = tokenizer.convert_tokens_to_ids("<ctc_blank>")

    # Initialize the Model
    model = Wav2Vec2MTL(
                                config=custom_cfg,
                                tokenizer=tokenizer,
                                pretrained_wav2vec_model=w2v_base_model,
                                emotion_model_output_size=9,
                                asr_model_output_size=len(tokenizer),
                                prosodic_prominence_model_output_size=MAX_PROSODY_LABELS_LEN,
                                prosody_model_lstm_hidden_size=192
                            ).to(device)
    

    # load the checkpoints of the model
    model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'))

    # move the loaded model to the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # initialise lists to hold the asr results
    all_pred_asr = []
    all_true_asr = []

    # initialise lists to hold the prosodic annotation results
    all_pred_prosodic_annotation = []
    all_true_prosodic_annotation = []

    # initialise lists to hold the emotion results
    all_true_emotion = []
    all_pred_emotion = []

    # obtain a data loader for the test dataset
    test_dataset = SERDataset(test_data)

    test_loader = DataLoader(
                            test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=lambda batch: collate_fn(batch, MAX_PROSODY_LABELS_LEN)
                        
                        )

    #ensure that now gradients are computed when the model is used to generate an output
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating model ..."):
            (
            audio_features,       # (B, T_max)
            attention_mask,     # (B, T_max)
            audio_lengths,      # (B,)
            asr_labels,         # (B, L_asr)
            asr_lengths,        # (B,)
            prosody_labels,     # (B, L_prosody)
            prosody_lengths,    # (B,)
            emotion_labels      # (B,)
            ) = batch_data

            audio_features   = audio_features.to(device)
            attention_mask   = attention_mask.to(device)
            asr_labels       = asr_labels.to(device)
            prosody_labels   = prosody_labels.to(device)
            emotion_labels   = emotion_labels.to(device)

            model_outputs = model(
                                    audio_features,
                                    attention_mask=attention_mask,
                                    prosodic_prominence_annotation_labels=prosody_labels,
                                    asr_labels = asr_labels,
                                    emotion_labels=emotion_labels,
                                    mtl_head=['asr','prosodic_prominence_annotation','ser']
                                )
                
            #ASR PREDICTIONS
            asr_preds = torch.argmax(model_outputs['asr_output'], dim=-1).cpu().tolist()
            asr_targets = asr_labels.cpu().tolist()

            #convert ids to strings
            for pred_ids, target_ids in zip(asr_preds, asr_targets):
                pred_tokens = tokenizer.batch_decode([pred_ids], skip_special_tokens=True)[0]
                target_tokens = tokenizer.batch_decode([target_ids], skip_special_tokens=True)[0]
                all_pred_asr.append(pred_tokens)
                all_true_asr.append(target_tokens)

            #PROSODY PREDICTIONS
            pred_prosody = (torch.sigmoid(model_outputs['prosody_logits']) > 0.5).int().cpu().tolist()
            true_prosody = prosody_labels.int().cpu().tolist()

            for pred, true in zip(pred_prosody, true_prosody):
                all_pred_prosodic_annotation.extend(pred)
                all_true_prosodic_annotation.extend(true)
            
            #EMOTION PREDICTIONS
            pred_emotion = torch.argmax(model_outputs['emotion_logits'], dim=1).cpu()
            all_pred_emotion.extend(pred_emotion.tolist())
            all_true_emotion.extend(emotion_labels.cpu().tolist())
        
        #print('true prosodic annotation: ', all_true_prosodic_annotation)
        #print('predicted prosodic annotation: ', all_pred_prosodic_annotation)
        #print('true asr: ', all_true_asr)
        #print('predicted asr: ', all_pred_asr)


        
        metrics = {
                    'wer': compute_wer(all_pred_asr, all_true_asr),
                    'prosody_accuracy': accuracy_score(all_true_prosodic_annotation, all_pred_prosodic_annotation),
                    'prosody_precision': precision_score(all_true_prosodic_annotation, all_pred_prosodic_annotation, zero_division=0),
                    'prosody_recall': recall_score(all_true_prosodic_annotation, all_pred_prosodic_annotation, zero_division=0),
                    'prosody_f1': f1_score(all_true_prosodic_annotation, all_pred_prosodic_annotation, zero_division=0),
                    'emotion_accuracy': accuracy_score(all_true_emotion, all_pred_emotion),
                    'emotion_precision': precision_score(all_true_emotion, all_pred_emotion, average='macro', zero_division=0),
                    'emotion_recall': recall_score(all_true_emotion, all_pred_emotion, average='macro', zero_division=0),
                    'emotion_macro_f1': f1_score(all_true_emotion, all_pred_emotion, average='macro', zero_division=0),
                    'emotion_weighted_f1': f1_score(all_true_emotion, all_pred_emotion, average='weighted', zero_division=0)
                }
        

        print('Metrics: ', metrics)


    return metrics

def train(train_data, val_data, test_data, training_date, checkpoint_id, lstm_hidden_size, batch_size=1, epochs=50, lr=1e-4, alpha_ctc=1.0, alpha_ser=1.0, alpha_prosody=1.0):
    '''
    Trains the specified model using the provided data points.

    Params:
        train_data (obj): This is the train data json file.
        val_data (obj): This is the validation data json file.
        batch_size (int): This is the size of the batch of data fed into the model for training or validation.

    Returns:
        trained_model_ckpt_path (str): Path to the stored trained model checkpoint.
    '''
    wandb.init(
        project="multitask-speech-model",
        name="experiment_1",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "alpha_ctc": alpha_ctc,
            "alpha_ser": alpha_ser,
            "alpha_prosody": alpha_prosody
        }
    )

    global MAX_PROSODY_LABELS_LEN
    global MAX_ASR_LABELS_LEN
    global device

    # create the dataset
    train_dataset = SERDataset(train_data)
    val_dataset = SERDataset(val_data)

    # get the max len of prosody labels and asr labels
    MAX_PROSODY_LABELS_LEN = get_max_prosody_label_len(train_data, val_data, test_data)
    MAX_ASR_LABELS_LEN = get_max_asr_label_len(train_data, val_data, test_data)

    # construct the data loaders
    train_loader = DataLoader(
                                train_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=lambda batch: collate_fn(batch, MAX_PROSODY_LABELS_LEN)
                            
                            )
    val_loader = DataLoader(
                                val_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                collate_fn=lambda batch: collate_fn(batch, MAX_PROSODY_LABELS_LEN)
                            )


    wav2vec2_model_base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")

    base_dict = wav2vec2_model_base_config.to_dict()

    # remove any existing entries we intend to override
    for k in ("pad_token_id", "vocab_size"):
        base_dict.pop(k, None)

    # create a new custom config file
    custom_cfg = Wav2Vec2Config(
                                    **base_dict,
                                    pad_token_id = tokenizer.pad_token_id,    
                                    vocab_size = tokenizer.vocab_size
                                )

    #get the configuration for the xlsr model using the new custom config
    model = Wav2Vec2ForCTC.from_pretrained(
                                            "facebook/wav2vec2-xls-r-300m", 
                                            config=custom_cfg
                                        )
    #enable gradient checkpointing on the model
    model.gradient_checkpointing_enable()

    #blank_id = tokenizer.convert_tokens_to_ids("<ctc_blank>")

    # Initialize the Model
    xlsr_model = Wav2Vec2MTL(
                                config=custom_cfg,
                                tokenizer=tokenizer,
                                pretrained_wav2vec_model=model,
                                emotion_model_output_size=9,
                                asr_model_output_size=len(tokenizer),
                                prosodic_prominence_model_output_size=MAX_PROSODY_LABELS_LEN,
                                prosody_model_lstm_hidden_size=lstm_hidden_size
                            ).to(device)

    # Freeze the xlsr base model
    #xlsr_model.freeze_xlsr_params()

    #state of model parameters
    #model_params_frozen = True

    '''
        #modify the asr model head to decrease its bias towards just printing out the blank id all the time
    with torch.no_grad():
        # push <blank> bias way down so the model won't default to predicting all blanks
        xlsr_model.asr_head.bias[blank_id] = -3.0
    '''


 
    # set the optimizer
    #optimizer = AdamW(xlsr_model.parameters(), lr=lr)
    head_params      = [p for n,p in xlsr_model.named_parameters() if "asr_head" in n]
    encoder_params   = [p for n,p in xlsr_model.named_parameters() if "asr_head" not in n]

    optimizer = AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": head_params   , "lr": 5e-4},
    ])

    #initialize the scaler
    scaler = GradScaler()

    #validation losses across all epochs of training
    collated_val_loss_across_epochs = []

    # commence training loop
    for epoch in range(epochs):
        xlsr_model.train()

        # initialize epoch loss
        loss = 0.0

        # iterate through the batches to perform training
        for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            (
            audio_features,       # (B, T_max)
            attention_mask,     # (B, T_max)
            audio_lengths,      # (B,)
            asr_labels,         # (B, L_asr)
            asr_lengths,        # (B,)
            prosody_labels,     # (B, L_prosody)
            prosody_lengths,    # (B,)
            emotion_labels      # (B,)
            ) = batch_data

            audio_features   = audio_features.to(device)
            attention_mask   = attention_mask.to(device)
            asr_labels       = asr_labels.to(device)
            prosody_labels   = prosody_labels.to(device)
            emotion_labels   = emotion_labels.to(device)


            try:
                #use autocast to reduce the precision of the floating point computations
                with autocast():
                    # perform a forward pass to the xlsr model
                    model_outputs = xlsr_model(
                                                    audio_features=audio_features,
                                                    attention_mask=attention_mask,
                                                    asr_labels=asr_labels,
                                                    prosodic_prominence_annotation_labels=prosody_labels,
                                                    emotion_labels=emotion_labels,
                                                    mtl_head=['asr','prosodic_prominence_annotation','ser']
                                                )
                    


                    # compute the loss 
                    computed_loss = (alpha_ctc*model_outputs['asr_loss']) + (alpha_prosody*model_outputs['prosody_loss']) + (alpha_ser*model_outputs['ser_loss'])

                # clear the gradients from the previous forward pass, calculate the new gradients and take the optimization step
                optimizer.zero_grad()
                scaler.scale(computed_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                #empty cache memory to free up memory space in the GPU
                torch.cuda.empty_cache()
                loss += computed_loss.item()

                # delete the computed loss since it is no longer being used to free up memory space.
                del computed_loss

                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('OOM in batch - skipping')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

            #delete the features that the we no longer need to free up memory space
            for obj in [model_outputs, audio_features, asr_labels, prosody_labels, emotion_labels]:
                del obj

        

        # compute the average loss across the batches for the current epoch
        avg_loss = loss / len(train_loader)

        # print the epoch number and the average loss
        print(f"Epoch {epoch+1}/{epochs} -- Average Train Loss: {avg_loss:.4f}")

        wandb.log({
            "avg_train_loss": avg_loss,
            "epoch": epoch
        })

        # Perform a validation step after the training epoch to ascertain if the model if getting better
        xlsr_model.eval()

        val_loss = 0.0

        val_count = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}"):
                (
                audio_features,       # (B, T_max)
                attention_mask,     # (B, T_max)
                audio_lengths,      # (B,)
                asr_labels,         # (B, L_asr)
                asr_lengths,        # (B,)
                prosody_labels,     # (B, L_prosody)
                prosody_lengths,    # (B,)
                emotion_labels      # (B,)
                ) = batch_data

                audio_features   = audio_features.to(device)
                attention_mask   = attention_mask.to(device)
                asr_labels       = asr_labels.to(device)
                prosody_labels   = prosody_labels.to(device)
                emotion_labels   = emotion_labels.to(device)

                if val_count <= 50:
                    try:
                        # perform a forward pass to the xlsr model
                        model_outputs = xlsr_model(
                                                        audio_features,
                                                        attention_mask=attention_mask,
                                                        prosodic_prominence_annotation_labels=prosody_labels,
                                                        asr_labels = asr_labels,
                                                        emotion_labels=emotion_labels,
                                                        mtl_head=['asr','prosodic_prominence_annotation','ser']
                                                    )
                        

                        #print the asr output of the val set to check on the progress of the model
                        #ASR PREDICTIONS
                        all_pred_asr = []
                        all_true_asr = []

                        asr_preds = torch.argmax(model_outputs['asr_output'], dim=-1).cpu().tolist()
                        print('asr preds: ', asr_preds)
                        asr_targets = asr_labels.cpu().tolist()

                        #convert ids to strings
                        for pred_ids, target_ids in zip(asr_preds, asr_targets):
                            pred_tokens = tokenizer.batch_decode([pred_ids], skip_special_tokens=False)[0]
                            target_tokens = tokenizer.batch_decode([target_ids], skip_special_tokens=False)[0]
                            all_pred_asr.append(pred_tokens)
                            all_true_asr.append(target_tokens)
                        
                        print('all true asr: ', all_true_asr)
                        print('all print asr: ', all_pred_asr)


                        # compute the loss 
                        computed_val_loss = (alpha_ctc*model_outputs['asr_loss']) + (alpha_prosody*model_outputs['prosody_loss']) + (alpha_ser*model_outputs['ser_loss'])
                        val_loss += computed_val_loss.item()

                        val_count += 1

                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('OOM in batch - skipping')
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                else:
                    break
            
                
            # compute the average of the validation loss
            avg_val_loss = val_loss / len(val_loader)

            #collect the avg_val_loss computed
            collated_val_loss_across_epochs.append(avg_val_loss)


            # print the validation loss
            print(f"Epoch {epoch+1}/{epochs} -- Average Validation Loss: {avg_val_loss:.4f}")

            wandb.log({
                "avg_val_loss": avg_val_loss,
                "epoch": epoch
            })
        

        #unfreeze the model after some number of epochs
        '''
        if epoch >= 3:
            if model_params_frozen == True:
                xlsr_model.unfreeze_xlsr_model()
                model_params_frozen = False
        '''



    
    #create the path to the trained model checkpoint
    model_checkpoint_path = f"./model_checkpoints/xlsr_multitask_model_{training_date}_{checkpoint_id}.pt"

    # save the final model 
    torch.save(xlsr_model.state_dict(), model_checkpoint_path)

    print("Training for the model is complete!")

    #compute the average val loss across all of the training epochs to be used for
    #hyperparameter optimization
    avg_collated_val_loss = sum(collated_val_loss_across_epochs) / len(collated_val_loss_across_epochs)

    return model_checkpoint_path, avg_collated_val_loss


def train_with_hugging_face_trainer():
    return

#train(train_data, val_data, test_data, training_date, checkpoint_id, batch_size=1, epochs=50, lr=1e-4, alpha_ctc=1.0, alpha_ser=1.0, alpha_prosody=1.0)
def objective(trial, train_data, val_data, test_data, training_date, checkpoint_id):
    #hyperparamter suggestions for the multi-task wav2vec2 model
    suggested_lr = trial.suggest_float('lr', 1e-8, 1e-2, log=True)
    suggested_alpha_ctc = trial.suggest_float('alpha_ctc', 0.1, 1.0, step=0.1, log=False)
    suggested_alpha_ser = trial.suggest_float('alpha_ser', 0.1, 1.0, step=0.1, log=False)
    suggested_alpha_prosody = trial.suggest_float('alpha_prosody', 0.1, 1.0, step=0.1, log=False)


    #hyperparameter suggestions for the LSTM hidden dimension size for the prosody arm
    suggested_lstm_hidden_size = trial.suggest_int('prosody_model_lstm_hidden_size', 128, 512, step=64)

    #train the model to obtain the value to be optimised for
    model_checkpoint_path, avg_collated_val_loss = train(
                                                            train_data, 
                                                            val_data, 
                                                            test_data, 
                                                            training_date=training_date, 
                                                            checkpoint_id=checkpoint_id, 
                                                            lstm_hidden_size=suggested_lstm_hidden_size,
                                                            batch_size=1, 
                                                            epochs=5, 
                                                            lr=suggested_lr, 
                                                            alpha_ctc=suggested_alpha_ctc, 
                                                            alpha_ser=suggested_alpha_ser, 
                                                            alpha_prosody=suggested_alpha_prosody
                                                        )
    return avg_collated_val_loss


def perform_hyperparameter_search(num_trials, train_data, val_data, test_data, training_date, checkpoint_id):
    '''
    Performs a hyperparameter search to find the best combination of hyperparameters to lead to the best model
    performance.

    Params:
        num_trials (int): The number of search trials to conduct.
        train_data (obj): Training data
        val_data (obj): Validation data
        test_data (obj): Test data
        training_date (str): Date for commencing training.
        checkpoint_id (int): Id to represent index of checkpoint.


    Returns:
        best_hyperparameters (obj): JSON object containing the best found hyperparameters.
    '''
    wrapped_objective = partial(objective, train_data=train_data, val_data=val_data, test_data=test_data, training_date=training_date, checkpoint_id=checkpoint_id)
    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=num_trials)

    try:
        best_found_hyperparameters = study.best_params
        lowest_computed_val_loss = study.best_value

        print('best found hyperparameters: ', best_found_hyperparameters)
        print('best computed val loss: ', lowest_computed_val_loss)

        #save the best found hyperparameter values
        with open('optimal_hyperparameters.json', 'w') as input_json:
            json.dump(study.best_params, input_json, indent=4)
        
        return best_found_hyperparameters
    
    except Exception as e:
        print('Encountered error: ', e)
    

if __name__ == '__main__':
    #get the path to where the data is stored as a command-line argument
    data_path = sys.argv[1]

    #get the date in which the training script is being run
    cur_date = sys.argv[2]

    #get the current experiment run 
    cur_exp_run = int(sys.argv[3])

    #read the json object from the given data path
    with open(data_path, 'r') as input_json:
        input_data = json.load(input_json)
    
    #extract the training, val and test data
    train_data = input_data['train']
    test_data = input_data['test']
    val_data = input_data['val']


    #perform hyper-parameter search
    '''
    perform_hyperparameter_search(
                                10, 
                                train_data,
                                val_data,
                                test_data,
                                cur_date,
                                cur_exp_run
                                )
    '''



    #train(train_data, val_data, test_data, training_date, checkpoint_id, lstm_hidden_size, batch_size=1, epochs=50, lr=1e-4, alpha_ctc=1.0, alpha_ser=1.0, alpha_prosody=1.0)
    #train the model
    #checkpoint_pth, avg_val_loss = train(train_data, val_data, test_data, cur_date, cur_exp_run, 192, batch_size=1, epochs=60, lr=1.747962919342474e-06, alpha_ctc=0.6, alpha_ser=0.1, alpha_prosody=0.6)

    checkpoint_pth, avg_val_loss = train(train_data, val_data, test_data, cur_date, cur_exp_run, 192, batch_size=100, epochs=500, lr=3e-4, alpha_ctc=1.0, alpha_ser=0.0, alpha_prosody=0.0)

    #checkpoint_pth = "/home/dasa/ser_project/codebase/model_checkpoints/xlsr_multitask_model_17_05_1.pt"

    #MAX_PROSODY_LABELS_LEN = get_max_prosody_label_len(train_data, val_data, test_data)
    #test the model
    #test(checkpoint_pth, test_data, 1)

