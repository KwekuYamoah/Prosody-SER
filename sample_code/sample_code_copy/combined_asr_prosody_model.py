import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from torch.nn import CTCLoss, MSELoss
import datasets
from transformers import Wav2Vec2Config
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Wav2Vec2CombinedASR(nn.Module):
    def __init__(self, config, wav2vec2_model):
        super().__init__()

        self.config = config

        # Shared Wav2Vec2 model
        self.wav2vec2 = wav2vec2_model
        self.wav2vec2.freeze_feature_encoder()

        # ASR Component Initialization
        self.asr_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.dropout = nn.Dropout(self.config.final_dropout)

        # Prosodic Labeling Component Initialization
        self.lstm_hidden_size = 256  # Example size, adjust as needed
        self.lstm = nn.LSTM(self.config.hidden_size + 1, self.lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.prosody_classifier = nn.Linear(self.lstm_hidden_size * 2, self.config.num_labels)

    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        config = Wav2Vec2Config.from_pretrained(model_name_or_path, *args, **kwargs)
        wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name_or_path, *args, **kwargs)
        return cls(config, wav2vec2_model)
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Save the model state
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save the configuration
        self.config.save_pretrained(save_directory)
  


    def forward(self, input_values, pitch=None, attention_mask=None, asr_labels=None, prosodic_labels=None):
        include_prosody = False
        include_asr = True

        # Shared feature extraction
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        # ASR Component
        hidden_states = self.dropout(hidden_states)
        asr_logits = self.asr_head(hidden_states)
        asr_output = torch.functional.F.log_softmax(asr_logits, dim=-1)


        # Prosodic Labeling Component
        prosody_logits = None
        if pitch is not None:
            pitch = pitch[0]
            if (len(pitch) < hidden_states.shape[1]):
                pitch = F.pad(pitch, pad=(0, hidden_states.shape[1] - len(pitch)), mode='constant', value=0)
            elif (len(pitch) > hidden_states.shape[1]):
                pitch = pitch[:hidden_states.shape[1]]
            pitch = pitch.reshape(1, len(pitch),1)


        tup = (pitch.to(device, dtype=torch.float),hidden_states)
        x_ = torch.cat(tup,2)
        x_, (_, _) = self.lstm(x_)
        prosody_logits = self.prosody_classifier(x_)

          
        # Loss computation
        loss = None
        if asr_labels is not None or prosodic_labels is not None:
            loss = 0
            if asr_labels is not None and include_asr:
                # CTC Loss for ASR
                log_probs = asr_output.transpose(0, 1)  # Needed for CTC Loss
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
                target_lengths = torch.tensor([len(asr_labels)], dtype=torch.long)
                ctc_loss = CTCLoss(blank=self.config.pad_token_id, zero_infinity=True)
                loss += ctc_loss(log_probs, asr_labels, input_lengths, target_lengths)

            if prosodic_labels is not None and prosody_logits is not None and include_prosody:
                # MSE Loss for Prosody Labeling
                mse_loss = MSELoss()
                loss += mse_loss(prosody_logits.squeeze(2), prosodic_labels)

        #print('loss: ', loss)
        return {
            "asr_output": asr_output,
            "prosody_logits": prosody_logits,
            "loss": loss
        }




def load_json_dataset(file_train,file_eval,file_valid=None):
    data_files = {}
    if file_train is not None:
        data_files["train"] = file_train
    if file_eval is not None:
        data_files["eval"] = file_eval
    if file_valid is not None:
        data_files["valid"] = file_valid

   
    phrasing_features = datasets.Features({
    'path': datasets.features.Value('string'), 
    'label': datasets.features.Sequence(datasets.features.Value(dtype='float64')),
    'text': datasets.features.Value('string'),

    })
   
    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)
    return dataset