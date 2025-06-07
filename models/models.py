import json 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import adaptive_avg_pool1d

from transformers import Wav2Vec2Model
import datasets
from transformers import Wav2Vec2Config

import soundfile as sf
import numpy as np
import random
from pydub import AudioSegment
from io import BytesIO
import os

# ESPnet imports
from espnet2.tasks.ssl import SSLTask

# set global device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 2. Multi-Task Model: Shared Xeus backbone + Task-specific heads
class XeusMTL(nn.Module):
    def __init__(
        self,
        xeus_checkpoint_path,
        asr_vocab_size= 32, # using the vocab size from the SER paper
        num_emotions = 9, # number of emotions in the dataset (0-8)
        hidden_dim = 1024,  # Changed from 768 to 1024 to match Xeus output dimension
        prosody_hidden_dim = 256,
        use_asr = True,
        use_prosody = True,
        use_ser = True,
    ):
        super().__init__()

        # Load pre-trained Xeus
        # Checkpoint path is local, so you need it for model training
        # passing None to config_path for demonstration

        xeus_model, _ = SSLTask.build_model_from_file(
            config_file=None,
            model_file=xeus_checkpoint_path,
            device=device
        )

        self.xeus_model = xeus_model
        self.hidden_dim = hidden_dim

        # ASR head for CTC
        self.use_asr = use_asr
        if self.use_asr:
            self.asr_fc = nn.Linear(hidden_dim, asr_vocab_size) # final projection for CTC
        
        # SER head (multi-class classification)
        self.use_ser = use_ser
        if self.use_ser:
            self.ser_fc = nn.Linear(hidden_dim, num_emotions)

        # Prosody classification head (binary classification for each word)
        self.use_prosody = use_prosody
        if self.use_prosody:
            self.prosody_bilstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size = prosody_hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

            #  Output dimension from BiLSTM = 2*prosody_hidden_dim
            self.prosody_fc = nn.Linear(2 * prosody_hidden_dim, 1) # binary classification for each word

    def forward(self, audio_features, lengths, use_mask=False, prosody_target_len=None):
        """
        waveforms: (batch, time)
        lengths: (batch,)
        prosody_target_len: Optional target length for prosody predictions
        Returns a tuple of (asr_logits, ser_logits, prosody_logits).
        If a head is disabled, returns None in its place.
        """
        # Ensure we have a valid batch dimension to avoid segfaults
        if audio_features.dim() == 1:
            audio_features = audio_features.unsqueeze(0)  # Add batch dimension if missing
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)  # Add batch dimension if missing
            
        with torch.no_grad():
            xeus_outs = self.xeus_model.encode(
                audio_features, 
                lengths, 
                use_mask=use_mask,
                use_final_output=False
            )[0][-1]
        
        asr_logits = None
        ser_logits = None
        prosody_logits = None

        if self.use_asr:
            asr_logits = self.asr_fc(xeus_outs)
            asr_logits = asr_logits.permute(1, 0, 2)
            
            # Ensure CTC loss will work with batch size 1
            if asr_logits.size(1) == 1 and asr_logits.size(0) < 2:
                # CTC requires sequence length > 1, pad if necessary
                asr_logits = F.pad(asr_logits, (0, 0, 0, 0, 0, 1), "constant", 0)

        if self.use_ser:
            pooled_features = xeus_outs.mean(dim=1)
            ser_logits = self.ser_fc(pooled_features)
        
        if self.use_prosody:
            # if no valid prosody_target_len, skip
            if prosody_target_len is None or prosody_target_len <= 0:
                prosody_logits = None
            else:
                B, T, D = xeus_outs.shape
                # Pool from T frame -> P "word-level" frames
                if T < 1:
                    # handle edge case of no frames
                    prosody_logits = torch.zeros(B, prosody_target_len, device=xeus_outs.device)
                else:
                    # Transpose xeus output from (B, T, D) -> (B, D, T)
                    xeus_outs_t = xeus_outs.transpose(1,2)
                    # adaptive_avg_pool1d to shape [B, D, P]
                    pooled_features = F.adaptive_avg_pool1d(xeus_outs_t, prosody_target_len)
                    # Transpose back to (B, P, D)
                    pooled_features = pooled_features.transpose(1,2)

                    # pass through BiLSTM
                    lstm_out, _ = self.prosody_bilstm(pooled_features)

                    # Final FC -> (B, P, 1) -> (B, P)
                    prosody_out = self.prosody_fc(lstm_out).squeeze(-1)
                    # shape: (batch, seq)
                    prosody_logits = prosody_out 
        

        return asr_logits, ser_logits, prosody_logits
    
class Wav2Vec2MTL(nn.Module):
    def __init__(self, 
                 config,
                 pretrained_wav2vec_model, 
                 emotion_model_output_size, 
                 prosodic_prominence_model_output_size, 
                 prosody_model_lstm_hidden_size):
        self.config = config
        self.wav2vec2_model = pretrained_wav2vec_model

        #freeze the encoder of the provided wav2vec2 model
        self.wav2vec2_model.freeze_encoder()

        #initialize the dropout value of the network
        self.dropout = nn.Dropout(self.config.final_dropout)

        #initialize the ASR Component Head of the MTL model
        self.asr_head = nn.Linear(self.config.hidden_size, self.config.vocabulary_size)

        #initialize the Prosodic Prominence Annotation Head of the MTL Model
        self.prosody_model_bilstm = nn.LSTM(self.config.hidden_size, prosody_model_lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.prosodic_prominence_annotation_head = nn.Linear(prosody_model_lstm_hidden_size*2, prosodic_prominence_model_output_size)

        #initialise the Emotion Prediction Model Head of the MTL Model
        self.emotion_prediction_head = nn.Linear(self.config.hidden_size, emotion_model_output_size)
    


    def forward(self,
                audio_features,
                attention_mask=None,
                prosodic_prominence_annotation_labels=None,
                asr_labels=None,
                emotion_labels=None,
                mtl_head=['asr']):
    
        #obtain the shared audio feature representations
        w2v2_model_outputs = self.wav2vec2_model(audio_features, attention_mask=attention_mask)
        extracted_audio_feature_representations = w2v2_model_outputs[0]

        #perform a dropout on the extracted audio feature representations
        processed_extracted_audio_feature_representations = self.dropout(extracted_audio_feature_representations)


        #ASR HEAD
        #push the processed extracted audio representations into the ASR head to generate logits for the ASR labels
        asr_logits = self.asr_head(processed_extracted_audio_feature_representations)

        #generate probability values from the generated asr logits by using the softmax function
        asr_prob_labels = torch.functional.F.log_softmax(asr_logits, dim=-1)

        #EMOTION RECOGNITION HEAD
        #perform a mean pool of the processed extracted features from the w2v2 model
        mean_pool_w2v2_features = processed_extracted_audio_feature_representations.mean(dim=1)

        #push the pooled features into the emotions head
        emotion_logits = self.emotion_prediction_head(mean_pool_w2v2_features)

        #generate probability values from the generated emotion logits using the softmax function
        #emotion_prob_labels = torch.functional.F.log_softmax(emotion_logits, dim=-1)

        #PROSODIC PROMINENCE HEAD
        #push the processed extracted features into the bilstm model
        bilstm_model_hidden_states = self.prosody_model_bilstm(processed_extracted_audio_feature_representations)

        #perform a mean pool of the output of the bilstm model hidden states
        mean_pool_bilstm_hidden_states = bilstm_model_hidden_states.mean(dim=1)

        #push the pooled features into the prosodic prominence annotation head to obtain the logits
        prosodic_prominence_label_logits = self.prosodic_prominence_annotation_head(mean_pool_bilstm_hidden_states)

        #compute the loss for the MTL Model
        total_loss = 0

        if 'asr' in mtl_head:
            #compute the asr head loss using CTC
            log_probs = asr_prob_labels.transpose(0, 1)
            
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)

            target_lengths = torch.tensor([len(asr_labels)], dtype=torch.long)

            ctc_loss = nn.CTCLoss(blank=self.config.pad_token_id, zero_infinity=True)

            total_loss += ctc_loss(log_probs, asr_labels, input_lengths, target_lengths)

        if 'prosodic_prominence_annotation' in mtl_head:
            bce_loss = nn.BCEWithLogitsLoss()

            total_loss += bce_loss(prosodic_prominence_label_logits, prosodic_prominence_annotation_labels)

        
        if 'ser' in mtl_head:
            ce_loss = nn.CrossEntropyLoss()

            total_loss += ce_loss(emotion_logits, emotion_labels)
        


        return {
                    "asr_output": asr_prob_labels,
                    "prosody_logits": prosodic_prominence_label_logits,
                    "loss": total_loss
                }

# 3. Loss Functions for Multi-Task Learning
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha_ctc=1.0, alpha_ser=1.0, alpha_prosody=1.0):
        super().__init__()
        self.alpha_ctc = alpha_ctc
        self.alpha_ser = alpha_ser
        self.alpha_prosody = alpha_prosody

        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.ser_loss_fn = nn.CrossEntropyLoss()
        self.prosody_loss_fn = nn.BCEWithLogitsLoss() # binary classification for each word

    def forward(
        self, 
        asr_logits, # (time, batch, vocab_size) or None
        asr_targets, # (batch, seq_len of tokens) or None
        asr_input_lengths,
        asr_target_lengths,
        ser_logits, # (batch, num_emotions) or None
        ser_targets,
        prosody_logits, # (batch, seq) or None
        prosody_targets,
    ):
        """
            Combine the three losses with weighting factors alpha_ctc, alpha_ser, alpha_prosody.
            Return total_loss, plus each sub-loss for logging.
        """
        loss_ctc = torch.tensor(0.0, device=device)
        loss_ser = torch.tensor(0.0, device=device)
        loss_prosody = torch.tensor(0.0, device=device)

        # ASR: CTC
        if asr_logits is not None:
            loss_ctc = self.ctc_loss_fn(
                asr_logits,
                asr_targets,
                asr_input_lengths,
                asr_target_lengths
            ) * self.alpha_ctc

        # SER: Cross Entropy
        if ser_logits is not None:
            loss_ser = self.ser_loss_fn(ser_logits, ser_targets) * self.alpha_ser

        # Prosody: BCEWithLogits
        if prosody_logits is not None:
            # mask out padded positions
            mask = (prosody_targets != -1)
            valid_logits = prosody_logits[mask]
            valid_targets = prosody_targets[mask]
            loss_prosody = self.prosody_loss_fn(valid_logits, valid_targets) * self.alpha_prosody
        # Total loss
        total_loss = loss_ctc + loss_ser + loss_prosody
        return total_loss, loss_ctc, loss_ser, loss_prosody
            