
'''
Multi-task Learning framework that perfomrs speech emotion recognition as well as automatic speech recognition and prosodic prominence annotation.
The task is to ascertain if prosodic prominence annotation labels provided by annotators are useful for speech emotion recognition.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from torch.nn import CTCLoss, BCEWithLogitsLoss, CrossEntropyLoss
import datasets
from transformers import Wav2Vec2Config
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

            ctc_loss = CTCLoss(blank=self.config.pad_token_id, zero_infinity=True)

            total_loss += ctc_loss(log_probs, asr_labels, input_lengths, target_lengths)

        if 'prosodic_prominence_annotation' in mtl_head:
            bce_loss = BCEWithLogitsLoss()

            total_loss += bce_loss(prosodic_prominence_label_logits, prosodic_prominence_annotation_labels)

        
        if 'ser' in mtl_head:
            ce_loss = CrossEntropyLoss()

            total_loss += ce_loss(emotion_logits, emotion_labels)
        


        return {
                    "asr_output": asr_prob_labels,
                    "prosody_logits": prosodic_prominence_label_logits,
                    "loss": total_loss
                }







        
        








