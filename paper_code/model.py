"""model.py

Neural network model for multi-task speech learning with a shared encoder and
three task-specific heads:

Main task
---------
Emotion recognition (multi-class softmax)

Auxiliary tasks
---------------
1. Automatic speech recognition – Connectionist Temporal Classification (CTC)
2. Prosody classification – binary (or *n*-class) softmax per word

Loss: ``L = L_emotion + α·L_ctc + β·L_prosody`` where α, β are provided at
initialisation.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

class MTLModel(PreTrainedModel):
    """Multi-task model with a shared HF backbone (e.g. Wav2Vec2/Whisper).

    Parameters
    ----------
    config : transformers.PretrainedConfig
        Backbone configuration.
    prosody_cls_len : int, default 2
        Number of prosody classes (0: non-prosodic, 1: prosodic).
    ser_cls_len : int, default 9
        Number of emotion classes.
    alpha : float, default 0.1
        Weight for ASR CTC loss.
    beta : float, default 0.1
        Weight for prosody loss.
    **kwargs
        Extra args forwarded from ``from_pretrained`` (e.g. ``vocab_size``).
    """
    # allow `from_pretrained` to fetch a config automatically
    config_class = AutoConfig

    def __init__(
        self,
        config,
        prosody_cls_len: int = 2,  # Changed default to 2 for binary classification
        ser_cls_len: int = 9,
        alpha: float = 0.1,
        beta: float = 0.1,
        **kwargs,
    ):
        # inherit the config from PreTrainedModel
        super().__init__(config)
        # build the shared backbone from the provided configuration
        self.model = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.final_dropout)
        # update config vocab_size if passed via **kwargs (e.g., from_pretrained parameter)
        if "vocab_size" in kwargs and kwargs["vocab_size"] is not None:
            config.vocab_size = kwargs["vocab_size"]

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)  # language model head
        self.prosody_head = nn.Linear(config.hidden_size, prosody_cls_len) # prosody classification head (per-frame)
        self.ser_head = nn.Linear(config.hidden_size, ser_cls_len) # speech emotion recognition head
        self.alpha = alpha  # weight for ASR loss
        self.beta = beta    # weight for Prosody loss
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Freeze the feature extractor layers of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        warnings.warn("Feature extractor layers are frozen. Only task-specific heads will be trained.")
        
    # Asr loss 
    def _ctc_loss(self, logits: torch.Tensor, labels: torch.Tensor, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the CTC loss for ASR task.
        
        Args:
            logits (torch.Tensor): Logits from the model.
            labels (torch.Tensor): Ground truth labels for ASR.
            input_values (torch.Tensor): Input audio values.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            
        Returns:
            torch.Tensor: Computed CTC loss.
        """
        loss = None
        if labels is not None:
            # get input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )

            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            # assuming the padded tokens are -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return loss
    
    # Prosody loss - now handles sequence-level predictions
    def _prosody_loss(self, logits: torch.Tensor, pro_labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the cross-entropy loss for word-level Prosody task.
        
        Args:
            logits (torch.Tensor): Logits from the model (batch_size, seq_len, num_classes).
            pro_labels (torch.Tensor): Ground truth labels for Prosody (batch_size, word_len).
            attention_mask (Optional[torch.Tensor]): Attention mask for valid positions.
            
        Returns:
            torch.Tensor: Computed Prosody loss.
        """
        loss = None
        if pro_labels is not None:
            # Get the device of logits
            device = logits.device
            
            # Move pro_labels to the same device as logits
            pro_labels = pro_labels.to(device)
            
            # The prosody labels are at the word level (batch_size, num_words)
            # The logits are at the frame level (batch_size, num_frames, num_classes)
            # We need to handle this mismatch
            
            # For now, we'll average pool the frame-level predictions to match word count
            # This is a simplification - in practice you might want to use alignments
            
            batch_size = logits.size(0)
            num_frames = logits.size(1)
            num_classes = logits.size(2)
            
            # Get the number of words for each example (excluding padding)
            word_lengths = (pro_labels != -100).sum(dim=1)
            max_word_length = word_lengths.max().item()
            
            if max_word_length == 0:
                # No valid prosody labels in this batch
                return torch.tensor(0.0, device=device)
            
            # Create pooled logits for word-level predictions
            pooled_logits = []
            
            for b in range(batch_size):
                num_words = word_lengths[b].item()
                if num_words == 0:
                    # No words for this example, add dummy logits
                    pooled_logits.append(torch.zeros(max_word_length, num_classes, device=device))
                    continue
                
                # Divide frames equally among words
                frames_per_word = num_frames // num_words if num_words > 0 else num_frames
                
                word_logits = []
                for w in range(max_word_length):
                    if w < num_words:
                        # Average pool frames for this word
                        start_frame = w * frames_per_word
                        end_frame = (w + 1) * frames_per_word if w < num_words - 1 else num_frames
                        word_logit = logits[b, start_frame:end_frame, :].mean(dim=0)
                        word_logits.append(word_logit)
                    else:
                        # Padding
                        word_logits.append(torch.zeros(num_classes, device=device))
                
                pooled_logits.append(torch.stack(word_logits))
            
            # Stack to create batch
            pooled_logits = torch.stack(pooled_logits)  # (batch_size, max_word_length, num_classes)
            
            # Flatten for loss computation
            pooled_logits_flat = pooled_logits.view(-1, num_classes)
            pro_labels_flat = pro_labels.view(-1)
            
            # Only compute loss on non-padded positions
            valid_mask = pro_labels_flat != -100
            
            if valid_mask.sum() > 0:
                valid_logits = pooled_logits_flat[valid_mask]
                valid_labels = pro_labels_flat[valid_mask]
                loss = F.cross_entropy(valid_logits, valid_labels.long())
            else:
                loss = torch.tensor(0.0, device=device)
        
        return loss
    
    # SER loss, multi-class classification
    def _ser_loss(self, logits: torch.Tensor, ser_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss for Speech Emotion Recognition task.
        
        Args:
            logits (torch.Tensor): Logits from the model.
            ser_labels (torch.Tensor): Ground truth labels for SER.
            
        Returns:
            torch.Tensor: Computed SER loss.
        """
        loss = None
        if ser_labels is not None:
            loss = F.cross_entropy(logits, ser_labels.to(logits.device))
        return loss

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    # Forward pass
    def forward(
            self,
            input_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None, # tuple: (asr_labels, pro_labels, ser_labels)
            if_asr: bool = True, # whether to compute ASR loss
            if_prosody: bool = True, # whether to compute Prosody loss
            if_ser: bool = True, # whether to compute SER loss
    ):
        """
        Forward pass of the MTL model.
        Args:
            input_values (torch.Tensor): Input audio values.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            output_attentions (Optional[bool]): Whether to return attentions.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs.
            labels (Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Tuple of labels for ASR, Prosody, and SER tasks.
            if_asr (bool): Whether to compute ASR loss.
            if_prosody (bool): Whether to compute Prosody loss.
            if_ser (bool): Whether to compute SER loss.
        Returns:
            Union[Tuple, CausalLMOutput]: Model outputs, either as a tuple or a dictionary. 
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # model outputs = last hidden state
        hidden_states = outputs[0] # last layer's hidden states
        hidden_states = self.dropout(hidden_states) # apply dropout

        logits_ctc = self.lm_head(hidden_states)  # logits for ASR (batch_size, seq_len, vocab_size)
        logits_prosody = self.prosody_head(hidden_states)  # logits for Prosody (batch_size, seq_len, prosody_cls_len)
        logits_ser = self.ser_head(torch.mean(hidden_states, dim=1))  # logits for SER (batch_size, ser_cls_len)

        # compute losses
        loss = None
        loss_ctc = 0
        loss_prosody = 0
        loss_ser = 0
        
        if labels is not None:
            if if_asr and labels[0] is not None:
                loss_ctc = self._ctc_loss(logits_ctc, labels[0], input_values, attention_mask)
            if if_prosody and labels[1] is not None:
                # For prosody, we pass the frame-level attention mask
                # The loss function will handle the word-level alignment
                loss_prosody = self._prosody_loss(logits_prosody, labels[1], attention_mask)
            if if_ser and labels[2] is not None:
                loss_ser = self._ser_loss(logits_ser, labels[2])
            
            # Combine losses
            loss = loss_ser
            if if_asr:
                loss = loss + self.alpha * loss_ctc
            if if_prosody:
                loss = loss + self.beta * loss_prosody

        return CausalLMOutput(
            loss=loss,
            logits=(logits_ctc, logits_prosody, logits_ser),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _get_prosody_attention_mask(self, attention_mask):
        """
        Convert input attention mask to output attention mask based on convolution layers.
        This function is kept for compatibility but not used in the new approach.
        """
        # Get the output lengths after convolution layers
        input_lengths = attention_mask.sum(dim=-1)
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        
        # Create new attention mask
        batch_size = attention_mask.size(0)
        max_length = self._get_feat_extract_output_lengths(torch.tensor([attention_mask.size(1)])).item()
        
        prosody_attention_mask = torch.zeros((batch_size, max_length), dtype=attention_mask.dtype, device=attention_mask.device)
        for i, length in enumerate(output_lengths):
            prosody_attention_mask[i, :length] = 1
            
        return prosody_attention_mask