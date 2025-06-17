import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModel, WhisperModel
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass

from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.ctc_decoder import CTCDecoder


@dataclass
class MTLOutput(BaseModelOutput):
    """Output type for MTL model"""
    loss: Optional[torch.FloatTensor] = None
    ser_logits: Optional[torch.FloatTensor] = None
    asr_logits: Optional[torch.FloatTensor] = None
    prosody_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_details: Optional[Dict[str, float]] = None


class MTLModel(PreTrainedModel):
    """
    Multi-task learning model.
    Performs SER (main task) + ASR + Prosody (auxiliary tasks)
    """

    config_class = MTLConfig

    def __init__(self, config: MTLConfig):
        # Initialize the base Module class first
        super().__init__(config)

        # Store config
        self.config = config

        # Load backbone using AutoModel
        self.backbone = AutoModel.from_pretrained(
            config.backbone_config.pretrained_model_name,
            config=config.pretrained_config if hasattr(
                config, 'pretrained_config') else None
        )

        # Get the actual hidden size from the backbone
        self.hidden_size = self._get_hidden_size()

        # Dropout layer
        self.dropout = nn.Dropout(config.final_dropout)

        # Task-specific heads
        # SER head (main task) - classification
        self.ser_head = nn.Linear(self.hidden_size, config.emotion_classes)

        # ASR head (auxiliary task) - CTC
        self.asr_head = nn.Linear(self.hidden_size, config.vocab_size)

        # Prosody head (auxiliary task) - sequence labeling
        # Binary classification per frame
        self.prosody_head = nn.Linear(self.hidden_size, 1)

        # Initialize weights
        self.init_weights()

        # Task weights (following paper style)
        self.alpha_asr = config.alpha_asr
        self.alpha_prosody = config.alpha_prosody

        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"MTLModel initialized on device: {device}")

    def _get_hidden_size(self) -> int:
        """Get hidden size from backbone model"""
        if hasattr(self.backbone.config, 'hidden_size'):
            return self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'd_model'):
            return self.backbone.config.d_model
        elif hasattr(self.backbone.config, 'encoder_embed_dim'):
            return self.backbone.config.encoder_embed_dim
        else:
            raise ValueError("Cannot determine hidden size from backbone")

    def freeze_feature_extractor(self):
        """Freeze the feature extractor if available (for wav2vec2-like models)"""
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        elif hasattr(self.backbone, 'freeze_feature_encoder'):
            self.backbone.freeze_feature_encoder()

    def _compute_ctc_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                          input_lengths: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss for ASR task"""
        # Move tensors to the same device as the model
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        input_lengths = input_lengths.to(self.device).to(dtype=torch.long)
        label_lengths = label_lengths.to(self.device).to(dtype=torch.long)

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        # Flatten labels if needed
        if labels.dim() == 2:
            # Labels are padded sequences, need to flatten
            labels_flat = []
            for i, length in enumerate(label_lengths):
                labels_flat.extend(labels[i, :length].tolist())
            labels = torch.tensor(
                labels_flat, dtype=torch.long, device=self.device)

        # Compute CTC loss
        blank_token = getattr(self.config, 'pad_token_id', None)
        if blank_token is None:
            blank_token = 0

        loss = F.ctc_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=blank_token,
            reduction='mean',
            zero_infinity=True
        )

        return loss

    def _compute_ser_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for SER task"""
        # Move tensors to the same device as the model
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        return F.cross_entropy(logits, labels)

    def _compute_prosody_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss for prosody task"""
        # Move tensors to the same device as the model
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        # Flatten both logits and labels
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1).float()

        # Mask out padding (assuming padding value is -100 or 0)
        mask = labels_flat >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        logits_masked = logits_flat[mask]
        labels_masked = labels_flat[mask]

        return F.binary_cross_entropy_with_logits(logits_masked, labels_masked)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MTLOutput]:
        """
        Forward pass through the model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Choose the right input based on model type
        if input_features is not None:
            inputs = {"input_features": input_features.to(self.device)}
        elif input_values is not None:
            inputs = {"input_values": input_values.to(self.device)}
        else:
            raise ValueError(
                "Either input_values or input_features must be provided")

        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(self.device)

        # Handle different model architectures
        if self.config.backbone_name == "whisper":
            # For Whisper, use only the encoder
            if hasattr(self.backbone, 'encoder'):
                encoder = self.backbone.encoder
            elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'encoder'):
                encoder = self.backbone.model.encoder
            else:
                raise ValueError("Cannot find encoder in Whisper model")

            encoder_outputs = encoder(
                **inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            hidden_states = encoder_outputs.last_hidden_state
        else:
            # For other models (wav2vec2, etc.)
            outputs = self.backbone(
                **inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Task-specific predictions
        # ASR: sequence-level predictions
        asr_logits = self.asr_head(hidden_states)

        # SER: utterance-level prediction (pool over time)
        pooled_hidden = torch.mean(hidden_states, dim=1)
        ser_logits = self.ser_head(pooled_hidden)

        # Prosody: sequence-level binary predictions
        prosody_logits = self.prosody_head(hidden_states).squeeze(-1)

        # Compute losses if labels are provided
        loss = None
        loss_details = {}

        if labels is not None:
            # Unpack labels - expecting ((asr_labels, asr_lengths), ser_labels, prosody_labels)
            asr_labels_info, ser_labels, prosody_labels = labels

            # SER loss (main task, weight = 1.0)
            if ser_labels is not None:
                ser_loss = self._compute_ser_loss(
                    ser_logits, ser_labels.to(self.device))
                loss = ser_loss
                loss_details['ser_loss'] = ser_loss.item()

            # ASR loss (auxiliary task, weighted by alpha)
            if asr_labels_info is not None and self.alpha_asr > 0:
                # Compute input lengths from attention mask or assume full length
                input_lengths = torch.full(
                    (asr_logits.size(0),), asr_logits.size(1),
                    dtype=torch.long, device=self.device
                )

                # asr_labels_info should be either:
                # 1. A tuple/list of (labels, lengths)
                # 2. Just labels tensor
                if isinstance(asr_labels_info, (tuple, list)) and len(asr_labels_info) == 2:
                    asr_labels_tensor, label_lengths = asr_labels_info

                    # Move to device if needed
                    if isinstance(asr_labels_tensor, torch.Tensor):
                        asr_labels_tensor = asr_labels_tensor.to(self.device)
                    else:
                        asr_labels_tensor = torch.tensor(
                            asr_labels_tensor, dtype=torch.long, device=self.device)

                    if isinstance(label_lengths, torch.Tensor):
                        label_lengths = label_lengths.to(self.device)
                    else:
                        label_lengths = torch.tensor(
                            label_lengths, dtype=torch.long, device=self.device)

                elif isinstance(asr_labels_info, torch.Tensor):
                    # It's just a tensor of labels
                    asr_labels_tensor = asr_labels_info.to(self.device)
                    # Compute lengths from padding (assuming -100 is padding)
                    label_lengths = (asr_labels_tensor != -100).sum(-1)
                else:
                    # Handle None case
                    asr_labels_tensor = None
                    label_lengths = None

                if asr_labels_tensor is not None:
                    asr_loss = self._compute_ctc_loss(
                        asr_logits, asr_labels_tensor, input_lengths, label_lengths
                    )

                    if loss is None:
                        loss = self.alpha_asr * asr_loss
                    else:
                        loss = loss + self.alpha_asr * asr_loss

                    loss_details['asr_loss'] = asr_loss.item()
                    loss_details['asr_loss_weighted'] = (
                        self.alpha_asr * asr_loss).item()

            # Prosody loss (auxiliary task, weighted by alpha)
            if prosody_labels is not None and self.alpha_prosody > 0:
                prosody_labels = prosody_labels.to(self.device)

                # Ensure prosody labels match sequence length
                if prosody_labels.size(1) != prosody_logits.size(1):
                    # Simple interpolation to match sizes
                    prosody_labels = F.interpolate(
                        prosody_labels.unsqueeze(1).float(),
                        size=prosody_logits.size(1),
                        mode='nearest'
                    ).squeeze(1).long()

                prosody_loss = self._compute_prosody_loss(
                    prosody_logits, prosody_labels)

                if loss is None:
                    loss = self.alpha_prosody * prosody_loss
                else:
                    loss = loss + self.alpha_prosody * prosody_loss

                loss_details['prosody_loss'] = prosody_loss.item()
                loss_details['prosody_loss_weighted'] = (
                    self.alpha_prosody * prosody_loss).item()

            if loss is not None:
                loss_details['total_loss'] = loss.item()

        if not return_dict:
            output = (ser_logits, asr_logits, prosody_logits)
            return ((loss,) + output) if loss is not None else output

        return MTLOutput(
            loss=loss,
            ser_logits=ser_logits,
            asr_logits=asr_logits,
            prosody_logits=prosody_logits,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
            loss_details=loss_details,
        )

    def get_backbone_hidden_size(self) -> int:
        """Get the hidden size of the backbone model"""
        return self.hidden_size

    def num_parameters(self, only_trainable: bool = True, exclude_embeddings: bool = False) -> int:
        """Get number of parameters"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
