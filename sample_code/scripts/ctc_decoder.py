import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class CTCDecoder:
    """
    Enhanced CTC Decoder with multiple decoding strategies.
    Supports both greedy and beam search decoding with proper blank handling.
    """

    def __init__(self, blank_id: int = 0, beam_width: int = 10):
        self.blank_id = blank_id
        self.beam_width = beam_width

    def greedy_decode(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Greedy CTC decoding with improved blank handling.

        Args:
            logits: (batch, time, vocab) tensor of logits
            lengths: (batch,) tensor of actual sequence lengths

        Returns:
            List of decoded token sequences
        """
        batch_size, max_time, _ = logits.shape
        # Get most likely tokens
        predictions = torch.argmax(logits, dim=-1)  # (batch, time)

        decoded_sequences = []
        for b in range(batch_size):
            # Get actual length for this sequence
            seq_len = lengths[b] if lengths is not None else max_time
            pred_seq = predictions[b, :seq_len].cpu().tolist()

            # Remove blank and merge repeated tokens (standard CTC)
            decoded = []
            prev_token = self.blank_id
            for token in pred_seq:
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            decoded_sequences.append(decoded)

        return decoded_sequences

    def beam_search_decode(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Beam search CTC decoding with proper probability handling.

        Args:
            logits: (batch, time, vocab) tensor of logits
            lengths: (batch,) tensor of actual sequence lengths

        Returns:
            List of decoded token sequences
        """
        batch_size = logits.shape[0]
        decoded_sequences = []

        for b in range(batch_size):
            seq_len = lengths[b] if lengths is not None else logits.shape[1]
            single_logits = logits[b, :seq_len].cpu()  # (time, vocab)

            # Apply log_softmax for numerical stability
            log_probs = F.log_softmax(single_logits, dim=-1).numpy()

            # Beam search for single sequence
            decoded = self._beam_search_single(log_probs)
            decoded_sequences.append(decoded)

        return decoded_sequences

    def _beam_search_single(self, log_probs: np.ndarray) -> List[int]:
        """
        Beam search for a single sequence with proper CTC constraints.
        """
        T, V = log_probs.shape

        # Initialize beam with empty sequence
        beam = [([], 0.0)]  # (sequence, score)

        for t in range(T):
            new_beam = []

            for seq, score in beam:
                for v in range(V):
                    new_score = score + log_probs[t, v]

                    if v == self.blank_id:
                        # Blank extends sequence without adding token
                        new_beam.append((seq, new_score))
                    else:
                        # Non-blank token
                        if len(seq) == 0 or seq[-1] != v:
                            # Add new token
                            new_beam.append((seq + [v], new_score))
                        else:
                            # Repeated token - just update score
                            new_beam.append((seq, new_score))

            # Keep top beam_width candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]

            # Merge identical sequences using log-sum-exp
            merged_beam = {}
            for seq, score in beam:
                seq_tuple = tuple(seq)
                if seq_tuple in merged_beam:
                    merged_beam[seq_tuple] = np.logaddexp(
                        merged_beam[seq_tuple], score)
                else:
                    merged_beam[seq_tuple] = score

            beam = [(list(seq), score) for seq, score in merged_beam.items()]
            beam.sort(key=lambda x: x[1], reverse=True)
            beam = beam[:self.beam_width]

        # Return best sequence
        return beam[0][0] if beam else []

    def decode_batch(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                     method: str = "greedy") -> List[List[int]]:
        """
        Decode a batch of logits using specified method.

        Args:
            logits: (batch, time, vocab) tensor
            lengths: (batch,) tensor of sequence lengths
            method: "greedy" or "beam"

        Returns:
            List of decoded sequences
        """
        if method == "greedy":
            return self.greedy_decode(logits, lengths)
        elif method == "beam":
            return self.beam_search_decode(logits, lengths)
        else:
            raise ValueError(f"Unknown decoding method: {method}")


class CTCLoss(torch.nn.Module):
    """
    CTC Loss with regularization to prevent blank token collapse.

    Implements multiple strategies to address the blank prediction issue:
    1. Entropy regularization (from paper)
    2. Blank penalty (direct approach)
    3. Label smoothing (optional)
    4. Confidence penalty
    """

    def __init__(self, blank_id: int = 0, zero_infinity: bool = True,
                 entropy_weight: float = 0.01, blank_penalty: float = 0.1,
                 label_smoothing: float = 0.0, confidence_penalty: float = 0.0,
                 blank_threshold: float = 0.3):  # Much more reasonable threshold
        super().__init__()
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, zero_infinity=zero_infinity, reduction='mean')
        self.blank_id = blank_id

        # Regularization weights
        self.entropy_weight = entropy_weight
        self.blank_penalty = blank_penalty
        self.label_smoothing = label_smoothing
        self.confidence_penalty = confidence_penalty
        self.blank_threshold = blank_threshold  # Store threshold as parameter

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute enhanced CTC loss with regularization.

        Args:
            log_probs: (T, N, C) tensor of log probabilities
            targets: flattened target sequences
            input_lengths: (N,) tensor of input lengths
            target_lengths: (N,) tensor of target lengths

        Returns:
            Total loss and detailed loss components
        """
        # Standard CTC loss
        ctc_loss = self.ctc_loss(
            log_probs, targets, input_lengths, target_lengths)

        # Convert log_probs to probabilities for regularization
        probs = torch.exp(log_probs)  # (T, N, C)

        # 1. Entropy regularization (encourages less confident predictions)
        entropy_loss = torch.tensor(0.0, device=log_probs.device)
        if self.entropy_weight > 0:
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (T, N)
            # Mask by actual lengths to avoid padding positions
            length_mask = torch.arange(log_probs.size(
                0), device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
            masked_entropy = entropy * length_mask.float()
            # Negative because we want to maximize entropy
            entropy_loss = -torch.mean(masked_entropy)

        # 2. Blank penalty (directly penalize excessive blank predictions)
        blank_penalty_loss = torch.tensor(0.0, device=log_probs.device)
        if self.blank_penalty > 0:
            blank_probs = probs[:, :, self.blank_id]  # (T, N)
            
            # Mask by actual lengths
            length_mask = torch.arange(log_probs.size(
                0), device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
            masked_blank_probs = blank_probs * length_mask.float()

            
            # exponentially increasing penalty for excessive blanks
            masked_blank_probs = masked_blank_probs.exp()

            # Much more aggressive blank penalty strategy
            # Penalize if average blank probability exceeds reasonable threshold (e.g., 0.3)
            avg_blank_prob = masked_blank_probs.mean()

            # apply the penalty now
            blank_penalty_loss = avg_blank_prob * 100



        # 3. Label smoothing (optional - smooths the target distribution)
        label_smoothing_loss = torch.tensor(0.0, device=log_probs.device)
        if self.label_smoothing > 0:
            # Uniform distribution over vocabulary
            uniform_dist = torch.ones_like(probs) / probs.size(-1)
            # Smooth between true targets and uniform distribution
            kl_loss = F.kl_div(log_probs, uniform_dist, reduction='none')
            length_mask = torch.arange(log_probs.size(
                0), device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
            masked_kl = kl_loss.sum(-1) * length_mask.float()
            label_smoothing_loss = torch.mean(masked_kl)

        # 4. Confidence penalty (penalize overconfident predictions)
        confidence_penalty_loss = torch.tensor(0.0, device=log_probs.device)
        if self.confidence_penalty > 0:
            max_probs = torch.max(probs, dim=-1)[0]  # (T, N)
            length_mask = torch.arange(log_probs.size(
                0), device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
            masked_max_probs = max_probs * length_mask.float()
            # Penalize very high confidence (> 0.95)
            confidence_threshold = 0.95
            confidence_penalty_loss = torch.mean(
                torch.relu(masked_max_probs - confidence_threshold))

        # Total loss combination
        total_loss = (ctc_loss +
                      self.entropy_weight * entropy_loss +
                      self.blank_penalty * blank_penalty_loss +
                      self.label_smoothing * label_smoothing_loss +
                      self.confidence_penalty * confidence_penalty_loss)

        # Detailed loss components for monitoring
        loss_details = {
            'ctc_loss': ctc_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'blank_penalty': blank_penalty_loss.item(),
            'label_smoothing_loss': label_smoothing_loss.item(),
            'confidence_penalty': confidence_penalty_loss.item(),
            'total_regularization': (total_loss - ctc_loss).item(),
            # Debug information
            'avg_blank_prob': torch.mean(probs[:, :, self.blank_id]).item(),
            'blank_threshold': self.blank_threshold,
            'blank_penalty_weight': self.blank_penalty,
            'entropy_weight': self.entropy_weight
        }

        return total_loss, loss_details

    def get_blank_statistics(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> Dict:
        """Get statistics about blank predictions for monitoring"""
        probs = torch.exp(log_probs)
        blank_probs = probs[:, :, self.blank_id]

        # Mask by actual lengths
        length_mask = torch.arange(log_probs.size(
            0), device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
        masked_blank_probs = blank_probs * length_mask.float()

        return {
            'avg_blank_prob': torch.mean(masked_blank_probs).item(),
            'max_blank_prob': torch.max(masked_blank_probs).item(),
            'min_blank_prob': torch.min(masked_blank_probs[length_mask]).item(),
            'blank_dominance': (masked_blank_probs > self.blank_threshold).float().mean().item(),
            # Timesteps > 50% blank
            'high_blank_timesteps': (masked_blank_probs > 0.5).float().mean().item(),
            # Timesteps > 70% blank
            'excessive_blank_timesteps': (masked_blank_probs > 0.7).float().mean().item(),
            'blank_threshold': self.blank_threshold
        }
