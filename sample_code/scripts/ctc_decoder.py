import torch 
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class CTCDecoder:
    """
    CTC Decoder with multiple decoding strategies.
    Supports both greedy and beam search decoding. 
    """

    def __init__(self, blank_id: int=0, beam_width: int=10):
        self.blank_id = blank_id
        self.beam_width = beam_width

    def greedy_decode(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Greedy CTC decoding - fast and simple.
        
        Args:
            logits: (batch, time, vocab) tensor of logits
            lengths: (batch,) tensor of actual sequence lengths
        
        Returns:
            List of decoded token sequences
        """
        batch_size, max_time, _ = logits.shape
        # get most likely tokens
        predictions = torch.argmax(logits, dim=-1) # (batch, time)

        decoded_sequences = []
        for b in range(batch_size):
            # get actual length for this sequence
            seq_len = lengths[b] if lengths is not None else max_time
            pred_seq = predictions[b, :seq_len].cpu().tolist()

            # remove blank and merge repeated tokens
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
        Beam search CTC decoding - more accurate but slower.
        
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
        single_logits = logits[b, :seq_len].cpu() # (time, vocab)

        # apply log_softmax for stability
        log_probs = F.log_softmax(single_logits, dim=-1).numpy()

        # Beam search
        decoded = self._beam_search_single(log_probs)
        decoded_sequences.append(decoded)
           
    def _beam_search_single(self, log_probs: np.ndarray) -> List[int]:
        """
        Beam search for a single sequence.
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
                        # Blank extends the sequence without adding token
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

            # Merge identical sequences
            merged_beam = {}
            for seq, score in beam:
                seq_tuple = tuple(seq)
                if seq_tuple in merged_beam:
                    # Use log-sum-exp for numerical stability
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
    

class CTCLossWithRegularization(torch.nn.Module):
    """
    CTC Loss with additional regularization to prevent collapse.
    """

    def __init__(self, blank_id: int = 0, zero_infinity: bool = True,
                 entropy_weight: float = 0.01, blank_weight: float = 0.95):
        super().__init__()
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, zero_infinity=zero_infinity, reduction='mean')
        self.entropy_weight = entropy_weight
        self.blank_weight = blank_weight
        self.blank_id = blank_id

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss with entropy regularization.
        
        Args:
            log_probs: (T, N, C) tensor
            targets: flattened target sequences
            input_lengths: (N,) tensor
            target_lengths: (N,) tensor
        
        Returns:
            Scalar loss tensor
        """
        # Standard CTC loss
        ctc_loss = self.ctc_loss(
            log_probs, targets, input_lengths, target_lengths)

        # Entropy regularization to prevent collapse
        # Encourages the model to be less confident
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        # Negative because we want to maximize entropy
        entropy_loss = -torch.mean(entropy)

        # Blank regularization - penalize excessive blank predictions
        blank_probs = probs[:, :, self.blank_id]
        blank_loss = torch.mean(blank_probs) - self.blank_weight
        # Only penalize if blank prob > threshold
        blank_penalty = torch.relu(blank_loss) * 10.0

        # Total loss
        total_loss = ctc_loss + self.entropy_weight * entropy_loss + blank_penalty

        return total_loss, {
            'ctc_loss': ctc_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'blank_penalty': blank_penalty.item()
        }
