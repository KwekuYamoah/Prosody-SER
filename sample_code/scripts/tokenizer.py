import sentencepiece as spm
from pathlib import Path
from typing import List, Dict, Optional


class SentencePieceTokenizer:
    """Wrapper class for SentencePiece tokenizer with MTL-specific functionality"""

    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 4000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None

        # special token IDs for CTC
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2    # Beginning of sequence
        self.eos_id = 3    # End of Sequence
        # CRITICAL FIX: blank_id should be set after loading the model
        # as it will be the last token in the vocabulary
        self.blank_id = None

    def train_tokenizer(self, text_data_path: str, model_prefix: str = 'mtl_tokenizer'):
        """Train SentencePiece model on available text data"""
        # CRITICAL FIX: Don't add <blank> as user_defined_symbol
        # CTC blank should be implicit (ID 0 or last ID)
        spm.SentencePieceTrainer.train(
            input=text_data_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            # Remove the user_defined_symbols for blank
            character_coverage=0.995,
            normalization_rule_name='identity'  # don't normalize
        )

        self.model_path = f"{model_prefix}.model"
        self.load_tokenizer()

    def load_tokenizer(self):
        """Loading existing SentencePiece model"""
        if self.model_path and Path(self.model_path).exists():
            self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
            # CRITICAL FIX: Set blank_id to 0 (standard CTC convention)
            # This ensures CTC loss uses the correct blank token
            self.blank_id = 0
            print(
                f"Tokenizer loaded. Vocab size: {self.get_vocab_size()}, Blank ID: {self.blank_id}")
        else:
            raise FileNotFoundError(
                f"SentencePiece model not found at {self.model_path}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs
        Args:
            text: sentence for ASR
            add_special_tokens: Whether to add BOS/EOS tokens
        Returns:
            List of token IDs
        """
        if not self.sp:
            raise RuntimeError(
                "Tokenizer not loaded. Call load_tokenizer() first.")

        # encode the text
        token_ids = self.sp.encode_as_ids(text)

        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        CRITICAL FIX: Filter out blank tokens (ID 0) before decoding
        """
        if not self.sp:
            raise RuntimeError("Tokenizer not loaded.")

        # Filter out blank tokens and padding
        if skip_special_tokens:
            # Remove blank (0), pad (0), bos, eos tokens
            filtered_ids = [id for id in token_ids if id not in [
                self.blank_id, self.pad_id, self.bos_id, self.eos_id]]
        else:
            filtered_ids = token_ids

        # Handle empty sequence
        if not filtered_ids:
            return ""

        try:
            return self.sp.decode_ids(filtered_ids)
        except Exception as e:
            print(f"Decoding error with IDs {filtered_ids}: {e}")
            return ""

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.sp) if self.sp else self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict"""
        if not self.sp:
            return {}
        return {self.sp.id_to_piece(i): i for i in range(len(self.sp))}