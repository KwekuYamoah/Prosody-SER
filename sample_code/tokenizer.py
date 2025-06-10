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
        self.blank_id = 2  # CTC blank token
        self.bos_id = 3    # Beginning of sequence
        self.eos_id = 4    # End of Sequence

    def train_tokenizer(self, text_data_path: str, model_prefix: str = 'mtl_tokenizer'):
        """Train SentencePiece model on available text data"""
        spm.SentencePieceTrainer.train(
            input=text_data_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            # adding blank token for CTC
            user_defined_symbols=['<blank>'],
            character_coverage=0.995,
            normalization_rule_name='identity'  # don't normalize
        )

        self.model_path = f"{model_prefix}.model"
        self.load_tokenizer()

    def load_tokenizer(self):
        """Loading existing SentencePiece model"""
        if self.model_path and Path(self.model_path).exists():
            self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
            # override blank_id if it exists in vocab
            if '<blank>' in self.get_vocab():
                self.blank_id = self.sp.piece_to_id('<blank>')
        else:
            raise FileNotFoundError(f"SentencePiece model not found at {self.model_path}")

    def encode(self, word_str: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode list of words to token IDs
        Args:
            words_str: sentence for ASR
            add_special_tokens: Whether to add BOS/EOS tokens
        Returns:
            List of token IDs
        """
        if not self.sp:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")

        # encode
        
        token_ids = self.sp.encode_as_ids(word_str)

        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.sp.decode_ids(token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.sp) if self.sp else self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict"""
        if not self.sp:
            return {}
        return {self.sp.id_to_piece(i): i for i in range(len(self.sp))} 