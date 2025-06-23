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
        self.pad_id = 3
        
        # as it will be the last token in the vocabulary
        self.blank_id = None

    def train_tokenizer(self, text_data_path: str, model_type: str = 'bpe', model_prefix: str = 'akan_mtl_tokenizer'):
        """Train SentencePiece model on available text data"""
        # FIX: Add explicit user_defined_symbols for CTC compatibility
        # This ensures special tokens are properly reserved in the vocabulary
        spm.SentencePieceTrainer.train(
            input=text_data_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=model_type,
            pad_id=self.pad_id,
            user_defined_symbols=['<pad>', '<blank>'],
            character_coverage=0.995,
            normalization_rule_name='identity'  # don't normalize
        )

        self.model_path = f"{model_prefix}.model"
        self.load_tokenizer()

    def load_tokenizer(self):
        """Loading existing SentencePiece model"""
        if self.model_path and Path(self.model_path).exists():
            self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

            # FIX: Properly set blank_id and verify special token mappings
            vocab = self.get_vocab()

            # Check if <blank> token exists in vocabulary
            if '<blank>' in vocab:
                self.blank_id = vocab['<blank>']
                print(f"✓ Found <blank> token with ID: {self.blank_id}")
            else:
                # Fallback: use the last token ID as blank (CTC convention)
                self.blank_id = len(vocab) - 1
                print(
                    f"⚠️ <blank> token not found, using last ID as blank: {self.blank_id}")

            # Verify special token mappings
            print(f"Special token mappings:")
            print(f"  <pad>: {vocab.get('<pad>', 'NOT FOUND')}")
            print(f"  <unk>: {vocab.get('<unk>', 'NOT FOUND')}")
            print(f"  <s>: {vocab.get('<s>', 'NOT FOUND')}")
            print(f"  </s>: {vocab.get('</s>', 'NOT FOUND')}")
            print(f"  <blank>: {vocab.get('<blank>', 'NOT FOUND')}")

            print(
                f"Tokenizer loaded. Vocab size: {self.get_vocab_size()}, Blank ID: {self.blank_id}")
        else:
            raise FileNotFoundError(
                f"SentencePiece model not found at {self.model_path}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        Args:
            text: sentence for ASR
        Returns:
            List of token IDs
        """
        if not self.sp:
            raise RuntimeError(
                "Tokenizer not loaded. Call load_tokenizer() first.")

        # encode the text
        token_ids = self.sp.encode_as_ids(text)

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
                self.blank_id, self.pad_id]]
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
