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
from models import XeusMTL, MultiTaskLoss, Wav2Vec2MTL

# set global device
device = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Using device: {device}")

def build_word_dict(json_path):
    """
    Go through the entire JSON dataset, gather all distinct words, 
    and build a word->ID mapping. A few reserved tokens added:
      <blank> -> 0 (CTC blank)
      <unk>   -> 1 (unknown word)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = set()
    for _, entry in data.items():
        words = entry["words"]
        for w in words:
            w_stripped = w.strip()
            if w_stripped:
                vocab.add(w_stripped)

    # Create a simple word-level dictionary
    word2id = {
        "<blank>": 0,   # for CTC blank
        "<unk>":   1
    }
    current_index = 2
    for w in sorted(vocab):
        if w not in word2id:
            word2id[w] = current_index
            current_index += 1
    
    return word2id


# set global seed to ground experiments
def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 1. Dataset
class AkanSERDataset(Dataset):
    """
    Expects a JSON file with entries like:
    {
      "audio/path.mp3": {
          "words": [...],
          "prosody_annotations": [...],
          "emotion": 0,
          "split": "train"
      },
      ...
    }
    """
    def __init__(self, json_path, word2id, split="train", sample_rate = 16000):
        super().__init__()
        self.word2id = word2id
        # read json file
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.items = []
        for audio_path, meta in self.data.items():
            if meta["split"] == split:
                self.items.append((audio_path, meta))
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, meta = self.items[idx]

        base_path = "audio/"

        # Load audio (WAV) -> raw audio features
        samples, sr = sf.read(base_path + audio_path)
        if len(samples) < 2:
            # Raise an explicit error
            raise ValueError(f"Audio {audio_path} is too short (length {len(samples)}).")

        # words for ASR head + convert words -> token IDs
        words = meta["words"]
        token_ids = []
        for w in words:
            w = w.strip()
            if w in self.word2id:
                token_ids.append(self.word2id[w])
            else:
                token_ids.append(self.word2id["<unk>"])
        asr_target = torch.tensor(token_ids)

        # prosody annotations
        prosody_annotations = meta["prosody_annotations"]
        # emotion labels
        emotion_labels = meta["emotion"]

        # Return raw audio features plus labels
        return(
            torch.tensor(samples, dtype= torch.float),
            words,
            asr_target,
            torch.tensor(prosody_annotations, dtype=torch.float),
            emotion_labels
        )
        
def collate_fn(batch):
    """
    Collate a list of (raw audio features, words, prosody, emotion labels) into a batch.
    """

    audio_features = [item[0] for item in batch]
    words = [item[1] for item in batch]
    asr_target = [item[2] for item in batch]
    prosody_annotations = [item[3] for item in batch]
    emotion_labels = [item[4] for item in batch]

    # Pad the audio features to the maximum length in the batch
    lengths = [len(w) for w in audio_features]
    audio_features = pad_sequence(audio_features, batch_first=True).to(device)
    lengths = torch.LongTensor(lengths).to(device)

    # Pad Asr targets
    asr_lengths = [len(t) for t in asr_target]
    asr_target = pad_sequence(asr_target, batch_first=True, padding_value=0).to(device) # 0 is the <unk> token
    asr_lengths = torch.LongTensor(asr_lengths).to(device)

    # Get prosody sequence length (number of words)
    prosody_lengths = [len(p) for p in prosody_annotations]
    max_prosody_len = max(prosody_lengths)
    prosody_lengths = torch.LongTensor(prosody_lengths).to(device)

    # Prosody is of variable length since the utterances have different word counts
    # We'll also pad prosody labels 
    prosody_labels = pad_sequence(prosody_annotations, batch_first=True, padding_value=-1).to(device)

    # Emotion labels are already integers, so we can convert them to a tensor
    emotion_labels = torch.LongTensor(emotion_labels).to(device)

    # Return the batch
    return audio_features, lengths, asr_target, asr_lengths, words, prosody_labels, prosody_lengths, emotion_labels


# 4. Training Loop
def train_model(json_path, xeus_checkpoint_path, epochs=2, batch_size=2,
                alpha_ctc=1.0, alpha_ser=1.0, alpha_prosody=1.0, lr=1e-4):
    """
    1. Build a word dictionary from the entire dataset.
    2. Create train/val DataLoaders that produce real asr_targets from 'words'.
    3. Forward pass -> compute multi-task loss -> backprop -> optimize.
    4. Save model at the end.
    """
    word2id = build_word_dict(json_path)
    print(f"Dictionary size: {len(word2id)} tokens")
    # create datasets
    train_dataset = AkanSERDataset(json_path, word2id=word2id, split="train")
    val_dataset = AkanSERDataset(json_path, word2id=word2id, split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Build Model
    model = XeusMTL(
        xeus_checkpoint_path = xeus_checkpoint_path,
        asr_vocab_size = 100,
        num_emotions = 9,
        hidden_dim = 1024,
        prosody_hidden_dim = 256,
        use_asr = True,
        use_prosody = True,
        use_ser = True,
    ).to(device)

    # Build Loss
    loss_fn = MultiTaskLoss(
        alpha_ctc=alpha_ctc,
        alpha_ser=alpha_ser,
        alpha_prosody=alpha_prosody
    ).to(device)

    # Set Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss_accum = 0.0

        for batch_data in train_loader:
            audio_features, lengths, asr_target, asr_lengths, words, prosody_labels, prosody_lengths, emotion_labels = batch_data

            # Move data to device
            audio_features = audio_features.to(device)
            lengths = lengths.to(device)
            asr_target = asr_target.to(device)
            asr_lengths = asr_lengths.to(device)
            words = words
            prosody_labels = prosody_labels.to(device)
            prosody_lengths = prosody_lengths.to(device)
            emotion_labels = emotion_labels.to(device)

            # Forward pass
            asr_logits, ser_logits, prosody_logits = model(
                audio_features, 
                lengths, 
                use_mask=True,
                prosody_target_len=prosody_labels.size(1) # use length of prosody labels
            )

            # For CTC, the "input lengths" (time dimension) = asr_logits.size(0), 
            # but that's in the first dimension of asr_logits => shape: (time, batch, vocab_size).
            # We'll replicate that for each item in the batch:
            if asr_logits is not None:
                time_dim = asr_logits.size(0)
                asr_input_lengths = torch.full((audio_features.size(0),), time_dim, dtype=torch.long, device=device)
            else:
                asr_input_lengths = None

            # Compute combined loss
            total_loss, loss_ctc, loss_ser, loss_prosody = loss_fn(
                asr_logits,
                asr_target,
                asr_input_lengths,
                asr_lengths,
                ser_logits,
                emotion_labels,
                prosody_logits,
                prosody_labels
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_accum += total_loss.item()
        
        avg_loss = total_loss_accum / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
    
        # Validation step
        if len(val_dataset) > 0:
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for batch_data in val_loader:
                    audio_features, lengths, asr_target, asr_lengths, words, prosody_labels, prosody_lengths, emotion_labels = batch_data
                    audio_features = audio_features.to(device)
                    lengths = lengths.to(device)
                    asr_target = asr_target.to(device)
                    asr_lengths = asr_lengths.to(device)
                    prosody_labels = prosody_labels.to(device)
                    emotion_labels = emotion_labels.to(device)
                    
                    # Forward pass in validation
                    asr_logits, ser_logits, prosody_logits = model(
                        audio_features, 
                        lengths, 
                        use_mask=False,
                        prosody_target_len=prosody_labels.size(1) # use length of prosody labels
                    )
                    
                    if asr_logits is not None:
                        time_dim = asr_logits.size(0)
                        asr_input_lengths = torch.full((audio_features.size(0),), time_dim, dtype=torch.long, device=device)
                    else:
                        asr_input_lengths = None
                    
                    
                    total_loss, _, _, _ = loss_fn(
                        asr_logits,
                        asr_target,
                        asr_input_lengths,
                        asr_lengths,
                        ser_logits,
                        emotion_labels,
                        prosody_logits,
                        prosody_labels
                    )
                    val_loss_accum += total_loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)
            print(f"         Validation Loss: {avg_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "xeus_multitask_model.pt")
    print("Training complete, model saved to multitask_model.pt")


# 5. Inference with SER head only
def predict_emotions(model, audio_files):
    """
    Demo inference function that uses only the SER head.
    Args:
       model: a trained MultiTaskModel (with SER head)
       audio_files: list of paths to audio segments
    Returns:
       predicted_emotions: list of integers (class IDs) or distribution
    """
    model.eval()

    # Temporarily turn off other heads if they exist
    old_use_asr = model.use_asr
    old_use_prosody = model.use_prosody
    model.use_asr = False
    model.use_prosody = False

    predicted_emotions = []

    with torch.no_grad():
        for path in audio_files:
            audio = AudioSegment.from_file(path, format="mp3")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels > 1:
                samples = samples[::audio.channels]
            
            waveforms = torch.tensor(samples, dtype=torch.float, device=device).unsqueeze(0)
            lengths = torch.LongTensor([waveforms.shape[1]]).to(device)
            
            # forward pass (SER only)
            _, ser_logits, _ = model(waveforms, lengths, use_mask=False)
            # get predicted emotion
            pred_emotion = torch.argmax(ser_logits, dim=-1).item()
            predicted_emotions.append(pred_emotion)
    
    # Restore heads
    model.use_asr = old_use_asr
    model.use_prosody = old_use_prosody

    return predicted_emotions


if __name__ == "__main__":
    # Suppose you have a local JSON with data
    json_path = "data/ser_audio_features_wav.json"
    xeus_checkpoint_path = "XEUS/model/xeus_checkpoint.pth"

    # set seed
    seed = 42
    set_seed(seed)  # Set seed for reproducibility

    # Train
    train_model(
        json_path=json_path,
        xeus_checkpoint_path=xeus_checkpoint_path,
        epochs=1,               # for demonstration
        batch_size=4,
        alpha_ctc=1.0,
        alpha_ser=1.0,
        alpha_prosody=1.0,
        lr=1e-4
    )
    
    # Load the model for inference
    inference_model = XeusMTL(
        xeus_checkpoint_path=xeus_checkpoint_path,
        asr_vocab_size=100,
        num_emotions=9,
        hidden_dim=1024,
        prosody_hidden_dim=256,
        use_asr=True,
        use_ser=True,
        use_prosody=True
    ).to(device)

    inference_model.load_state_dict(torch.load("multitask_model.pt", map_location=device))
    
    # Now run SER-only inference
    sample_audio_files = [
        "movie_audio_segments_mp3/kae/kae_54990_58500.mp3",
        "movie_audio_segments_mp3/kae/kae_57990_65500.mp3"
    ]
    emotions = predict_emotions(inference_model, sample_audio_files)
    print("Predicted emotions:", emotions)
