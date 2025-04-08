import json 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import adaptive_avg_pool1d

import soundfile as sf
import numpy as np
import random
from pydub import AudioSegment
from io import BytesIO

# ESPnet imports
from espnet2.tasks.ssl import SSLTask

# set global device
device = "cuda" if torch.cuda.is_available() else "cpu"


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

        base_path = "../audio/"

        # Load audio (WAV) -> raw audio features
        samples, sr = sf.read(base_path + audio_path)

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
            B, T, D = xeus_outs.shape
            # Use provided target length or default to 2
            target_len = prosody_target_len if prosody_target_len is not None else 2
            
            # Safety check for adaptive pooling
            if T < 1:
                # Handle case where time dimension is too small
                pooled_features = xeus_outs.repeat(1, max(1, target_len), 1)
            else:
                try:
                    # Create adaptive pooling layer with correct output size
                    adaptive_pool = nn.AdaptiveAvgPool1d(target_len).to(device)
                    # Apply pooling with extra safety checks
                    pooled_features = adaptive_pool(xeus_outs.transpose(1, 2))
                    pooled_features = pooled_features.transpose(1, 2)
                except Exception as e:
                    print(f"Warning: Adaptive pooling failed: {e}")
                    # Fallback to simple repeat/slice for emergency handling
                    if T < target_len:
                        pooled_features = xeus_outs.repeat(1, max(1, target_len // T + 1), 1)[:, :target_len, :]
                    else:
                        indices = torch.linspace(0, T-1, target_len).long()
                        pooled_features = xeus_outs[:, indices, :]
            
            # Ensure proper dimensions for LSTM
            if pooled_features.size(0) == 0 or pooled_features.size(1) == 0:
                # Create dummy tensor in case of dimension issues
                pooled_features = torch.zeros(max(1, B), max(1, target_len), D).to(device)
            
            prosody_bilstm_out, _ = self.prosody_bilstm(pooled_features)
            prosody_out = self.prosody_fc(prosody_bilstm_out)
            prosody_logits = prosody_out.squeeze(-1)

        return asr_logits, ser_logits, prosody_logits
    

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
        if asr_logits is not None and asr_input_lengths is not None and asr_targets is not None:
            try:
                # For CTC, input sequence length must be >= target sequence length
                # Add additional check for batch size 1
                if asr_input_lengths.dim() == 0:
                    asr_input_lengths = asr_input_lengths.unsqueeze(0)
                if asr_target_lengths.dim() == 0:
                    asr_target_lengths = asr_target_lengths.unsqueeze(0)
                
                # Ensure CTC requirements are met (input_len >= target_len, input_len > 0)
                valid_ctc = True
                for b in range(asr_input_lengths.size(0)):
                    if asr_input_lengths[b] < asr_target_lengths[b]:
                        asr_input_lengths[b] = asr_target_lengths[b]
                        valid_ctc = False
                    if asr_input_lengths[b] <= 0:
                        asr_input_lengths[b] = 1
                        valid_ctc = False
                    if asr_target_lengths[b] <= 0:
                        asr_target_lengths[b] = 1
                        valid_ctc = False
                
                if valid_ctc:
                    loss_ctc = self.ctc_loss_fn(
                        asr_logits,
                        asr_targets,
                        asr_input_lengths,
                        asr_target_lengths
                    ) * self.alpha_ctc
            except Exception as e:
                print(f"Warning: CTC loss calculation failed: {e}")

        # SER: CrossEntropy
        if ser_logits is not None and ser_targets is not None:
            try:
                if ser_logits.size(0) > 0 and ser_targets.size(0) > 0:
                    # Check if we have a batch dimension
                    if ser_logits.dim() == 1:
                        ser_logits = ser_logits.unsqueeze(0)
                    if ser_targets.dim() == 0:
                        ser_targets = ser_targets.unsqueeze(0)
                    
                    # Verify classes are in range
                    if ser_targets.max() < ser_logits.size(1):
                        loss_ser = self.ser_loss_fn(
                            ser_logits,
                            ser_targets
                        ) * self.alpha_ser
            except Exception as e:
                print(f"Warning: SER loss calculation failed: {e}")

        # Prosody: BCE
        if prosody_logits is not None and prosody_targets is not None:
            try:
                # mask out padded values
                mask = (prosody_targets != -1)
                if mask.sum() > 0:  # Only proceed if we have valid targets
                    valid_logits = prosody_logits[mask]
                    valid_targets = prosody_targets[mask]
                    
                    if valid_logits.size(0) > 0 and valid_targets.size(0) > 0:
                        loss_prosody = self.prosody_loss_fn(
                            valid_logits,
                            valid_targets
                        ) * self.alpha_prosody
            except Exception as e:
                print(f"Warning: Prosody loss calculation failed: {e}")

        # Total loss
        total_loss = loss_ctc + loss_ser + loss_prosody
        return total_loss, loss_ctc, loss_ser, loss_prosody
            

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
                prosody_target_len=prosody_labels.size(1)  # Use actual prosody sequence length
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
                        prosody_target_len=prosody_labels.size(1)
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
    json_path = "../data/ser_audio_features_wav.json"
    xeus_checkpoint_path = "../XEUS/model/xeus_checkpoint.pth"

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
