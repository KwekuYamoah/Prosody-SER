import torch.utils.data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from jiwer import wer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
import wandb
from xlsr_model_ddp import *
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2ForCTC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
import jsonlines
import json
import torch.nn as nn
import torch.optim as optim
import joblib
from functools import partial
import optuna
import sys
import torch
import os

# Set OMP_NUM_THREADS early before importing other libraries
# For Python 3.11, we need to use alternative methods to get CPU count
try:
    import psutil
    # Get physical CPU cores
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None:
        physical_cores = psutil.cpu_count(logical=True)
except ImportError:
    # Fallback if psutil is not installed
    physical_cores = os.cpu_count()
    if physical_cores is None:
        physical_cores = 1

torch.backends.cudnn.deterministic = True
# Get number of processes from environment if running with torchrun
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Set OMP_NUM_THREADS if not already set
if "OMP_NUM_THREADS" not in os.environ:
    omp_threads = max(1, physical_cores // world_size)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    print(
        f"Setting OMP_NUM_THREADS to {omp_threads} (physical_cores={physical_cores}, world_size={world_size})")

# Now import the rest of the libraries

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# distributed training imports

# Function to initialize DDP when using torchrun


def init_ddp():
    """Initialize distributed process group for torchrun"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Set device BEFORE initializing process group
        torch.cuda.set_device(local_rank)

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


# Initialize at module level
rank, local_rank, world_size = init_ddp()

# Rest of your imports and global variables...
MAX_PROSODY_LABELS_LEN = 0
MAX_ASR_LABELS_LEN = 0

# Your existing functions remain the same...


def cleanup():
    """Cleanup distributed training environment"""
    if dist.is_initialized():
        destroy_process_group()


def load_jsonl(file_path: str):
    """Load data from JSONL file"""
    data = {}
    data['audio'] = []
    data['prosody'] = []
    data['asr'] = []
    data['emotion'] = []

    with jsonlines.open(file_path) as reader:
        for line in reader:
            data['audio'].append(line['audio'])
            data['prosody'].append(line['prosody'])
            data['asr'].append(line['asr'])
            data['emotion'].append(line['emotion'])
    return data


# Load tokenizer from vocab.json
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file="new_vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]"
)

if rank == 0:
    print(tokenizer.get_vocab())

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# set global device
device = "cuda" if torch.cuda.is_available() else "cpu"


class SERDataset(Dataset):
    def __init__(self, data_json):
        self.audio_features = data_json['audio']
        self.prosody_features = data_json['prosody']
        self.asr_features = data_json['asr']
        self.emotion_labels = data_json['emotion']

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, index):
        if isinstance(self.asr_features[index], (list)):
            return (
                self.audio_features[index],
                self.prosody_features[index],
                self.asr_features[index],
                self.emotion_labels[index]
            )
        else:
            return (
                self.audio_features[index],
                self.prosody_features[index],
                [self.asr_features[index]],
                self.emotion_labels[index]
            )


def get_max_prosody_label_len(train_data, val_data, test_data):
    prosody_vectors = train_data['prosody'] + \
        val_data['prosody'] + test_data['prosody']
    prosody_labels_len = []
    for prosody_vector in prosody_vectors:
        prosody_labels_len.append(len(prosody_vector))
    return max(prosody_labels_len)


def get_max_asr_label_len(train_data, val_data, test_data):
    asr_vectors = train_data['asr'] + val_data['asr'] + test_data['asr']
    asr_labels_len = []
    for asr_vector in asr_vectors:
        if isinstance(asr_vector, (list)):
            asr_labels_len.append(len(asr_vector))
        else:
            print(
                f"Warning: Expected list but got {type(asr_vector)} - value: {asr_vector}")
            asr_labels_len.append(1)
    return max(asr_labels_len)


def collate_fn(batch, max_prosody_len):
    """
    batch: List of tuples (raw_waveform_tensor, prosody_labels, asr_labels, emotion_label)
    """
    raw_waveforms = [item[0] for item in batch]
    prosody_labels = [torch.tensor(item[1]) for item in batch]
    asr_labels = [torch.tensor(item[2]) for item in batch]
    emotion_labels = [torch.tensor(item[3]) for item in batch]

    inputs = processor(
        raw_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    audio_lengths = [len(audio_feature) for audio_feature in raw_waveforms]
    audio_lengths = torch.tensor(audio_lengths)

    asr_lens = [lbl.size(0) for lbl in asr_labels]
    asr_labels = pad_sequence(
        asr_labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    asr_lengths = torch.tensor(asr_lens)

    prosody_lens = [lbl.size(0) for lbl in prosody_labels]
    prosody_labels = pad_sequence(prosody_labels, batch_first=True)
    pad_diff = max_prosody_len - prosody_labels.size(1)
    if pad_diff > 0:
        prosody_labels = F.pad(prosody_labels, (0, pad_diff), value=0)
    prosody_lengths = torch.tensor(prosody_lens)

    emotion_labels = torch.tensor(emotion_labels)

    return (
        input_values,
        attention_mask,
        audio_lengths,
        asr_labels,
        asr_lengths,
        prosody_labels,
        prosody_lengths,
        emotion_labels
    )


def compute_wer(pred, true):
    pairs = [(p, t) for p, t in zip(pred, true) if t.strip() != ""]
    if pairs:
        pred_labels, true_labels = zip(*pairs)
        generated_wer = wer(list(true_labels), list(pred_labels))
    else:
        generated_wer = float("nan")
    return generated_wer


def train_ddp(train_data, val_data, test_data, training_date, checkpoint_id, lstm_hidden_size,
              batch_size=1, epochs=50, lr=1e-4, alpha_ctc=1.0, alpha_ser=1.0,
              alpha_prosody=1.0):
    '''
    Trains the specified model using the provided data points with DDP.
    Note: rank, local_rank, and world_size are already initialized at module level
    '''

    # Verify that distributed is initialized
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized. "
                           "Please run this script with torchrun.")

    # Get the global rank (already set at module level)
    global_rank = rank

    # Only initialize wandb on global rank 0
    if global_rank == 0:
        wandb.init(
            project="multitask-speech-model",
            name=f"experiment_{training_date}_{checkpoint_id}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "total_batch_size": batch_size * world_size,
                "learning_rate": lr,
                "alpha_ctc": alpha_ctc,
                "alpha_ser": alpha_ser,
                "alpha_prosody": alpha_prosody,
                "world_size": world_size
            }
        )

    # set the need globals
    global MAX_PROSODY_LABELS_LEN
    global MAX_ASR_LABELS_LEN

    # create the dataset
    train_dataset = SERDataset(train_data)
    val_dataset = SERDataset(val_data)

    # SYNCHRONIZED MAX LENGTH CALCULATION - FIX FOR RANK 0 ISSUE
    if global_rank == 0:
        max_prosody = get_max_prosody_label_len(
            train_data, val_data, test_data)
        max_asr = get_max_asr_label_len(train_data, val_data, test_data)
    else:
        max_prosody = 0
        max_asr = 0

    # Convert to tensors and broadcast from rank 0
    max_prosody_tensor = torch.tensor(max_prosody, device=local_rank)
    max_asr_tensor = torch.tensor(max_asr, device=local_rank)
    dist.broadcast(max_prosody_tensor, src=0)
    dist.broadcast(max_asr_tensor, src=0)

    MAX_PROSODY_LABELS_LEN = max_prosody_tensor.item()
    MAX_ASR_LABELS_LEN = max_asr_tensor.item()

    if global_rank == 0:
        print(f"MAX_PROSODY_LABELS_LEN set to: {MAX_PROSODY_LABELS_LEN}")
        print(f"MAX_ASR_LABELS_LEN set to: {MAX_ASR_LABELS_LEN}")

    # create the distributed samplers
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )

    # construct the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, MAX_PROSODY_LABELS_LEN),
        num_workers=4,  # Reduced for better stability
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, MAX_PROSODY_LABELS_LEN),
        num_workers=4,  # Reduced for better stability
        sampler=val_sampler,
        pin_memory=True,
    )

    wav2vec2_model_base_config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-xls-r-300m")
    base_dict = wav2vec2_model_base_config.to_dict()

    for k in ("pad_token_id", "vocab_size"):
        base_dict.pop(k, None)

    custom_cfg = Wav2Vec2Config(
        **base_dict,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        config=custom_cfg
    )
    # Disable gradient checkpointing for DDP compatibility
    # model.gradient_checkpointing_enable()

    # Initialize the Model
    xlsr_model = Wav2Vec2MTL(
        config=custom_cfg,
        tokenizer=tokenizer,
        pretrained_wav2vec_model=model,
        emotion_model_output_size=9,
        asr_model_output_size=len(tokenizer),
        prosodic_prominence_model_output_size=MAX_PROSODY_LABELS_LEN,
        prosody_model_lstm_hidden_size=lstm_hidden_size
    )

    # **ADD THIS: Validate model parameters before moving to device**
    param_count = sum(p.numel() for p in xlsr_model.parameters())
    # CRITICAL SAFETY CHECK - FIX FOR RANK 0 MODEL INIT FAILURE
    if param_count == 0:
        raise RuntimeError(
            f"Rank {global_rank}: Model initialization failed - 0 parameters detected")

    print(f"[Rank {global_rank}] Model parameter count: {param_count}")

    # Synchronize parameter counts across all ranks
    param_tensor = torch.tensor([param_count], dtype=torch.long)
    if torch.cuda.is_available():
        param_tensor = param_tensor.cuda(local_rank)

    # Gather parameter counts from all ranks
    param_counts = [torch.zeros_like(param_tensor) for _ in range(world_size)]
    dist.all_gather(param_counts, param_tensor)

    # Check if all ranks have the same parameter count
    param_counts_cpu = [p.cpu().item() for p in param_counts]
    if not all(count == param_counts_cpu[0] for count in param_counts_cpu):
        raise RuntimeError(
            f"Model parameter count mismatch across ranks: {param_counts_cpu}")

    if global_rank == 0:
        print(
            f"All ranks have consistent parameter count: {param_counts_cpu[0]}")

    # move the model to the specified local_rank's device
    xlsr_model = xlsr_model.to(local_rank)

    # **ADD THIS: Synchronize before DDP wrapping**
    dist.barrier()

    # Wrap the model with DDP
    xlsr_model = DDP(xlsr_model,
                     device_ids=[local_rank],
                     output_device=local_rank,
                     find_unused_parameters=True,  # Set to True to handle dynamic graph
                     gradient_as_bucket_view=True,   # Memory optimization
                     static_graph=False)  # Disable static graph for dynamic training graph

    # set the optimizer
    head_params = [p for n, p in xlsr_model.named_parameters()
                   if "asr_head" in n]
    encoder_params = [
        p for n, p in xlsr_model.named_parameters() if "asr_head" not in n]

    optimizer = AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": head_params, "lr": 5e-4},
    ])

    scaler = GradScaler(device='cuda')
    collated_val_loss_across_epochs = []

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        xlsr_model.train()

        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", disable=(global_rank != 0))
        for batch_data in progress_bar:
            (
                audio_features, attention_mask, audio_lengths,
                asr_labels, asr_lengths, prosody_labels,
                prosody_lengths, emotion_labels
            ) = batch_data

            # Use local_rank for device placement
            audio_features = audio_features.to(local_rank, non_blocking=True)
            attention_mask = attention_mask.to(local_rank, non_blocking=True)
            asr_labels = asr_labels.to(local_rank, non_blocking=True)
            prosody_labels = prosody_labels.to(local_rank, non_blocking=True)
            emotion_labels = emotion_labels.to(local_rank, non_blocking=True)

            try:
                with autocast(device_type='cuda', dtype=torch.float16):
                    model_outputs = xlsr_model(
                        audio_features=audio_features,
                        attention_mask=attention_mask,
                        asr_labels=asr_labels,
                        prosodic_prominence_annotation_labels=prosody_labels,
                        emotion_labels=emotion_labels,
                        mtl_head=[
                            'asr', 'prosodic_prominence_annotation', 'ser']
                    )

                    # All losses are always computed, alpha parameters control their contribution
                    computed_loss = (alpha_ctc*model_outputs['asr_loss']) + (
                        alpha_prosody*model_outputs['prosody_loss']) + (alpha_ser*model_outputs['ser_loss'])

                optimizer.zero_grad()
                scaler.scale(computed_loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    xlsr_model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += computed_loss.item()
                num_batches += 1

                if global_rank == 0:
                    progress_bar.set_postfix({'loss': computed_loss.item()})

                del computed_loss, model_outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if global_rank == 0:
                        print('OOM in training batch - skipping')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        # Synchronize and compute average loss
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_loss_tensor = torch.tensor([avg_loss]).to(local_rank)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / world_size

            if global_rank == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} -- Average Train Loss: {avg_loss:.4f}")
                wandb.log({
                    "avg_train_loss": avg_loss,
                    "epoch": epoch
                })

        # Validation phase
        xlsr_model.eval()
        val_loss = 0.0
        val_count = 0
        max_val_samples = 500

        with torch.no_grad():
            val_progress_bar = tqdm(
                val_loader, desc=f"Validation {epoch+1}/{epochs}", disable=(global_rank != 0))
            for batch_idx, batch_data in enumerate(val_progress_bar):
                if val_count >= max_val_samples:
                    break

                (
                    audio_features, attention_mask, audio_lengths,
                    asr_labels, asr_lengths, prosody_labels,
                    prosody_lengths, emotion_labels
                ) = batch_data

                audio_features = audio_features.to(
                    local_rank, non_blocking=True)
                attention_mask = attention_mask.to(
                    local_rank, non_blocking=True)
                asr_labels = asr_labels.to(local_rank, non_blocking=True)
                prosody_labels = prosody_labels.to(
                    local_rank, non_blocking=True)
                emotion_labels = emotion_labels.to(
                    local_rank, non_blocking=True)

                try:
                    model_outputs = xlsr_model(
                        audio_features,
                        attention_mask=attention_mask,
                        prosodic_prominence_annotation_labels=prosody_labels,
                        asr_labels=asr_labels,
                        emotion_labels=emotion_labels,
                        mtl_head=[
                            'asr', 'prosodic_prominence_annotation', 'ser']
                    )

                    if global_rank == 0 and val_count < 10:
                        asr_preds = torch.argmax(
                            model_outputs['asr_output'], dim=-1).cpu().tolist()
                        asr_targets = asr_labels.cpu().tolist()

                        for pred_ids, target_ids in zip(asr_preds, asr_targets):
                            pred_tokens = tokenizer.batch_decode(
                                [pred_ids], skip_special_tokens=False)[0]
                            target_tokens = tokenizer.batch_decode(
                                [target_ids], skip_special_tokens=False)[0]
                            print(f'ASR Pred: {pred_tokens}')
                            print(f'ASR True: {target_tokens}')

                    # All losses are always computed, alpha parameters control their contribution
                    computed_val_loss = (alpha_ctc*model_outputs['asr_loss']) + (
                        alpha_prosody*model_outputs['prosody_loss']) + (alpha_ser*model_outputs['ser_loss'])
                    val_loss += computed_val_loss.item()
                    val_count += 1

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if global_rank == 0:
                            print('OOM in validation batch - skipping')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

            # Compute average validation loss
            if val_count > 0:
                avg_val_loss = val_loss / val_count
                val_stats = torch.tensor(
                    [val_loss, float(val_count)]).to(local_rank)
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)

                total_val_loss = val_stats[0].item()
                total_val_count = int(val_stats[1].item())

                avg_val_loss = total_val_loss / total_val_count
                collated_val_loss_across_epochs.append(avg_val_loss)

                if global_rank == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} -- Average Validation Loss: {avg_val_loss:.4f}")
                    wandb.log({
                        "avg_val_loss": avg_val_loss,
                        "epoch": epoch
                    })

    # Save the model only on rank 0
    if global_rank == 0:
        os.makedirs("./model_checkpoints", exist_ok=True)
        model_checkpoint_path = f"./model_checkpoints/xlsr_multitask_model_{training_date}_{checkpoint_id}.pt"
        torch.save(xlsr_model.module.state_dict(), model_checkpoint_path)
        print("Training for the model is complete!")
    else:
        model_checkpoint_path = None

    if collated_val_loss_across_epochs:
        avg_collated_val_loss = sum(
            collated_val_loss_across_epochs) / len(collated_val_loss_across_epochs)
    else:
        avg_collated_val_loss = float('inf')

    return model_checkpoint_path, avg_collated_val_loss


def objective(trial, train_data, val_data, test_data, training_date, checkpoint_id):
    # hyperparamter suggestions for the multi-task wav2vec2 model
    suggested_lr = trial.suggest_float('lr', 1e-8, 1e-2, log=True)
    suggested_alpha_ctc = trial.suggest_float(
        'alpha_ctc', 0.1, 1.0, step=0.1, log=False)
    suggested_alpha_ser = trial.suggest_float(
        'alpha_ser', 0.1, 1.0, step=0.1, log=False)
    suggested_alpha_prosody = trial.suggest_float(
        'alpha_prosody', 0.1, 1.0, step=0.1, log=False)

    # hyperparameter suggestions for the LSTM hidden dimension size for the prosody arm
    suggested_lstm_hidden_size = trial.suggest_int(
        'prosody_model_lstm_hidden_size', 128, 512, step=64)

    # train the model to obtain the value to be optimised for
    model_checkpoint_path, avg_collated_val_loss = train(
        train_data,
        val_data,
        test_data,
        training_date=training_date,
        checkpoint_id=checkpoint_id,
        lstm_hidden_size=suggested_lstm_hidden_size,
        batch_size=1,
        epochs=5,
        lr=suggested_lr,
        alpha_ctc=suggested_alpha_ctc,
        alpha_ser=suggested_alpha_ser,
        alpha_prosody=suggested_alpha_prosody
    )
    return avg_collated_val_loss


def perform_hyperparameter_search(num_trials, train_data, val_data, test_data, training_date, checkpoint_id):
    '''
    Performs a hyperparameter search to find the best combination of hyperparameters to lead to the best model
    performance.
    '''
    wrapped_objective = partial(objective, train_data=train_data, val_data=val_data,
                                test_data=test_data, training_date=training_date, checkpoint_id=checkpoint_id)
    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=num_trials)

    try:
        best_found_hyperparameters = study.best_params
        lowest_computed_val_loss = study.best_value

        print('best found hyperparameters: ', best_found_hyperparameters)
        print('best computed val loss: ', lowest_computed_val_loss)

        # save the best found hyperparameter values
        with open('optimal_hyperparameters.json', 'w') as input_json:
            json.dump(study.best_params, input_json, indent=4)

        return best_found_hyperparameters

    except Exception as e:
        print('Encountered error: ', e)


if __name__ == '__main__':
    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    # Only print on rank 0 to avoid duplicate prints
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())
        print(
            f"Running with world_size={world_size}, rank={rank}, local_rank={local_rank}")

    torch.manual_seed(123)

    # Get command line arguments
    train_data_path = sys.argv[1]
    val_data_path = sys.argv[2]
    test_data_path = sys.argv[3]
    cur_date = sys.argv[4]
    cur_exp_run = int(sys.argv[5])

    # Load data
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    test_data = load_jsonl(test_data_path)

    # Check if running with distributed training
    if world_size > 1:
        if rank == 0:
            print(f"Running distributed training with {world_size} GPUs")
        checkpoint_pth, avg_val_loss = train_ddp(
            train_data, val_data, test_data, cur_date, cur_exp_run, 192,
            batch_size=4, epochs=10, lr=3e-4,
            alpha_ctc=1.0, alpha_ser=0.0, alpha_prosody=0.0
        )
    else:
        print("Running single GPU training")
        # You'll need to add your train() function here
        raise NotImplementedError(
            "Single GPU training function not included in this snippet")

    # Clean up distributed training
    if dist.is_initialized():
        dist.barrier()  # Wait for all processes
        cleanup()

    # Test the model (only on rank 0)
    if rank == 0 and checkpoint_pth:
        print("Starting model evaluation...")
        # MAX_PROSODY_LABELS_LEN and MAX_ASR_LABELS_LEN should already be set
        # test(checkpoint_pth, test_data, 1)
