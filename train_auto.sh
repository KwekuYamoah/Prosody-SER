#!/bin/bash

# Get CPU information
echo "Detecting CPU configuration..."

# Try to get physical CPU cores
if command -v lscpu &> /dev/null; then
    # Get physical cores from lscpu
    PHYSICAL_CORES=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
    SOCKETS=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
    TOTAL_PHYSICAL_CORES=$((PHYSICAL_CORES * SOCKETS))
    echo "Physical cores detected: $TOTAL_PHYSICAL_CORES"
else
    # Fallback to nproc
    TOTAL_PHYSICAL_CORES=$(nproc)
    echo "Using logical cores as fallback: $TOTAL_PHYSICAL_CORES"
fi

# Number of GPUs to use
NUM_GPUS=4

# Calculate optimal OMP_NUM_THREADS
OMP_THREADS=$((TOTAL_PHYSICAL_CORES / NUM_GPUS))
# Ensure at least 1 thread per process
if [ $OMP_THREADS -lt 1 ]; then
    OMP_THREADS=1
fi

echo "Setting OMP_NUM_THREADS=$OMP_THREADS for $NUM_GPUS processes"

# Export environment variables
export OMP_NUM_THREADS=$OMP_THREADS
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your available GPUs

# For better NCCL performance
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Training parameters
TRAIN_FILE="json_data/extracted_ser_audio_features_wav_train.jsonl"
VAL_FILE="json_data/extracted_ser_audio_features_wav_val.jsonl"
TEST_FILE="json_data/extracted_ser_audio_features_wav_test.jsonl"
TRAINING_DATE="09_07"
CHECKPOINT_ID="1"

# Check if extracted files exist, if not, run preprocessing
MISSING=0
if [ ! -f "$TRAIN_FILE" ]; then
    echo "$TRAIN_FILE not found. Running preprocessing..."
    python preprocess_audio.py \
        --input_jsonl "json_data/ser_audio_features_wav_train.jsonl" \
        --output_jsonl "$TRAIN_FILE" \
        --audio_base_path "../github_repo/data/" \
        --vocab_json_path "new_vocab.json"
    MISSING=1
fi
if [ ! -f "$VAL_FILE" ]; then
    echo "$VAL_FILE not found. Running preprocessing..."
    python preprocess_audio.py \
        --input_jsonl "json_data/ser_audio_features_wav_val.jsonl" \
        --output_jsonl "$VAL_FILE" \
        --audio_base_path "../github_repo/data/" \
        --vocab_json_path "new_vocab.json"
    MISSING=1
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "$TEST_FILE not found. Running preprocessing..."
    python preprocess_audio.py \
        --input_jsonl "json_data/ser_audio_features_wav_test.jsonl" \
        --output_jsonl "$TEST_FILE" \
        --audio_base_path "audio" \
        --vocab_json_path "new_vocab.json"
    MISSING=1
fi

if [ $MISSING -eq 0 ]; then
    echo "All extracted files found. Proceeding to training."
else
    echo "Preprocessing completed for missing files. Proceeding to training."
fi

echo "Starting distributed training on $NUM_GPUS GPUs..."
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Run with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    xlsr_train_ddp.py \
    $TRAIN_FILE \
    $VAL_FILE \
    $TEST_FILE \
    $TRAINING_DATE \
    $CHECKPOINT_ID