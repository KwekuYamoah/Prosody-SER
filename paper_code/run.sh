#!/bin/bash

# Set environment variables for offline mode and disable wandb
# export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Model and training hyperparameters
export MODEL=wav2vec2-xls-r-300m
export TOKENIZER=wav2vec2-xls-r-300m
export ALPHA=0.1    # Weight for ASR loss (auxiliary task)
export BETA=0.1       # Weight for Prosody loss (auxiliary task) 
export LR=5e-5      # Learning rate
export ACC=1        # Gradient accumulation steps (effective batch size = batch_size * acc = 8)
export WORKER_NUM=1 # Number of workers for data processing

# Run the multi-task learning training script
python paper_code/run_emotion.py \
    --model_name_or_path facebook/$MODEL \
    --vocab_file paper_code/david_vocab.json \
    --train_json json_data/ser_audio_features_wav_train.jsonl \
    --val_json json_data/ser_audio_features_wav_val.jsonl \
    --audio_base_path "./AUDIO/" \
    --output_dir output3/final2/alpha0.1beta0 \
    --cache_dir cache/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $ACC \
    --alpha $ALPHA \
    --beta $BETA \
    --dataset_name emotion \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 50 \
    --logging_dir final_log2/beta0/alpha0.1beta0 \
    --load_best_model_at_end \
    --metric_for_best_model eval_acc \
    --do_train \
    --do_eval \
    --learning_rate $LR \
    --preprocessing_num_workers $WORKER_NUM \
    --dataloader_num_workers $WORKER_NUM \
    --freeze_feature_extractor \
    --fp16

echo "Training completed!"