#!/bin/bash

#SBATCH --job-name=xlsr_finetuning
#SBATCH --output=./model_outputs/xlsr_model.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dasa@itu.dk
#SBATCH --time=3-00:00:00
#SBATCH --partition=acltr
#SBATCH --mem=200G


# python xlsr_train.py json_data/extracted_ser_audio_features_wav_train.jsonl json_data/extracted_ser_audio_features_wav_val.jsonl json_data/extracted_ser_audio_features_wav_test.jsonl 05_07 1


#srun python3 main.py conformed_librispeech_dataset.json 17_06 2
#srun python3 helper_scripts.py



#!/bin/bash
echo "Running training on multiple with torchrun..."
torchrun \
    --standalone \
    --nproc_per_node=4 \
    xlsr_train_v2.py \
    json_data/extracted_ser_audio_features_wav_train.jsonl \
    json_data/extracted_ser_audio_features_wav_val.jsonl \
    json_data/extracted_ser_audio_features_wav_test.jsonl \
    09_07 \
    1