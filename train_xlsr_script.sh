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


srun python3 main.py extracted_input_features.json 20_06 1
#srun python3 main.py conformed_librispeech_dataset.json 17_06 2
#srun python3 helper_scripts.py
