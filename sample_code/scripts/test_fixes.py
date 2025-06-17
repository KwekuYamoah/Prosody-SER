#!/usr/bin/env python3
"""
Improved training script for MTL model following the paper's approach.
Cleaner, more efficient, and easier to debug.
"""

import os
import sys
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import torch
from torch import nn
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoProcessor,
    AutoFeatureExtractor,
    AutoTokenizer,
    set_seed,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from sklearn.metrics import accuracy_score, f1_score
import wandb

from sample_code.scripts.backbone_models import BACKBONE_CONFIGS, BackboneModel
from sample_code.scripts.mtl_model import MTLModel, MTLOutput
from sample_code.scripts.mtl_dataset import MTLDataset, DataCollatorMTLWithPadding
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.tokenizer import SentencePieceTokenizer

# For ASR metrics
try:
    from jiwer import wer, cer
except ImportError:
    print("jiwer not installed. ASR metrics will not be available.")
    wer = cer = None

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    backbone_name: str = field(
        default="whisper",
        metadata={
            "help": "Pretrained model name  in [whisper, xlsr, mms, wav2vec2-bert]"}
    )
    vocab_size: int = field(
        default=16000,
        metadata={"help": "Vocabulary size for ASR"}
    )
    emotion_classes: int = field(
        default=9,
        metadata={"help": "Number of emotion classes"}
    )
    alpha_asr: float = field(
        default=0.1,
        metadata={"help": "Weight for ASR auxiliary task"}
    )
    alpha_prosody: float = field(
        default=0.1,
        metadata={"help": "Weight for Prosody auxiliary task"}
    )
    freeze_feature_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to freeze feature extractor"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for pretrained models"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_json: str = field(
        metadata={"help": "Path to training JSONL file"}
    )
    val_json: str = field(
        metadata={"help": "Path to validation JSONL file"}
    )
    audio_base_path: str = field(
        metadata={"help": "Base path for audio files"}
    )
    test_json: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test JSONL file"}
    )
    max_duration_in_seconds: Optional[float] = field(
        default=300.0,
        metadata={"help": "Maximum audio duration"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )
    emotion_label_map: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string mapping emotion names to indices"}
    )


class MTLTrainer(Trainer):
    """Custom trainer for multi-task learning"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute multi-task loss"""
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, MTLOutput) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step to handle multiple outputs"""
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
                loss = outputs.loss if isinstance(
                    outputs, MTLOutput) else outputs[0]

                if isinstance(outputs, MTLOutput):
                    # Extract logits for each task
                    ser_logits = outputs.ser_logits
                    asr_logits = outputs.asr_logits
                    prosody_logits = outputs.prosody_logits
                    logits = (ser_logits, asr_logits, prosody_logits)
                else:
                    # Assuming (loss, ser, asr, prosody, ...)
                    logits = outputs[1:4]

        if prediction_loss_only:
            return (loss, None, None)

        # Return predictions and labels
        labels = inputs.get("labels", None)
        return (loss, logits, labels)


def compute_metrics(eval_pred: EvalPrediction, tokenizer="None") -> Dict[str, float]:
    """Compute metrics for all tasks"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    metrics = {}

    # Unpack predictions and labels
    if len(predictions) == 3:
        ser_preds, asr_preds, prosody_preds = predictions
    else:
        logger.warning(
            f"Unexpected prediction format: {len(predictions)} elements")
        return metrics

    if isinstance(labels, tuple) and len(labels) == 3:
        asr_labels_info, ser_labels, prosody_labels = labels
        if isinstance(asr_labels_info, tuple):
            asr_labels, asr_lengths = asr_labels_info
        else:
            asr_labels = asr_labels_info
            asr_lengths = None
    else:
        logger.warning(f"Unexpected label format")
        return metrics

    # 1. SER metrics (emotion classification)
    if ser_preds is not None and ser_labels is not None:
        ser_predictions = np.argmax(ser_preds, axis=-1)
        ser_accuracy = accuracy_score(ser_labels, ser_predictions)
        ser_f1 = f1_score(ser_labels, ser_predictions, average='weighted')

        metrics['ser_accuracy'] = ser_accuracy
        metrics['ser_f1'] = ser_f1

    # 2. ASR metrics (if tokenizer provided and jiwer available)
    if asr_preds is not None and asr_labels is not None and tokenizer is not None and wer is not None:
        # Decode predictions
        asr_predictions = np.argmax(asr_preds, axis=-1)

        # Simple greedy decoding (you might want to use CTC decoder)
        pred_texts = []
        ref_texts = []

        for i in range(len(asr_predictions)):
            # Decode prediction
            pred_ids = asr_predictions[i]
            # Remove repetitions and blank tokens
            decoded_ids = []
            prev_id = -1
            for id in pred_ids:
                if id != 0 and id != prev_id:  # 0 is blank token
                    decoded_ids.append(id)
                prev_id = id

            pred_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)
            pred_texts.append(pred_text)

            # Decode reference
            if asr_lengths is not None:
                ref_ids = asr_labels[i][:asr_lengths[i]] if i < len(
                    asr_lengths) else asr_labels[i]
            else:
                ref_ids = asr_labels[i]
            ref_ids = ref_ids[ref_ids != -100]  # Remove padding
            ref_text = tokenizer.decode(
                ref_ids.tolist(), skip_special_tokens=True)
            ref_texts.append(ref_text)

        # Compute WER and CER
        word_error_rate = wer(ref_texts, pred_texts)
        char_error_rate = cer(ref_texts, pred_texts)

        metrics['asr_wer'] = word_error_rate
        metrics['asr_cer'] = char_error_rate

    # 3. Prosody metrics (binary sequence classification)
    if prosody_preds is not None and prosody_labels is not None:
        # Flatten predictions and labels
        prosody_predictions = (prosody_preds > 0).astype(float).flatten()
        prosody_labels_flat = prosody_labels.flatten()

        # Remove padding (assuming -100 or negative values are padding)
        mask = prosody_labels_flat >= 0
        if mask.sum() > 0:
            prosody_predictions_masked = prosody_predictions[mask]
            prosody_labels_masked = prosody_labels_flat[mask]

            prosody_accuracy = accuracy_score(
                prosody_labels_masked, prosody_predictions_masked)
            prosody_f1 = f1_score(prosody_labels_masked,
                                  prosody_predictions_masked, average='binary')

            metrics['prosody_accuracy'] = prosody_accuracy
            metrics['prosody_f1'] = prosody_f1

    return metrics


def load_datasets(data_args: DataArguments) -> Dict[str, List[Dict]]:
    """Load datasets from JSONL files"""
    import json

    datasets_dict = {}

    # Load training data
    with open(data_args.train_json, 'r') as f:
        train_data = [json.loads(line) for line in f]
    datasets_dict['train'] = train_data

    # Load validation data
    with open(data_args.val_json, 'r') as f:
        val_data = [json.loads(line) for line in f]
    datasets_dict['validation'] = val_data

    # Load test data if provided
    if data_args.test_json:
        with open(data_args.test_json, 'r') as f:
            test_data = [json.loads(line) for line in f]
        datasets_dict['test'] = test_data

    # Update audio paths
    for split in datasets_dict:
        for item in datasets_dict[split]:
            item['audio_filepath'] = os.path.join(
                data_args.audio_base_path, item['audio_filepath'])

    return datasets_dict


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from json file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"🚀 Starting MTL Training Pipeline")
    print("="*50)
    print(f"\n📱 Device Information:")
    print(f"   Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Number of GPUs available: {torch.cuda.device_count()}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n⚙️ Configuration:")
    print(f"   Training parameters: {training_args}")
    print(f"   Model parameters: {model_args}")
    print(f"   Data parameters: {data_args}")

    # Detect checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(f"\n📂 Checkpoint detected, resuming from: {last_checkpoint}")
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed
    set_seed(training_args.seed)
    print(f"\n🎲 Set random seed to {training_args.seed}")

    # Load datasets
    print("\n" + "="*50)
    print("📊 Stage 1/6: Loading datasets...")
    print("="*50)
    datasets_dict = load_datasets(data_args)
    print(f"   ✓ Loaded {len(datasets_dict['train'])} training samples")
    print(f"   ✓ Loaded {len(datasets_dict['validation'])} validation samples")

    # Load processor and tokenizer based on model type
    print("\n" + "="*50)
    print("🔧 Stage 2/6: Loading processor and tokenizer...")
    print("="*50)
    print(f"   Loading processor for {model_args.backbone_name}")

    # Try to load processor first, then feature extractor
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.backbone_name,
            cache_dir=model_args.cache_dir
        )
    except:
        processor = AutoFeatureExtractor.from_pretrained(
            model_args.backbone_name,
            cache_dir=model_args.cache_dir
        )

    # Load or create tokenizer
    tokenizer_path = os.path.join(training_args.output_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        print(f"   Loading existing tokenizer from {tokenizer_path}")
        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        tokenizer.load_tokenizer()
    else:
        print("   Creating new tokenizer")
        # Extract all text for tokenizer training
        all_texts = []
        for split in datasets_dict:
            for item in datasets_dict[split]:
                if 'words' in item:
                    all_texts.append(" ".join(item['words']))

        # Train tokenizer
        os.makedirs(tokenizer_path, exist_ok=True)
        text_file = os.path.join(tokenizer_path, "training_text.txt")
        with open(text_file, 'w') as f:
            for text in all_texts:
                f.write(text + '\n')

        tokenizer = SentencePieceTokenizer(vocab_size=model_args.vocab_size)
        tokenizer.train_tokenizer(
            text_file, model_prefix=os.path.join(tokenizer_path, "spm"))

    # Parse emotion label map if provided
    emotion_label_map = None
    if data_args.emotion_label_map:
        emotion_label_map = json.loads(data_args.emotion_label_map)

    # Create datasets
    print("\n" + "="*50)
    print("📝 Stage 3/6: Creating datasets...")
    print("="*50)
    train_dataset = MTLDataset(
        datasets_dict['train'],
        processor=processor,
        target_sr=16000,
        max_duration=data_args.max_duration_in_seconds,
        emotion_label_map=emotion_label_map
    )

    val_dataset = MTLDataset(
        datasets_dict['validation'],
        processor=processor,
        target_sr=16000,
        max_duration=data_args.max_duration_in_seconds,
        emotion_label_map=emotion_label_map
    )

    # Create data collator
    data_collator = DataCollatorMTLWithPadding(
        processor=processor,
        tokenizer=tokenizer,
        padding=True,
        return_attention_mask=True
    )

    # Create model configuration
    print("\n" + "="*50)
    print("⚙️ Stage 4/6: Creating model configuration...")
    print("="*50)
    config = MTLConfig(
        backbone_name=model_args.backbone_name,
        vocab_size=tokenizer.get_vocab_size(),
        emotion_classes=model_args.emotion_classes,
        alpha_asr=model_args.alpha_asr,
        alpha_prosody=model_args.alpha_prosody,
        freeze_encoder=model_args.freeze_feature_extractor
    )

    # Create or load model
    print("\n" + "="*50)
    print("🤖 Stage 5/6: Creating/loading model...")
    print("="*50)
    if last_checkpoint is not None:
        print(f"   Loading model from checkpoint {last_checkpoint}")
        model = MTLModel.from_pretrained(last_checkpoint, config=config)
    else:
        print("   Creating new model")
        model = MTLModel(config)

        if model_args.freeze_feature_extractor:
            model.freeze_feature_extractor()

    # Move model to GPU if available
    model = model.to(device)

    # Log model info
    trainable_params = model.num_parameters(only_trainable=True)
    total_params = model.num_parameters(only_trainable=False)
    print(f"\n📊 Model Statistics:")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / (1024**2):.2f} MB")

    # Create trainer
    print("\n" + "="*50)
    print("🎯 Stage 6/6: Setting up trainer...")
    print("="*50)
    trainer = MTLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=processor,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, tokenizer),
    )

    # Training
    if training_args.do_train:
        print("\n" + "="*50)
        print("🚂 Starting training...")
        print("="*50)
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("\n" + "="*50)
        print("✅ Training completed!")
        print("="*50)
        print("\n📊 Training Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

    # Evaluation
    if training_args.do_eval:
        print("\n" + "="*50)
        print("📊 Starting evaluation...")
        print("="*50)
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        print("\n" + "="*50)
        print("✅ Evaluation completed!")
        print("="*50)
        print("\n📊 Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

    # Test evaluation
    if training_args.do_predict and data_args.test_json:
        print("\n" + "="*50)
        print("🧪 Starting test evaluation...")
        print("="*50)
        test_dataset = MTLDataset(
            datasets_dict['test'],
            processor=processor,
            target_sr=16000,
            max_duration=data_args.max_duration_in_seconds,
            emotion_label_map=emotion_label_map
        )

        predictions, labels, metrics = trainer.predict(
            test_dataset, metric_key_prefix="test")

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        print("\n" + "="*50)
        print("✅ Test evaluation completed!")
        print("="*50)
        print("\n📊 Test Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

    print("\n" + "="*50)
    print("🎉 Pipeline completed successfully!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()