#!/usr/bin/env python3
"""run_emotion.py

Entry script for multi-task speech processing:
  • Automatic Speech Recognition (CTC)
  • Prosody classification (auxiliary) - word-level predictions
  • Emotion recognition (main)

Major Steps
-----------
1. Parse CLI arguments for model, data and training.
2. Build an `Orthography` object to handle tokenisation rules.
3. Instantiate `MTLModel` with a shared backbone and three task heads.
4. Pre-process datasets (audio + labels).
5. Train/evaluate with a custom `CTCTrainer` that supports multi-task losses.
6. Compute metrics: WER (ASR), prosody accuracy, emotion accuracy.

This file aims to be readable – unnecessary commented-out code has been
removed and all major blocks are documented.
"""

import copy
import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import librosa
from lang_trans import arabic

import soundfile as sf

from model import MTLModel
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.tensorboard import SummaryWriter  # tb
import os

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoProcessor,
    WhisperProcessor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor,
    AutoFeatureExtractor,
    is_apex_available,
    trainer_utils,
    EarlyStoppingCallback,
)


if is_apex_available():
    from apex import amp

logger = logging.getLogger(__name__)

writer = SummaryWriter(log_dir="logging_dir")   # tb


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    alpha: Optional[float] = field(
        default=0.01,
        metadata={"help": "loss_ser + alpha * loss_ctc"},
    )
    beta: Optional[float] = field(
        default=0.05,
        metadata={"help": "loss_ser + beta * loss_prosody"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to vocabulary file for tokenizer."}
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)
    
    # Also set the model logger to the same level
    logging.getLogger("model").setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_json: str = field(
        metadata={"help": "Path to training JSONL file"}
    )
    val_json: str = field(
        metadata={"help": "Path to validation JSONL file"}
    )
    dataset_name: str = field(
        default='emotion', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    audio_base_path: str = field(
        default="",
        metadata={
            "help": "Base path for audio files (not needed if using preprocessed data)"}
    )
    test_json: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test JSONL file"}
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={
            "help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file."},
    )


@dataclass
class Orthography:
    """
    Orthography scheme used for text normalization and tokenization.

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        vocab_file (:obj:`str`, `optional`, defaults to :obj:`None`):
            File containing the vocabulary.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for delimiting words; it needs to be in the vocabulary.
        translation_table (:obj:`Dict[str, str]`, `optional`, defaults to :obj:`{}`):
            Table to use with `str.translate()` when preprocessing text (e.g., "-" -> " ").
        words_to_remove (:obj:`Set[str]`, `optional`, defaults to :obj:`set()`):
            Words to remove when preprocessing text (e.g., "sil").
        untransliterator (:obj:`Callable[[str], str]`, `optional`, defaults to :obj:`None`):
            Function that untransliterates text back into native writing system.
    """

    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None
    tokenizer: Optional[str] = None

    @classmethod
    def from_name(cls, name: str):
        if name == "librispeech":
            return cls()
        if name == "timit":
            return cls(
                do_lower_case=True,
                # break compounds like "quarter-century-old" and replace pauses "--"
                translation_table=str.maketrans({"-": " "}),
            )
        if name == "buckwalter":
            translation_table = {
                "-": " ",  # sometimes used to represent pauses
                "^": "v",  # fixing "tha" in arabic_speech_corpus dataset
            }
            return cls(
                vocab_file=pathlib.Path(__file__).parent.joinpath(
                    "vocab/buckwalter.json"),
                word_delimiter_token="/",  # "|" is Arabic letter alef with madda above
                translation_table=str.maketrans(translation_table),
                # fixing "sil" in arabic_speech_corpus dataset
                words_to_remove={"sil"},
                untransliterator=arabic.buckwalter.untransliterate,
            )
        raise ValueError(f"Unsupported orthography: '{name}'.")

    def preprocess_for_training(self, text: str) -> str:
        # TODO(elgeish) return a pipeline (e.g., from jiwer) instead? Or rely on branch predictor as is
        if len(self.translation_table) > 0:
            text = text.translate(self.translation_table)
        if len(self.words_to_remove) == 0:
            try:
                text = " ".join(text.split())  # clean up whitespaces
            except:
                text = "NULL"
        else:
            # and clean up whilespaces
            text = " ".join(w for w in text.split()
                            if w not in self.words_to_remove)
        return text

    def create_processor(self, model_args: ModelArguments) -> AutoProcessor:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        # if vocab_file is provided, use it to create the tokenizer
        if self.vocab_file:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                vocab_file=self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        # if vocab_file is not provided, use the tokenizer from the model_args
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )

        if "whisper" in model_args.model_name_or_path.lower():
            processor = WhisperProcessor(feature_extractor, tokenizer)
        elif "wav2vec2-bert" in model_args.model_name_or_path.lower():
            processor = Wav2Vec2BertProcessor(feature_extractor, tokenizer)
        else:  # wav2vec2, xlsr, mms
            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

        return processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor/Wav2Vec2BertProcessor/WhisperProcessor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    # get the processor from the model name or path
    processor: Union[AutoProcessor, WhisperProcessor, Wav2Vec2BertProcessor, Wav2Vec2Processor] = field(
        metadata={"help": "The processor used for processing the data."}
    )
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # Indicates no need to adjust padded sequence length to a multiple of some integer
    pad_to_multiple_of_labels: Optional[int] = None
    audio_only = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        
        # Add debug logging
        if not self.audio_only and len(features) > 0:
            logger.debug(f"First feature keys: {features[0].keys()}")
            if "labels" in features[0]:
                logger.debug(f"First labels sample: {features[0]['labels'][:5] if isinstance(features[0]['labels'], list) else features[0]['labels']}")
            if "prosody_labels" in features[0]:
                logger.debug(f"First prosody_labels: {features[0]['prosody_labels']}")
            if "emotion_labels" in features[0]:
                logger.debug(f"First emotion_labels: {features[0]['emotion_labels']}")
        
        if self.audio_only is False:
            # Extract different label types with robust error handling
            label_features = []
            prosody_labels = []
            emotion_labels = []

            for feature in features:
                # Handle labels - they should already be tokenized from prepare_dataset
                labels = feature.get("labels", None)
                if labels is None:
                    # Use padding token if labels are missing
                    logger.warning(
                        "Found feature with missing labels, using padding token")
                    label_features.append(
                        {"input_ids": [self.processor.tokenizer.pad_token_id]})
                elif isinstance(labels, list) and len(labels) > 0:
                    # Filter out None values and ensure all are integers
                    filtered_labels = [x for x in labels if x is not None and isinstance(x, int)]
                    if filtered_labels:
                        label_features.append({"input_ids": filtered_labels})
                    else:
                        logger.warning(
                            f"All labels were None or invalid, using padding token")
                        label_features.append(
                            {"input_ids": [self.processor.tokenizer.pad_token_id]})
                else:
                    logger.warning(
                        f"Invalid labels format: {type(labels)}, using padding token")
                    label_features.append(
                        {"input_ids": [self.processor.tokenizer.pad_token_id]})

                # Handle prosody labels
                prosody = feature.get("prosody_labels", None)
                if prosody is None:
                    logger.warning("Missing prosody_labels, using empty list")
                    prosody_labels.append([])
                else:
                    prosody_labels.append(prosody)

                # Handle emotion labels
                emotion = feature.get("emotion_labels", None)
                if emotion is None:
                    logger.warning("Missing emotion_labels, using 0")
                    emotion_labels.append(0)
                else:
                    emotion_labels.append(emotion)

        # This method will pad the input features to make them have the same length and return a batch tensor
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.audio_only is False:
            # Use the tokenizer directly instead of deprecated as_target_processor
            try:
                tokenizer = self.processor.tokenizer
                labels_batch = tokenizer.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.error(f"Error during label padding: {e}")
                logger.error(f"Label features sample: {label_features[:2]}")
                raise

            # replace padding with -100 to ignore loss correctly
            ctc_labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

            # Pad prosody labels to match the number of words in the batch
            max_prosody_len = max(
                len(pl) if pl else 0 for pl in prosody_labels)
            padded_prosody_labels = []
            for pl in prosody_labels:
                if not pl:  # Handle empty prosody labels
                    padded = [-100] * max_prosody_len
                else:
                    # Pad with -100 to ignore in loss computation
                    padded = pl + [-100] * (max_prosody_len - len(pl))
                padded_prosody_labels.append(padded)

            # Convert to tensors
            prosody_tensor = torch.tensor(padded_prosody_labels)
            emotion_tensor = torch.tensor(emotion_labels)

            # Log the shapes for debugging
            logger.debug(f"Batch input shape: {batch['input_values'].shape}")
            logger.debug(f"CTC labels shape: {ctc_labels.shape}")
            logger.debug(f"Prosody labels shape: {prosody_tensor.shape}")
            logger.debug(f"Emotion labels shape: {emotion_tensor.shape}")

            # labels = (ctc_labels, prosody_labels, emotion_labels)
            batch["labels"] = (ctc_labels, prosody_tensor, emotion_tensor)

        return batch


# Trainer class
class CTCTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for training by moving tensors to the appropriate device.
        Special handling for 'labels' which contains a list of tensors rather than a single tensor.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    kwargs.update(
                        dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)
            if k == 'labels':  # labels are list of tensors, not a single tensor, special handling here
                for i in range(len(inputs[k])):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k][i].dtype != torch.int64:
                        kwargs.update(
                            dict(dtype=self.args.hf_deepspeed_config.dtype()))
                    inputs[k][i] = inputs[k][i].to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            num_items_in_batch (:obj:`int`):
                Number of items in the batch.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Use the parent class implementation for mixed precision
        # This ensures proper handling of automatic mixed precision
        if self.use_cpu_amp:
            with torch.cpu.amp.autocast(enabled=True, dtype=self.amp_dtype):
                loss = self.compute_loss(model, inputs)
        else:
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Use the accelerator's backward method which handles mixed precision properly
        self.accelerator.backward(loss)

        return loss.detach()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set remove_unused_columns to False to keep our custom columns
    training_args.remove_unused_columns = False

    configure_logger(model_args, training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    orthography = Orthography.from_name(data_args.orthography.lower())
    # set the tokenizer if provided in model_args
    if model_args.tokenizer is not None:
        orthography.tokenizer = model_args.tokenizer
    # set the vocab_file if provided in model_args
    if model_args.vocab_file is not None:
        orthography.vocab_file = model_args.vocab_file
    processor = orthography.create_processor(model_args)

    if data_args.dataset_name == 'emotion':
        # load the train data from the json file from model_args.train_json
        if data_args.train_json is not None:
            train_dataset = datasets.load_dataset(
                'json', data_files=data_args.train_json, cache_dir=model_args.cache_dir)['train']
        else:
            raise ValueError(
                "Please provide the path to the training JSONL file using --train_json argument.")

        # load the validation data from the json file from model_args.val_json
        if data_args.val_json is not None:
            val_dataset = datasets.load_dataset(
                'json', data_files=data_args.val_json, cache_dir=model_args.cache_dir)['train']
        else:
            raise ValueError(
                "Please provide the path to the validation JSONL file using --val_json argument.")

        # Two prosody classes (binary: non-prosodic vs prosodic)
        # Emotion labels are already integers in the data

    model = MTLModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=True,  # training_args.gradient_checkpointing,
        vocab_size=len(processor.tokenizer),
        ser_cls_len=9,  # 9 emotion classes
        prosody_cls_len=2,  # Binary prosody classification
        alpha=model_args.alpha,
        beta=model_args.beta,
    )

    logger.info("alpha = ")
    logger.info(model_args.alpha)
    logger.info("beta = ")
    logger.info(model_args.beta)

    # datasets.load_metric was deprecated; fall back to `evaluate` if unavailable
    try:
        wer_metric = datasets.load_metric("wer")
    except AttributeError:
        import evaluate
        wer_metric = evaluate.load("wer")

    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    vocabulary_chars_str = "".join(
        t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        # allow space in addition to chars in vocabulary
        rf"[^\s{re.escape(vocabulary_chars_str)}]",
        flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
    )
    text_updates = []
    
    # Log tokenizer vocabulary info for debugging
    logger.info(f"Tokenizer vocab size: {len(processor.tokenizer)}")
    logger.info(f"Pad token: {processor.tokenizer.pad_token} (id: {processor.tokenizer.pad_token_id})")
    logger.info(f"Vocabulary sample: {list(processor.tokenizer.get_vocab().items())[:10]}")

    def prepare_example(example, audio_only=False):
        """
        Prepare a single example for training by loading audio and preprocessing text.

        Args:
            example: Dataset example containing audio file path and text
            audio_only: If True, skip text preprocessing

        Returns:
            example: Updated example with loaded audio and preprocessed text
        """
        # Load audio file and resample to target sampling rate
        example["speech"], example["sampling_rate"] = librosa.load(
            os.path.join(data_args.audio_base_path, example["audio_filepath"]),
            sr=target_sr
        )

        # Calculate duration if max duration constraint is specified
        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(
                example["speech"]) / example["sampling_rate"]

        if audio_only is False:
            # Handle missing or None words
            if "words" not in example or example["words"] is None:
                logger.warning(
                    f"Missing or None words in example: {example.get('audio_filepath', 'unknown')}")
                example["full_text"] = ""
            else:
                # join words with word delimiter token
                text = " ".join(example["words"]).strip()
                # add to example dict
                example["full_text"] = text

                # Normalize and clean up text; order matters!
                updated_text = orthography.preprocess_for_training(text)
                updated_text = vocabulary_text_cleaner.sub("", updated_text)

                # Track text changes for debugging/logging purposes
                if updated_text != text:
                    text_updates.append((text, updated_text))
                    example["full_text"] = updated_text

        return example

    if training_args.do_train:
        train_dataset = train_dataset.map(prepare_example)
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            prepare_example, fn_kwargs={'audio_only': True})

    elif training_args.do_eval:
        val_dataset = val_dataset.map(prepare_example)

    if data_args.max_duration_in_seconds is not None:
        logger.info("data_args.max_duration_in_seconds is not None")

        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds

        if training_args.do_train:
            old_train_size = len(train_dataset)
            train_dataset = train_dataset.filter(
                filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(train_dataset) > old_train_size:
                logger.warning(
                    f"Filtered out {len(train_dataset) - old_train_size} train example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
        if training_args.do_predict or training_args.do_eval:
            old_val_size = len(val_dataset)
            val_dataset = val_dataset.filter(
                filter_by_max_duration, remove_columns=["duration_in_seconds"])

            logger.info("after filter, val_dataset: ")
            logger.info(val_dataset)

            if len(val_dataset) > old_val_size:
                logger.warning(
                    f"Filtered out {len(val_dataset) - old_val_size} validation example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )

    logger.warning(
        f"Updated {len(text_updates)} transcript(s) using '{data_args.orthography}' orthography rules.")
    if logger.isEnabledFor(logging.DEBUG):
        for original_text, updated_text in text_updates:
            logger.debug(
                f'Updated text: "{original_text}" -> "{updated_text}"')
    text_updates = None

    def prepare_dataset(batch, audio_only=False):
        """
        Prepare a batch of dataset examples for training by processing audio and creating multi-task labels.

        Args:
            batch: Batch of examples containing audio, text, prosody annotations, and emotion data
            audio_only: If True, skip text and label processing

        Returns:
            batch: Updated batch with processed input_values and multi-task labels
        """
        try:
            # Check that all files have the correct sampling rate
            assert (
                len(set(batch["sampling_rate"])) == 1
            ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

            # Process audio features
            batch["input_values"] = processor(
                batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

            if audio_only is False:
                # Check if 'full_text' exists, if not create it from 'words'
                if 'full_text' not in batch:
                    # Create full_text from words if it doesn't exist
                    batch_full_texts = []
                    for words in batch['words']:
                        if words is None:
                            batch_full_texts.append("")
                        else:
                            batch_full_texts.append(
                                " ".join(words).strip() if words else "")
                    batch['full_text'] = batch_full_texts

                # Handle potential None values in text
                safe_texts = []
                full_texts = batch['full_text']

                # Ensure we have a list
                if not isinstance(full_texts, list):
                    full_texts = [full_texts]

                # Validate and clean each text
                for i, text in enumerate(full_texts):
                    # Multiple checks for various types of invalid values
                    if (text is None or
                        (isinstance(text, str) and text.strip() == "") or
                        (isinstance(text, float) and np.isnan(text)) or
                            text == ""):
                        safe_texts.append("")  # Use empty string
                    else:
                        # Convert to string and ensure it's not empty
                        text_str = str(text).strip()
                        safe_texts.append(text_str if text_str else "")

                # Process text labels using the tokenizer directly
                # Avoid using processor which might have issues with None handling
                all_input_ids = []
                tokenizer = processor.tokenizer

                for i, text in enumerate(safe_texts):
                    try:
                        # Make absolutely sure text is a string
                        if not isinstance(text, str):
                            text = ""

                        # Handle empty text by using a minimal token sequence
                        if not text:
                            # Use padding token or a minimal sequence
                            all_input_ids.append([tokenizer.pad_token_id])
                        else:
                            # Directly call the tokenizer
                            encoding = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
                            tokens = encoding["input_ids"] if isinstance(encoding, dict) else encoding
                            
                            # Ensure we have a valid list of integers
                            if isinstance(tokens, list) and len(tokens) > 0:
                                # Filter any potential None values and non-integers
                                filtered_tokens = [int(t) for t in tokens if t is not None and str(t).isdigit()]
                                if filtered_tokens:
                                    all_input_ids.append(filtered_tokens)
                                else:
                                    all_input_ids.append([tokenizer.pad_token_id])
                            else:
                                all_input_ids.append([tokenizer.pad_token_id])
                    except Exception as e:
                        logger.warning(
                            f"Error tokenizing text at index {i}: '{text}' - {e}")
                        # Fallback: use padding token
                        all_input_ids.append([tokenizer.pad_token_id])

                batch["labels"] = all_input_ids

                # Store prosody annotations and emotion labels separately
                # These will be handled by the data collator
                # Handle potentially missing prosody_annotations
                if "prosody_annotations" in batch:
                    batch["prosody_labels"] = batch["prosody_annotations"]
                else:
                    logger.warning("Missing prosody_annotations in batch")
                    # Create empty prosody labels for each example
                    batch["prosody_labels"] = [[]
                                               for _ in range(len(batch["speech"]))]

                # Handle potentially missing emotion labels
                if "emotion" in batch:
                    batch["emotion_labels"] = batch["emotion"]
                else:
                    logger.warning("Missing emotion labels in batch")
                    # Create default emotion labels (0) for each example
                    batch["emotion_labels"] = [
                        0 for _ in range(len(batch["speech"]))]

        except Exception as e:
            logger.error(f"Error in prepare_dataset: {e}")
            logger.error(f"Batch keys: {list(batch.keys())}")
            if 'full_text' in batch:
                logger.error(
                    f"Full text sample: {batch['full_text'][:3] if isinstance(batch['full_text'], list) else batch['full_text']}")
            raise

        return batch

    if training_args.do_train:
        train_dataset = train_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=1,  # Use single process for debugging
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            prepare_dataset,
            fn_kwargs={'audio_only': True},
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=1,  # Use single process for debugging
        )

    elif training_args.do_eval:
        logger.info("do_eval")
        val_dataset = val_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=1,  # Use single process for debugging
        )

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)

    def compute_metrics(pred):
        """
        Compute evaluation metrics for multi-task learning (CTC + prosody classification + emotion classification).

        Args:
            pred: Prediction object containing predictions and label_ids
                - predictions[0]: CTC logits for speech recognition
                - predictions[1]: Classification logits for prosody recognition (word-level)
                - predictions[2]: Classification logits for emotion recognition  
                - label_ids[0]: CTC labels (true transcription tokens)
                - label_ids[1]: Prosody labels (true prosody classes, word-level)
                - label_ids[2]: Emotion labels (true emotion classes)

        Returns:
            dict: Dictionary containing accuracy, WER, and other metrics
        """
        # Log shapes for debugging
        logger.info(f"Prosody predictions shape: {pred.predictions[1].shape}")
        logger.info(f"Prosody labels shape: {pred.label_ids[1].shape}")
        
        # Calculate prosody classification accuracy (word-level)
        prosody_pred_logits = pred.predictions[1]
        # Get the class with highest probability for each word
        prosody_pred_ids = np.argmax(prosody_pred_logits, axis=-1)

        # Ensure shapes match - prosody predictions should already be pooled to word-level
        if prosody_pred_ids.shape != pred.label_ids[1].shape:
            logger.error(f"Shape mismatch: predictions {prosody_pred_ids.shape} vs labels {pred.label_ids[1].shape}")
            # Use a default accuracy
            prosody_correct = 0
            prosody_total = 1
        else:
            # Flatten predictions and labels, ignoring padding (-100)
            prosody_labels_flat = pred.label_ids[1].flatten()
            prosody_pred_flat = prosody_pred_ids.flatten()

            # Only evaluate on non-padded positions
            valid_mask = prosody_labels_flat != -100
            if valid_mask.sum() > 0:
                prosody_correct = (
                    prosody_pred_flat[valid_mask] == prosody_labels_flat[valid_mask]).sum().item()
                prosody_total = valid_mask.sum().item()
            else:
                prosody_correct = 0
                prosody_total = 1

        # Calculate emotion classification accuracy
        emotion_pred_logits = pred.predictions[2]
        emotion_pred_ids = np.argmax(emotion_pred_logits, axis=-1)
        emotion_total = len(pred.label_ids[2])
        emotion_correct = (emotion_pred_ids == pred.label_ids[2]).sum().item()

        # Calculate CTC word error rate
        ctc_pred_logits = pred.predictions[0]
        ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
        pred.label_ids[0][pred.label_ids[0] == -
                          100] = processor.tokenizer.pad_token_id
        ctc_pred_str = processor.batch_decode(ctc_pred_ids)
        ctc_label_str = processor.batch_decode(
            pred.label_ids[0], group_tokens=False)

        # Debug logging for transcription comparison
        if logger.isEnabledFor(logging.DEBUG):
            for reference, predicted in zip(ctc_label_str, ctc_pred_str):
                logger.debug(f'reference: "{reference}"')
                logger.debug(f'predicted: "{predicted}"')
                if orthography.untransliterator is not None:
                    logger.debug(
                        f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
                    logger.debug(
                        f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

        wer = wer_metric.compute(
            predictions=ctc_pred_str, references=ctc_label_str)

        return {
            "prosody_acc": prosody_correct / prosody_total if prosody_total > 0 else 0,
            "prosody_correct": prosody_correct,
            "prosody_total": prosody_total,
            "wer": wer,
            "emotion_acc": emotion_correct / emotion_total if emotion_total > 0 else 0,
            "emotion_correct": emotion_correct,
            "emotion_total": emotion_total,
            "strlen": len(ctc_label_str)
        }

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    logger.info("val_dataset")
    logger.info(val_dataset)

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=200, early_stopping_threshold=0.01),
        ],
    )

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    if training_args.do_predict:
        logger.info('******* Predict ********')

        data_collator.audio_only = True

        logger.info("before predict , val_dataset:")
        logger.info(val_dataset)

        predictions, labels, metrics = trainer.predict(
            val_dataset, metric_key_prefix="predict")
        logits_ctc, logits_prosody, logits_emotion = predictions

        # For emotion predictions (utterance-level)
        emotion_pred_ids = np.argmax(logits_emotion, axis=-1)
        emotion_pred_probs = F.softmax(
            torch.from_numpy(logits_emotion).float(), dim=-1)

        # For prosody predictions (word-level)
        prosody_pred_ids = np.argmax(logits_prosody, axis=-1)
        prosody_pred_probs = F.softmax(
            torch.from_numpy(logits_prosody).float(), dim=-1)

        with open(data_args.output_file, 'w') as f:
            for i in range(len(emotion_pred_ids)):
                # Write file info and emotion prediction
                f.write(val_dataset[i]['audio_filepath'].split("/")[-1] + " ")
                f.write(
                    f"duration: {len(val_dataset[i]['input_values']) / 16000:.2f}s ")
                f.write(f"emotion_pred: {emotion_pred_ids[i]} ")

                # Write emotion probabilities
                f.write("emotion_probs: ")
                for j in range(9):  # 9 emotion classes
                    f.write(f"{emotion_pred_probs[i][j].item():.4f} ")

                # Write word-level prosody predictions
                f.write("prosody_preds: ")
                word_prosody_preds = prosody_pred_ids[i]
                # Only write non-padded predictions
                valid_length = (word_prosody_preds != -100).sum()
                for j in range(valid_length):
                    f.write(f"{word_prosody_preds[j]} ")

                f.write('\n')
        f.close()

    elif training_args.do_eval:
        predictions, labels, metrics = trainer.predict(
            val_dataset, metric_key_prefix="eval")
        logits_ctc, logits_prosody, logits_emotion = predictions

        # Emotion accuracy
        emotion_pred_ids = np.argmax(logits_emotion, axis=-1)
        emotion_correct = np.sum(emotion_pred_ids == labels[2])
        emotion_acc = emotion_correct / len(emotion_pred_ids)

        # Prosody accuracy (word-level)
        prosody_pred_ids = np.argmax(logits_prosody, axis=-1)
        prosody_labels_flat = labels[1].flatten()
        prosody_pred_flat = prosody_pred_ids.flatten()
        valid_mask = prosody_labels_flat != -100
        prosody_correct = (
            prosody_pred_flat[valid_mask] == prosody_labels_flat[valid_mask]).sum()
        prosody_total = valid_mask.sum()
        prosody_acc = prosody_correct / prosody_total if prosody_total > 0 else 0

        print(f'Emotion - correct: {emotion_correct}, acc: {emotion_acc:.4f}')
        print(
            f'Prosody - correct: {prosody_correct}, total: {prosody_total}, acc: {prosody_acc:.4f}')

    writer.close()


if __name__ == "__main__":
    main()