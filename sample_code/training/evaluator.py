"""
MTL Evaluator Module - Memory Efficient Version
Handles model evaluation, metric computation, and result visualization
"""

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional

from jiwer import wer, cer

import gc

from sample_code.scripts.ctc_decoder import CTCDecoder


class MTLEvaluator:
    """Class for evaluating MTL model performance"""

    def __init__(self, model, tokenizer, device='cuda', use_amp=True, decode_method='greedy'):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.use_amp = use_amp
        self.decode_method = decode_method
        self.model.to(device)
        self.model.eval()

        # Initialize CTC decoder
        self.ctc_decoder = CTCDecoder(blank_id=0, beam_width=10)

    def compute_asr_metrics_from_texts(self, pred_texts: List[str], target_texts: List[str]) -> Dict:
        """
        Compute ASR metrics (WER and CER) from text lists.
        
        Args:
            pred_texts: List of predicted text strings
            target_texts: List of target text strings
        """
        # Debug print for first few examples
        for i in range(min(3, len(pred_texts))):
            print(f"Debug - Target: '{target_texts[i]}', Predicted: '{pred_texts[i]}'")

        detailed_results = [
            {"predicted": p, "target": t} for p, t in zip(pred_texts[:5], target_texts[:5])
        ]

        # Calculate WER and CER on the decoded text
        wer_score = wer(target_texts, pred_texts) if pred_texts and any(pred_texts) else 1.0
        cer_score = cer(target_texts, pred_texts) if pred_texts and any(pred_texts) else 1.0

        return {
            "wer": wer_score,
            "cer": cer_score,
            "detailed_results": detailed_results,
        }

    def flatten_and_filter_sequences(self, predictions, targets):
        """Flatten sequence predictions and targets, filtering out padding tokens"""
        flat_preds, flat_targets = [], []

        for pred_seq, target_seq in zip(predictions, targets):
            if torch.is_tensor(pred_seq):
                pred_seq = pred_seq.cpu().numpy()
            if torch.is_tensor(target_seq):
                target_seq = target_seq.cpu().numpy()

            valid_mask = target_seq != 0  # Assuming padding value is 0
            flat_preds.extend(pred_seq[valid_mask])
            flat_targets.extend(target_seq[valid_mask])
        return np.array(flat_preds), np.array(flat_targets)

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a data loader with proper CTC handling and memory management"""
        # CRITICAL FIX: Store only text for ASR, not tensors!
        asr_pred_texts = []
        asr_target_texts = []

        # For other tasks, use numpy arrays
        prosody_predictions = []
        prosody_targets = []
        emotion_predictions = []
        emotion_targets = []

        detailed_results = {"prosody": [], "emotion": []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                # Move batch to device
                input_features = batch["input_features"].to(self.device, non_blocking=True)
                asr_targets_batch = batch["asr_targets"].to(self.device, non_blocking=True)
                asr_lengths_batch = batch["asr_lengths"].to(self.device, non_blocking=True)
                prosody_targets_batch = batch["prosody_targets"].to(self.device, non_blocking=True)
                emotion_targets_batch = batch["emotion_targets"].to(self.device, non_blocking=True)

                # Forward pass
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(
                        input_features=input_features,
                        asr_targets=asr_targets_batch,
                        asr_lengths=asr_lengths_batch,
                        prosody_targets=prosody_targets_batch,
                        emotion_targets=emotion_targets_batch,
                    )

                # CRITICAL FIX: Process ASR immediately and store only text
                if 'asr_logits' in outputs and asr_targets_batch is not None:
                    asr_logits = outputs['asr_logits'].detach()
                    
                    # Process each sample in the batch immediately
                    batch_size = asr_logits.size(0)
                    for i in range(batch_size):
                        # Get individual sample and move to CPU
                        logits_i = asr_logits[i].cpu()
                        target_i = asr_targets_batch[i].cpu()
                        length_i = asr_lengths_batch[i].cpu().item()
                        
                        # Decode immediately
                        decoded_ids = self.ctc_decoder.decode_batch(
                            logits_i.unsqueeze(0),  # Add batch dimension
                            lengths=[length_i],
                            method=self.decode_method
                        )[0]  # Get first result
                        
                        # Convert to text
                        pred_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
                        
                        # Process target
                        target_ids = target_i[:length_i].numpy()
                        filtered_ids = [int(id) for id in target_ids if id != self.tokenizer.pad_id]
                        target_text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
                        
                        # Store only text strings
                        asr_pred_texts.append(pred_text)
                        asr_target_texts.append(target_text)
                        
                        # Immediately delete tensor references
                        del logits_i, target_i
                    
                    # Delete the batch tensors
                    del asr_logits

                # Process Prosody - Convert to numpy immediately
                if 'prosody_logits' in outputs:
                    prosody_preds = (outputs['prosody_logits'] > 0).float().detach().cpu().numpy()
                    prosody_targets_np = prosody_targets_batch.detach().cpu().numpy()

                    # Extend with numpy arrays
                    prosody_predictions.extend(prosody_preds)
                    prosody_targets.extend(prosody_targets_np)

                    # Add limited detailed results
                    if len(detailed_results['prosody']) < 20:
                        for i in range(min(len(prosody_preds), 2)):
                            if 'words' in batch and i < len(batch['words']):
                                detailed_results['prosody'].append({
                                    'words': batch['words'][i],
                                    'predicted': prosody_preds[i].tolist(),
                                    'target': prosody_targets_np[i].tolist()
                                })

                # Process Emotion - Convert to numpy immediately
                if 'emotion_logits' in outputs:
                    emotion_preds = outputs['emotion_logits'].argmax(dim=-1).detach().cpu().numpy()
                    emotion_targets_np = emotion_targets_batch.detach().cpu().numpy()

                    # Extend with numpy arrays
                    emotion_predictions.extend(emotion_preds)
                    emotion_targets.extend(emotion_targets_np)

                    # Add limited detailed results
                    if len(detailed_results['emotion']) < 20:
                        emotion_labels = ['anger', 'contempt', 'disgust', 'fear',
                                          'guilt', 'happy', 'sadness', 'shame', 'surprise']
                        for i in range(min(len(emotion_preds), 2)):
                            detailed_results['emotion'].append({
                                'predicted': emotion_labels[emotion_preds[i]] if emotion_preds[i] < len(emotion_labels) else f"class_{emotion_preds[i]}",
                                'target': emotion_labels[emotion_targets_np[i]] if emotion_targets_np[i] < len(emotion_labels) else f"class_{emotion_targets_np[i]}"
                            })

                # Clear CUDA cache and delete references
                del batch, outputs, input_features, asr_targets_batch, prosody_targets_batch, emotion_targets_batch
                
                # Aggressive memory cleanup every 10 batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Calculate metrics with accumulated data
        metrics = {}

        # ASR metrics - now using text lists directly
        if asr_pred_texts:
            metrics['asr'] = self.compute_asr_metrics_from_texts(
                asr_pred_texts,
                asr_target_texts
            )
            # Clear lists after computing metrics
            del asr_pred_texts, asr_target_texts

        # Prosody metrics
        if prosody_predictions:
            flat_preds, flat_targets = self.flatten_and_filter_sequences(
                prosody_predictions, prosody_targets)
            if len(flat_preds) > 0:
                metrics['prosody'] = {
                    'accuracy': accuracy_score(flat_targets, flat_preds),
                    'f1': f1_score(flat_targets, flat_preds, average='weighted', zero_division=0),
                    'detailed_results': detailed_results['prosody'][:5]
                }
            del prosody_predictions, prosody_targets, flat_preds, flat_targets

        # Emotion metrics
        if emotion_predictions:
            metrics['emotion'] = {
                'accuracy': accuracy_score(emotion_targets, emotion_predictions),
                'f1': f1_score(emotion_targets, emotion_predictions, average='weighted', zero_division=0),
                'detailed_results': detailed_results['emotion'][:5]
            }
            del emotion_predictions, emotion_targets

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return metrics

    def print_detailed_results(self, metrics):
        """Print detailed results for each task"""
        print("\nDetailed Evaluation Results:")

        for task in ['asr', 'prosody', 'emotion']:
            if task not in metrics:
                continue

            print(f"\n{task.upper()} Results:")

            # Print metrics
            for metric_name, value in metrics[task].items():
                if metric_name != 'detailed_results' and not isinstance(value, list):
                    print(f"{metric_name}: {value:.4f}")

            # Print sample predictions if available
            if 'detailed_results' in metrics[task] and metrics[task]['detailed_results']:
                print("\nSample Predictions:")
                for i, result in enumerate(metrics[task]['detailed_results'][:5]):
                    print(f"\nExample {i+1}:")
                    if task == 'asr':
                        print(f"Target: {result['target']}")
                        print(f"Predicted: {result['predicted']}")
                    elif task == 'prosody':
                        if 'words' in result:
                            print("Words:", " ".join(result['words']))
                        print("Target Prominence:", " ".join(
                            map(str, result['target'])))
                        print("Predicted Prominence:", " ".join(
                            map(str, result['predicted'])))
                    else:  # emotion
                        print(f"Target Emotion: {result['target']}")
                        print(f"Predicted Emotion: {result['predicted']}")
            else:
                print("\nNo detailed results available for this task.")