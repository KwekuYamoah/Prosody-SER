#!/usr/bin/env python3
"""
Test script to verify blank penalty is working correctly
"""

import torch
import torch.nn.functional as F
from sample_code.scripts.ctc_decoder import CTCLoss


def test_blank_penalty():
    """Test the blank penalty with different scenarios"""

    print("Testing Blank Penalty Implementation")
    print("=" * 50)

    # Create CTC loss with strong penalty
    ctc_loss = CTCLoss(
        blank_id=0,
        blank_penalty=50.0,  # Strong penalty
        blank_threshold=0.3,  # Low threshold
        entropy_weight=0.01
    )

    # Test case 1: Low blank probability (should not trigger penalty)
    print("\nTest 1: Low blank probability (0.2)")
    log_probs = torch.randn(10, 2, 100)  # (T, N, C)
    # Make blank token (index 0) have low probability
    log_probs[:, :, 0] = -2.0  # Low probability for blank
    log_probs[:, :, 1:] = 0.0  # Higher probability for other tokens

    # The `targets` tensor must be a 1D tensor containing the concatenated labels for the entire batch.
    # Since target_lengths is [5, 5], the total length must be 10.
    targets = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    input_lengths = torch.tensor([10, 10])
    target_lengths = torch.tensor([5, 5])

    total_loss, details = ctc_loss(
        log_probs, targets, input_lengths, target_lengths)
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Blank Penalty: {details['blank_penalty']:.4f}")
    print(f"  Avg Blank Prob: {details['avg_blank_prob']:.4f}")

    # Test case 2: High blank probability (should trigger penalty)
    print("\nTest 2: High blank probability (0.6)")
    log_probs = torch.randn(10, 2, 100)
    # Make blank token have high probability
    log_probs[:, :, 0] = 1.0  # High probability for blank
    log_probs[:, :, 1:] = -1.0  # Lower probability for other tokens

    total_loss, details = ctc_loss(
        log_probs, targets, input_lengths, target_lengths)
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Blank Penalty: {details['blank_penalty']:.4f}")
    print(f"  Avg Blank Prob: {details['avg_blank_prob']:.4f}")

    # Test case 3: Very high blank probability (should trigger strong penalty)
    print("\nTest 3: Very high blank probability (0.8)")
    log_probs = torch.randn(10, 2, 100)
    # Make blank token have very high probability
    log_probs[:, :, 0] = 2.0  # Very high probability for blank
    log_probs[:, :, 1:] = -2.0  # Very low probability for other tokens

    total_loss, details = ctc_loss(
        log_probs, targets, input_lengths, target_lengths)
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Blank Penalty: {details['blank_penalty']:.4f}")
    print(f"  Avg Blank Prob: {details['avg_blank_prob']:.4f}")

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    test_blank_penalty()
