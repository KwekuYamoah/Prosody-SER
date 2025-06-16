#!/usr/bin/env python3
"""
Test script to verify the paper-style alpha implementation
Tests the core alpha control mechanism following the paper's methodology
"""

import torch
import numpy as np
from datasets import load_dataset, Audio
import os

# Import your modules
from sample_code.scripts.mtl_config import MTLConfig
from sample_code.scripts.mtl_model import MTLModel
from sample_code.scripts.mtl_dataset import MTLDataset
from sample_code.scripts.backbone_models import BACKBONE_CONFIGS, BackboneModel
from sample_code.scripts.tokenizer import SentencePieceTokenizer
from sample_code.training.utils import collate_fn_mtl


def test_alpha_config():
    """Test the paper-style alpha configuration"""
    print("="*50)
    print("TESTING ALPHA CONFIGURATION")
    print("="*50)
    
    # Test 1: Default configuration
    config = MTLConfig(
        backbone_name="whisper",
        alpha_asr=0.1,
        alpha_prosody=0.1
    )
    
    print(f"✓ Default config created")
    print(f"  SER weight: {config.loss_weights['ser']}")
    print(f"  ASR weight: {config.loss_weights['asr']} (alpha_asr: {config.alpha_asr})")
    print(f"  Prosody weight: {config.loss_weights['prosody']} (alpha_prosody: {config.alpha_prosody})")
    
    assert config.loss_weights['ser'] == 1.0, "SER should always be 1.0 (main task)"
    assert config.loss_weights['asr'] == 0.1, "ASR weight should match alpha_asr"
    assert config.loss_weights['prosody'] == 0.1, "Prosody weight should match alpha_prosody"
    
    # Test 2: Alpha update
    config.update_alpha_weights(0.5, 0.3)
    print(f"\n✓ Alpha values updated to ASR=0.5, Prosody=0.3")
    print(f"  New loss weights: SER={config.loss_weights['ser']}, ASR={config.loss_weights['asr']}, Prosody={config.loss_weights['prosody']}")
    
    assert config.loss_weights['ser'] == 1.0, "SER should remain 1.0"
    assert config.loss_weights['asr'] == 0.5, "ASR weight should be updated"
    assert config.loss_weights['prosody'] == 0.3, "Prosody weight should be updated"
    
    # Test 3: Paper-style values (matching Table 4 from paper)
    paper_alphas = [0.0, 0.001, 0.01, 0.1, 1.0]
    for alpha in paper_alphas:
        config.update_alpha_weights(alpha, alpha)
        print(f"  Paper alpha {alpha}: SER=1.0, ASR={config.loss_weights['asr']}, Prosody={config.loss_weights['prosody']}")
    
    print("✅ Alpha configuration tests passed!")
    return True


def test_model_forward_with_alpha():
    """Test model forward pass with different alpha values"""
    print("\n" + "="*50)
    print("TESTING MODEL FORWARD WITH ALPHA")
    print("="*50)
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.blank_id = 0
        
        def get_vocab_size(self):
            return 1000
        
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 5]  # Mock tokens
        
        def decode(self, token_ids, skip_special_tokens=True):
            return "mock decoded text"
    
    tokenizer = MockTokenizer()
    
    # Test different alpha combinations following the paper
    alpha_combinations = [
        (0.0, 0.0),    # No auxiliary tasks
        (0.1, 0.0),    # Only ASR auxiliary
        (0.0, 0.1),    # Only Prosody auxiliary
        (0.1, 0.1),    # Both auxiliary tasks (paper's best)
        (1.0, 1.0),    # Strong auxiliary tasks
    ]
    
    for alpha_asr, alpha_prosody in alpha_combinations:
        print(f"\nTesting alpha_asr={alpha_asr}, alpha_prosody={alpha_prosody}")
        
        # Create config
        config = MTLConfig(
            backbone_name="whisper",
            alpha_asr=alpha_asr,
            alpha_prosody=alpha_prosody,
            vocab_size=1000
        )
        
        # Create model
        model = MTLModel(
            config=config,
            use_asr=True,
            use_prosody=True,
            use_ser=True,
            tokenizer=tokenizer
        )
        model.eval()
        
        # Create mock input
        batch_size = 2
        n_mels = 80
        time_steps = 100
        prosody_len = 20
        
        input_features = torch.randn(batch_size, n_mels, time_steps)
        asr_targets = torch.randint(0, 100, (batch_size, 20))
        asr_lengths = torch.tensor([20, 15])
        prosody_targets = torch.randint(0, 2, (batch_size, prosody_len)).float()
        emotion_targets = torch.randint(0, 9, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                asr_targets=asr_targets,
                asr_lengths=asr_lengths,
                prosody_targets=prosody_targets,
                emotion_targets=emotion_targets,
                return_loss=True
            )
        
        # Check outputs
        assert 'total_loss' in outputs, "Total loss should be present"
        assert 'emotion_logits' in outputs, "SER outputs should always be present"
        
        # Check loss components and weights
        total_loss = outputs['total_loss'].item()
        print(f"  Total loss: {total_loss:.4f}")
        
        if 'alpha_values' in outputs:
            alpha_vals = outputs['alpha_values']
            print(f"  Alpha values: ASR={alpha_vals['alpha_asr']}, Prosody={alpha_vals['alpha_prosody']}")
            assert alpha_vals['alpha_asr'] == alpha_asr, "Alpha ASR should match config"
            assert alpha_vals['alpha_prosody'] == alpha_prosody, "Alpha Prosody should match config"
            assert alpha_vals['main_task_weight'] == 1.0, "Main task weight should be 1.0"
        
        # Verify task-specific behavior
        if alpha_asr > 0:
            assert 'asr_logits' in outputs, "ASR outputs should be present when alpha_asr > 0"
            if 'asr_loss' in outputs:
                print(f"  ASR loss: {outputs['asr_loss'].item():.4f}")
        
        if alpha_prosody > 0:
            assert 'prosody_logits' in outputs, "Prosody outputs should be present when alpha_prosody > 0"
            if 'prosody_loss' in outputs:
                print(f"  Prosody loss: {outputs['prosody_loss'].item():.4f}")
        
        if 'emotion_loss' in outputs:
            print(f"  SER loss: {outputs['emotion_loss'].item():.4f}")
        
        print(f"  ✓ Forward pass successful")
    
    print("✅ Model forward tests passed!")
    return True


def test_loss_weighting_paper_style():
    """Test that loss weighting follows the paper's methodology"""
    print("\n" + "="*50)
    print("TESTING PAPER-STYLE LOSS WEIGHTING")
    print("="*50)
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.blank_id = 0
        
        def get_vocab_size(self):
            return 1000
        
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 5]
        
        def decode(self, token_ids, skip_special_tokens=True):
            return "mock decoded text"
    
    tokenizer = MockTokenizer()
    
    # Test the paper's key finding: alpha=0.1 is optimal
    config = MTLConfig(
        backbone_name="whisper",
        alpha_asr=0.1,
        alpha_prosody=0.1,
        vocab_size=1000
    )
    
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    )
    model.eval()
    
    # Create consistent mock input
    torch.manual_seed(42)
    batch_size = 1
    input_features = torch.randn(batch_size, 80, 100)
    asr_targets = torch.randint(0, 100, (batch_size, 10))
    asr_lengths = torch.tensor([10])
    prosody_targets = torch.ones(batch_size, 10)
    emotion_targets = torch.tensor([3])
    
    print("Testing loss computation with paper's methodology:")
    print("SER is main task (weight=1.0), ASR and Prosody are auxiliary (weighted by alpha)")
    
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            asr_targets=asr_targets,
            asr_lengths=asr_lengths,
            prosody_targets=prosody_targets,
            emotion_targets=emotion_targets,
            return_loss=True
        )
    
    # Verify loss computation follows paper's formula:
    # L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody
    
    ser_loss = outputs.get('emotion_loss', torch.tensor(0.0)).item()
    asr_loss = outputs.get('asr_loss', torch.tensor(0.0)).item()
    prosody_loss = outputs.get('prosody_loss', torch.tensor(0.0)).item()
    total_loss = outputs['total_loss'].item()
    
    expected_total = ser_loss + 0.1 * asr_loss + 0.1 * prosody_loss
    
    print(f"  SER loss (weight=1.0): {ser_loss:.4f}")
    print(f"  ASR loss (weight=0.1): {asr_loss:.4f}")
    print(f"  Prosody loss (weight=0.1): {prosody_loss:.4f}")
    print(f"  Expected total: {expected_total:.4f}")
    print(f"  Actual total: {total_loss:.4f}")
    print(f"  Difference: {abs(expected_total - total_loss):.6f}")
    
    # Allow small numerical differences
    assert abs(expected_total - total_loss) < 1e-5, f"Loss computation doesn't match paper's formula"
    
    print("✅ Paper-style loss weighting verified!")
    return True


def test_alpha_zero_cases():
    """Test edge cases when alpha values are zero (single task scenarios)"""
    print("\n" + "="*50)
    print("TESTING ALPHA=0 CASES (SINGLE TASK)")
    print("="*50)
    
    class MockTokenizer:
        def __init__(self):
            self.pad_id = 0
            self.blank_id = 0
        
        def get_vocab_size(self):
            return 1000
        
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 5]
        
        def decode(self, token_ids, skip_special_tokens=True):
            return "mock decoded text"
    
    tokenizer = MockTokenizer()
    
    # Test alpha=0 case (SER only, like paper's baseline)
    print("Testing alpha=0 case (SER only - paper's baseline)")
    
    config = MTLConfig(
        backbone_name="whisper",
        alpha_asr=0.0,
        alpha_prosody=0.0,
        vocab_size=1000
    )
    
    model = MTLModel(
        config=config,
        use_asr=True,
        use_prosody=True,
        use_ser=True,
        tokenizer=tokenizer
    )
    model.eval()
    
    # Mock input
    input_features = torch.randn(1, 80, 100)
    asr_targets = torch.randint(0, 100, (1, 10))
    asr_lengths = torch.tensor([10])
    prosody_targets = torch.ones(1, 10)
    emotion_targets = torch.tensor([3])
    
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            asr_targets=asr_targets,
            asr_lengths=asr_lengths,
            prosody_targets=prosody_targets,
            emotion_targets=emotion_targets,
            return_loss=True
        )
    
    # In alpha=0 case, total loss should equal SER loss only
    ser_loss = outputs.get('emotion_loss', torch.tensor(0.0)).item()
    total_loss = outputs['total_loss'].item()
    
    print(f"  SER loss: {ser_loss:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Difference: {abs(ser_loss - total_loss):.6f}")
    
    assert abs(ser_loss - total_loss) < 1e-5, "When alpha=0, total loss should equal SER loss"
    
    print("✅ Alpha=0 case verified!")
    return True


def run_comprehensive_test():
    """Run all tests to verify paper-style implementation"""
    print("🧪 COMPREHENSIVE ALPHA IMPLEMENTATION TEST")
    print("Following the paper: 'Speech Emotion Recognition with Multi-task Learning'")
    print("="*70)
    
    tests = [
        ("Alpha Configuration", test_alpha_config),
        ("Model Forward with Alpha", test_model_forward_with_alpha),
        ("Paper-Style Loss Weighting", test_loss_weighting_paper_style),
        ("Alpha=0 Edge Cases", test_alpha_zero_cases),
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {str(e)}")
            failed_tests += 1
    
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📊 Success Rate: {passed_tests/(passed_tests+failed_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Your implementation correctly follows the paper's alpha control methodology:")
        print(f"  • SER is the main task (weight = 1.0)")
        print(f"  • ASR and Prosody are auxiliary tasks (weighted by alpha)")
        print(f"  • Loss formula: L = L_SER + α_ASR * L_ASR + α_Prosody * L_Prosody")
        print(f"  • Alpha values control auxiliary task importance")
        print(f"  • Alpha=0.1 gives best performance (as per paper)")
        return True
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print(f"Please check the implementation and fix the issues.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)