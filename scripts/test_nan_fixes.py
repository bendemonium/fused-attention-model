"""Test script to verify NaN fixes before H100 deployment."""

import torch
import math
from tqdm.auto import tqdm
from model.fuseformer import FuseFormerConfig, FuseFormerForMaskedLM
from utils.debug_utils import check_model_for_nans, debug_early_nans, trace_nan_origin
from utils.train_utils import safe_forward_step

def setup_device():
    """Setup compute device for testing."""
    if torch.backends.mps.is_available():
        # Use Metal Performance Shaders (MPS) on Mac
        device = torch.device("mps")
        print("Using MPS (Metal) device for testing")
    else:
        device = torch.device("cpu")
        print("Using CPU device for testing")
    return device

def test_model_initialization(device):
    """Test model initialization for NaNs."""
    print("\nTesting model initialization...")
    config = FuseFormerConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        num_heads=4,
        attention_dropout=0.1,
        hidden_dropout=0.1
    )
    
    # Initialize model
    model = FuseFormerForMaskedLM(config)
    model = model.to(device)
    model.train()  # Set to training mode
    
    # Check for NaNs in initialization
    has_nans = check_model_for_nans(model)
    assert not has_nans, "Found NaNs in model parameters after initialization!"
    print("✓ Model initialization passed (no NaNs)")
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model size: {param_count:,} parameters")
    
    return model

def test_forward_pass_no_padding():
    """Test forward pass with no padding."""
    print("\nTesting forward pass without padding...")
    model = test_model_initialization()
    model.eval()  # Set to eval mode for testing
    
    # Create a simple batch without padding
    input_ids = torch.randint(0, 1000, (1, 5))  # batch_size=1, seq_len=5
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 1000, (1, 5))
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    with torch.no_grad():
        outputs = safe_forward_step(model, batch, step=0, amp_enabled=False)
    
    assert not torch.isnan(outputs.loss), "Found NaNs in loss (no padding case)!"
    print("✓ Forward pass without padding passed (no NaNs)")

def test_forward_pass_with_padding(model, device):
    """Test forward pass with heavy padding to stress test attention masking."""
    print("\nTesting forward pass with heavy padding...")
    model.train()  # Test in training mode
    
    # Create batches with varying levels of padding
    batch_size = 4
    max_seq_len = 128
    padding_ratios = [0.25, 0.5, 0.75, 0.9]  # Test different padding amounts
    
    for padding_ratio in padding_ratios:
        print(f"\nTesting with {padding_ratio*100}% padding...")
        
        # Create sequence lengths with specified padding
        seq_lengths = [int(max_seq_len * (1 - padding_ratio)) for _ in range(batch_size)]
        max_len = max(seq_lengths)
        
        # Create input tensors
        input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
        attention_mask = torch.zeros((batch_size, max_seq_len), device=device)
        labels = torch.full((batch_size, max_seq_len), -100, device=device)
        
        # Fill in actual sequence data and masks
        for i, length in enumerate(seq_lengths):
            attention_mask[i, :length] = 1
            labels[i, :length] = input_ids[i, :length]
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Test with and without gradient computation
        for requires_grad in [False, True]:
            with torch.set_grad_enabled(requires_grad):
                outputs = model(**batch)
                
                # Check for NaNs in forward pass
                assert not torch.isnan(outputs["loss"]), f"Found NaNs in loss with {padding_ratio*100}% padding!"
                if requires_grad:
                    # Backprop
                    outputs["loss"].backward()
                    
                    # Check for NaNs in gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            assert not torch.isnan(param.grad).any(), f"Found NaNs in gradients of {name}!"
        
        print(f"✓ Forward/backward pass with {padding_ratio*100}% padding passed (no NaNs)")
    
    print("\n✓ All padding tests passed!")

if __name__ == "__main__":
    print("Running comprehensive NaN fix verification tests...")
    try:
        # Setup device
        device = setup_device()
        
        # Run tests
        print("\n1. Testing model initialization and basic ops")
        model = test_model_initialization(device)
        
        print("\n2. Testing attention masking and padding handling")
        test_forward_pass_with_padding(model, device)
        
        print("\n✅ All tests passed! The model should be safe for H100 deployment.")
        print("\nRecommendations:")
        print("1. Start with small batch sizes on H100")
        print("2. Monitor first 100 steps closely")
        print("3. Use gradient clipping (max_grad_norm=1.0)")
        print("4. Consider starting with mixed precision disabled")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("Fix required before H100 deployment!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise
