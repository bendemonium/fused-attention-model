"""Utilities for debugging NaN issues in the model."""

import torch
import pdb

def debug_early_nans(input_ids, scores, tokenizer, step):
    """Debug attention computation for NaN issues."""
    if step < 10:  # Only first 10 steps
        print(f"\nStep {step} Debug:")
        
        # Check input
        print(f"Input shape: {input_ids.shape}")
        print(f"Pad token count: {(input_ids == tokenizer.pad_token_id).sum()}")
        
        # Check raw scores (before any masking)
        print(f"Raw scores - min: {scores.min():.4f}, max: {scores.max():.4f}")
        print(f"Raw scores NaN count: {torch.isnan(scores).sum()}")
        
        # Check mask
        mask = (input_ids != tokenizer.pad_token_id)
        all_masked_queries = (~mask).all(dim=-1).sum()  # Queries with NO valid keys
        print(f"All-masked queries: {all_masked_queries}")
        
        return mask

def trace_nan_origin(loss, model_output, step):
    """Trace the origin of NaN values in model outputs."""
    if torch.isnan(loss) and step < 20:  # Focus on early NaNs
        print(f"\n🚨 EARLY NaN at step {step}")
        
        # Check each component
        if hasattr(model_output, 'logits'):
            logits_nan = torch.isnan(model_output.logits).sum()
            print(f"Logits NaN count: {logits_nan}")
            
            if logits_nan > 0:
                # Find which positions are NaN
                nan_positions = torch.isnan(model_output.logits)
                print(f"NaN positions shape: {nan_positions.sum(dim=-1)}")  # Per token
        
        # Stop for inspection
        pdb.set_trace()

def check_model_for_nans(model):
    """Check model parameters for NaN values after initialization."""
    has_nans = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter: {name}")
            has_nans = True
    return has_nans
