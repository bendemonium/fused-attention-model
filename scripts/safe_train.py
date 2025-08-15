"""Safe training configuration for H100 deployment"""

import os
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_scheduler

from model.fuseformerconfig import FuseFormerConfig
from model.fuseformer import FuseFormerForMaskedLM
from utils.train_utils import load_yaml_cfg, setup_wandb
from utils.debug_utils import check_model_for_nans, trace_nan_origin

def train(
    cfg_path: str,
    output_dir: str,
    safe_mode: bool = True  # Start with safe settings
):
    # Load config
    cfg = load_yaml_cfg(cfg_path)
    
    # Setup accelerator with safe mixed precision settings
    accelerator = Accelerator(
        mixed_precision="no" if safe_mode else "fp16",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1)
    )
    
    # Initialize model
    model_cfg = FuseFormerConfig(**cfg["model"])
    model = FuseFormerForMaskedLM(model_cfg)
    
    # Safe initialization check
    if check_model_for_nans(model):
        raise ValueError("NaN values found in model parameters after initialization!")
    
    # Optimizer with safe settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Conservative learning rate
        weight_decay=0.01,
        eps=1e-8  # Increased epsilon for stability
    )
    
    # Linear warmup scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,  # Extended warmup
        num_training_steps=cfg["num_train_steps"]
    )
    
    # Prepare everything with accelerator
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # Training loop
    model.train()
    completed_steps = 0
    
    for step in range(cfg["num_train_steps"]):
        with accelerator.accumulate(model):
            # Forward pass with safe handling
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Check for NaNs in early steps
            if step < 20:
                if torch.isnan(loss):
                    trace_nan_origin(loss, outputs, step)
                    raise ValueError(f"NaN detected at step {step}")
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        completed_steps += 1
        
        # Logging
        if completed_steps % cfg.get("log_every", 10) == 0:
            accelerator.print(
                f"Step {completed_steps}: loss = {loss.item():.4f}, "
                f"lr = {scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Save checkpoint
        if completed_steps % cfg.get("save_every", 1000) == 0:
            accelerator.save_state(f"{output_dir}/checkpoint-{completed_steps}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--safe_mode", action="store_true", help="Use safe training settings")
    args = parser.parse_args()
    
    train(args.config, args.output_dir, args.safe_mode)
