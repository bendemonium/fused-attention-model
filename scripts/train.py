from __future__ import annotations
import os
import math
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler

import geoopt

# Your modules
from models.fuseformer_config import FuseFormerConfig
from models.fuseformer import FuseFormerForMaskedLM
from models.euclidean import EuclideanEncoder
from models.hyperbolic import HyperbolicEncoder

# Utils
from utils.train_utils import (
    load_yaml_cfg, setup_wandb,
    hf_download_and_load_pkl, nonpad_token_count, MilestoneManager,
    ensure_repo_and_push, snapshot_modeling_files, save_local_checkpoint,
    split_params_euclid_vs_anchors, ProgressMeter, prefer_bf16
)

logger = get_logger(__name__)


# -----------------------------
# Dataset wrappers
# -----------------------------

class TokenizedListDataset(Dataset):
    def __init__(self, examples: List[Dict[str, torch.Tensor | List[int]]], pad_token_id: int):
        self.data = examples
        self.pad_id = pad_token_id
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def pad_to_max_len(batch: List[Dict[str, torch.Tensor | List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    def to_tensor(x):
        if isinstance(x, torch.Tensor): return x
        return torch.tensor(x, dtype=torch.long)

    max_len = max(len(item["input_ids"]) for item in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = None
    has_labels = all(("labels" in item) for item in batch)
    if has_labels:
        labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        ids = to_tensor(item["input_ids"]); L = ids.shape[0]
        input_ids[i, :L] = ids
        if "attention_mask" in item:
            am = to_tensor(item["attention_mask"])
            attn_mask[i, :L] = am
        else:
            attn_mask[i, :L] = (ids != pad_id).long()
        if has_labels:
            lab = to_tensor(item["labels"])
            labels[i, :min(L, lab.shape[0])] = lab[:min(L, lab.shape[0])]

    out = {"input_ids": input_ids, "attention_mask": attn_mask}
    if has_labels:
        out["labels"] = labels
    return out


def apply_mlm_mask(batch: Dict[str, torch.Tensor], vocab_size: int, pad_id: int, mask_id: int, mlm_prob: float) -> Dict[str, torch.Tensor]:
    if "labels" in batch:
        return batch
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = input_ids.clone()
    candidates = attention_mask.bool()
    prob = torch.full_like(input_ids, mlm_prob, dtype=torch.float32)
    masked = (torch.bernoulli(prob).bool() & candidates)
    labels[~masked] = -100
    replace = torch.bernoulli(torch.full_like(input_ids, 0.8, dtype=torch.float32)).bool() & masked
    input_ids[replace] = mask_id
    rnd = torch.bernoulli(torch.full_like(input_ids, 0.5, dtype=torch.float32)).bool() & masked & ~replace
    rand_tok = torch.randint(0, vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
    input_ids[rnd] = rand_tok[rnd]
    batch["input_ids"] = input_ids
    batch["labels"] = labels
    return batch


def collate_builder(pad_id: int, vocab_size: int, mask_id: int, mlm_prob: float):
    def _fn(batch: List[Dict[str, torch.Tensor | List[int]]]) -> Dict[str, torch.Tensor]:
        out = pad_to_max_len(batch, pad_id)
        return apply_mlm_mask(out, vocab_size=vocab_size, pad_id=pad_id, mask_id=mask_id, mlm_prob=mlm_prob)
    return _fn


# -----------------------------
# Wrappers (reuse your modules)
# -----------------------------

class EuclidForMLM(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.encoder = EuclideanEncoder(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            n_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.1),
        )
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encoder(input_ids, attention_mask)
        logits = hidden @ self.encoder.embed_tokens.weight.t()
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return out


class HyperbolicForMLM(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.encoder = HyperbolicEncoder(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            spatial_dim=cfg.get("lorentz_spatial_dim", cfg["d_model"] // cfg["num_heads"]),
            d_hidden=cfg["d_ff"],
            tau=cfg.get("lorentz_tau", 1.0),
            dropout=cfg.get("dropout", 0.1),
            attn_dropout=cfg.get("attn_dropout", 0.0),
            rope_max_seq_len=cfg.get("rope_max_seq_len", 4096),
            use_rope=True,
            ln_eps=cfg.get("layer_norm_eps", 1e-5),
            karcher_steps=cfg.get("karcher_steps", 1),
        )
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden, _attn, _anchors = self.encoder(input_ids, attention_mask, return_attn=False)
        logits = hidden @ self.encoder.token_embedding.weight.t()
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return out


# -----------------------------
# Training
# -----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mixed","euclid","native"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_cfg(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    if is_main:
        logger.info(accelerator.state)

    # WandB (main process only)
    wandb = setup_wandb(cfg) if is_main else None

    # ----- Load pickled, already-tokenized datasets from HF DATASET REPO -----
    dataset_repo = cfg["dataset_repo"]
    revision = cfg.get("dataset_revision", None)
    hf_token_data = cfg.get("dataset_hf_token", os.environ.get("HF_TOKEN", None))

    train_examples = hf_download_and_load_pkl(dataset_repo, cfg["train_pkl"], revision=revision, token=hf_token_data)
    dev_examples = hf_download_and_load_pkl(dataset_repo, cfg["dev_pkl"], revision=revision, token=hf_token_data)
    if is_main:
        logger.info(f"Loaded from HF dataset repo '{dataset_repo}': "
                    f"train={len(train_examples)} rows, dev={len(dev_examples)} rows")

    pad_id = int(cfg.get("pad_token_id", 0))
    mask_id = int(cfg.get("mask_token_id", 103))
    vocab_size = int(cfg["vocab_size"])
    mlm_prob = float(cfg.get("mlm_prob", 0.15))

    train_ds = TokenizedListDataset(train_examples, pad_token_id=pad_id)
    dev_ds = TokenizedListDataset(dev_examples, pad_token_id=pad_id)
    collate_fn = collate_builder(pad_id, vocab_size, mask_id, mlm_prob)

    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=True, num_workers=2, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 32))), shuffle=False, num_workers=2, collate_fn=collate_fn)

    # ----- Model -----
    if args.model == "mixed":
        hf_config = FuseFormerConfig(**cfg)
        model = FuseFormerForMaskedLM(hf_config)
        mixed_hf = True
    elif args.model == "euclid":
        model = EuclidForMLM(cfg); mixed_hf = False
    else:
        model = HyperbolicForMLM(cfg); mixed_hf = False

    # ----- Optimizers -----
    euclid_params, anchor_params = split_params_euclid_vs_anchors(model)
    optim_euclid = torch.optim.AdamW(
        euclid_params,
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        betas=tuple(cfg.get("adam_betas", (0.9, 0.999))),
        eps=float(cfg.get("adam_eps", 1e-8)),
    )
    optim_anchor = geoopt.optim.RiemannianAdam(anchor_params, lr=float(cfg.get("lr_anchors", cfg.get("lr", 3e-4)))) if anchor_params else None

    # ----- Scheduler -----
    epochs = int(cfg.get("epochs", 3))
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes)
    total_training_steps = epochs * num_update_steps_per_epoch
    scheduler = get_scheduler(
        cfg.get("lr_scheduler", "linear"),
        optimizer=optim_euclid,
        num_warmup_steps=int(cfg.get("warmup_steps", 1000)),
        num_training_steps=total_training_steps,
    )

    # ----- Prepare with accelerate -----
    if optim_anchor is not None:
        model, optim_euclid, optim_anchor, train_loader, dev_loader, scheduler = accelerator.prepare(
            model, optim_euclid, optim_anchor, train_loader, dev_loader, scheduler
        )
    else:
        model, optim_euclid, train_loader, dev_loader, scheduler = accelerator.prepare(
            model, optim_euclid, train_loader, dev_loader, scheduler
        )

    # ----- Milestones, bf16 preference -----
    milestones = MilestoneManager(cfg, output_dir)
    pbar = ProgressMeter(total_steps=total_training_steps, desc=f"{args.model}") if is_main else None
    use_bf16 = cfg.get("amp_dtype", "bf16" if prefer_bf16() else "fp16").lower() == "bf16"

    push_enabled = bool(cfg.get("push_every_milestone", True)) and "hf_repo" in cfg

    # ----- Training -----
    best_dev = float("inf")
    for epoch in range(epochs):
        model.train()
        if is_main:
            logger.info(f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=True):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))

            optim_euclid.step()
            scheduler.step()
            optim_euclid.zero_grad(set_to_none=True)
            if optim_anchor is not None:
                optim_anchor.step()
                optim_anchor.zero_grad(set_to_none=True)

            # Count words (ignoring pads)
            tokens_step = nonpad_token_count(batch["attention_mask"])
            tokens_step = accelerator.gather_for_metrics(torch.tensor([tokens_step], device=loss.device)).sum().item()
            milestones.update_counters(int(tokens_step))
            milestones.state.global_step += 1

            if is_main:
                next_target = milestones.next_target()
                pbar.update(1, milestones.state.words_total, next_target, float(loss.item()))
                # W&B logging
                if cfg.get("wandb_mode", "online") != "disabled" and (milestones.state.global_step % int(cfg.get("wandb_log_every", 10)) == 0):
                    import wandb
                    wandb.log({
                        "train/loss": float(loss.item()),
                        "train/step": milestones.state.global_step,
                        "train/tokens_total": milestones.state.tokens_total,
                        "train/words_total": milestones.state.words_total,
                        "lr": scheduler.get_last_lr()[0],
                    }, step=milestones.state.global_step)

                # Milestone checkpoint & push to MODEL REPO
                if push_enabled and (next_target is not None) and (milestones.state.words_total >= next_target):
                    tag = f"{next_target//1_000_000:03d}m"
                    branch = f"{cfg.get('hf_branch_prefix','words-')}{tag}"
                    ckpt_dir = output_dir / f"checkpoint-{tag}"
                    milestones.save()
                    save_local_checkpoint(accelerator.unwrap_model(model), cfg, Path(args.config), ckpt_dir, mixed_hf=(args.model=="mixed"))
                    snapshot_modeling_files(Path("."), ckpt_dir / "models")
                    # optional tokenizer files
                    tok_dir = cfg.get("tokenizer_dir", None)
                    if tok_dir:
                        dst = ckpt_dir / "tokenizer"
                        dst.mkdir(parents=True, exist_ok=True)
                        for p in Path(tok_dir).glob("*"):
                            if p.is_file():
                                shutil.copyfile(p, dst / p.name)
                    try:
                        ensure_repo_and_push(
                            local_dir=ckpt_dir,
                            repo_id=cfg["hf_repo"],  # MODEL REPO
                            branch=branch,
                            hf_token=cfg.get("hf_token", os.environ.get("HF_TOKEN")),
                            create_ok=bool(cfg.get("hf_create_repo", True)),
                            private=bool(cfg.get("hf_private", False)),
                        )
                        milestones.mark_done(next_target)
                        milestones.state.last_branch = branch
                        milestones.save()
                        print(f"\n✅ Pushed checkpoint @ {tag} words → branch {branch}\n")
                        if cfg.get("wandb_mode", "online") != "disabled":
                            import wandb
                            wandb.log({"milestone/words": next_target, "milestone/branch": branch}, step=milestones.state.global_step)
                    except Exception as e:
                        print(f"\n⚠️  Push failed for {branch}: {e}\n")

        # ----- Eval -----
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in dev_loader:
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                n = (batch["labels"] != -100).sum().item()
                total_loss += accelerator.gather_for_metrics(loss.detach() * n).sum().item()
                total_tokens += accelerator.gather_for_metrics(torch.tensor([n], device=loss.device)).sum().item()
        dev_loss = total_loss / max(1, total_tokens)
        if is_main:
            logger.info(f"Eval loss/token: {dev_loss:.4f}")
            if cfg.get("wandb_mode", "online") != "disabled":
                import wandb
                wandb.log({"eval/loss": dev_loss}, step=milestones.state.global_step)
            # save "best" locally
            milestones.save()
            best_dir = output_dir / "best"
            save_local_checkpoint(accelerator.unwrap_model(model), cfg, Path(args.config), best_dir, mixed_hf=(args.model=="mixed"))

    # ----- Final push to main (MODEL REPO) -----
    if accelerator.is_main_process and bool(cfg.get("push_every_milestone", True)) and "hf_repo" in cfg:
        final_dir = output_dir / "final-main"
        milestones.save()
        save_local_checkpoint(accelerator.unwrap_model(model), cfg, Path(args.config), final_dir, mixed_hf=(args.model=="mixed"))
        snapshot_modeling_files(Path("."), final_dir / "models")
        try:
            ensure_repo_and_push(
                local_dir=final_dir,
                repo_id=cfg["hf_repo"],
                branch="main",
                hf_token=cfg.get("hf_token", os.environ.get("HF_TOKEN")),
                create_ok=bool(cfg.get("hf_create_repo", True)),
                private=bool(cfg.get("hf_private", False)),
            )
            print("\n🏁 Final checkpoint pushed to main.\n")
        except Exception as e:
            print(f"\n⚠️  Final push failed: {e}\n")

    if accelerator.is_main_process and 'pbar' in locals() and pbar is not None:
        pbar.close()
    if accelerator.is_main_process and cfg.get("wandb_mode", "online") != "disabled":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    from accelerate import Accelerator
    main()