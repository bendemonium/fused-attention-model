# utils/train_utils.py

from __future__ import annotations
import os
import re
import io
import json
import math
import time
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import HfApi, create_repo, upload_folder, hf_hub_download


# =========================
# YAML / Config
# =========================

def load_yaml_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML must be a mapping/dict.")
    return cfg


# =========================
# Weights & Biases
# =========================

def setup_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    mode = cfg.get("wandb_mode", "online")  # "online" | "offline" | "disabled"
    if mode == "disabled":
        return None
    import wandb
    kwargs = {
        "project": cfg.get("wandb_project", "fuseformer"),
        "entity": cfg.get("wandb_entity", None),
        "name": cfg.get("wandb_run_name", None),
        "mode": "offline" if mode == "offline" else "online",
        "config": cfg,
        "resume": "allow",
    }
    wandb.init(**kwargs)
    return wandb


# =========================
# HF dataset repo -> pickled splits
# =========================

def _to_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    # HF Dataset-like
    if hasattr(obj, "column_names") and hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return [obj[i] for i in range(len(obj))]

    # DatasetDict -> take "train" if present, else first split
    if isinstance(obj, dict) and obj and any(hasattr(v, "column_names") for v in obj.values()):
        split = obj.get("train") or next(iter(obj.values()))
        return [split[i] for i in range(len(split))]

    # list-of-dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj

    # dict of lists/tuples
    if isinstance(obj, dict) and obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
        keys = list(obj.keys()); n = len(obj[keys[0]])
        return [{k: obj[k][i] for k in keys} for i in range(n)]

    # dict of tensors/arrays -> row dicts
    if isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values()):
        keys = list(obj.keys())
        n = int(next(iter(obj.values())).shape[0])
        out = []
        for i in range(n):
            d = {}
            for k, v in obj.items():
                vi = v[i]
                if isinstance(vi, np.ndarray):
                    if vi.dtype.kind in "iu":
                        vi = torch.from_numpy(vi).long()
                    else:
                        vi = torch.from_numpy(vi)
                elif isinstance(vi, torch.Tensor) and vi.dtype not in (torch.long, torch.int64):
                    vi = vi.long()
                d[k] = vi
            out.append(d)
        return out

    # tuple/list of tensors (input_ids[, attention_mask[, labels]])
    if isinstance(obj, (list, tuple)) and obj and all(hasattr(x, "shape") for x in obj):
        ids = obj[0]; am = obj[1] if len(obj) > 1 else None; lab = obj[2] if len(obj) > 2 else None
        n = ids.shape[0]
        out = []
        for i in range(n):
            d = {"input_ids": torch.as_tensor(ids[i]).long()}
            if am is not None:
                d["attention_mask"] = torch.as_tensor(am[i]).long()
            if lab is not None:
                d["labels"] = torch.as_tensor(lab[i]).long()
            out.append(d)
        return out

    # Generic indexable yielding dicts
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            first = obj[0]
            if isinstance(first, dict):
                return [obj[i] for i in range(len(obj))]
        except Exception:
            pass

    raise ValueError("Unsupported pickled dataset format after robust conversion.")


def hf_download_and_load_pkl(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        repo_type="dataset",
    )
    with open(local_path, "rb") as f:
        obj = pickle.load(f)
    data = _to_list_of_dicts(obj)
    if not data or "input_ids" not in data[0]:
        raise ValueError(f"{repo_id}/{filename} missing 'input_ids'")
    return data


# =========================
# Tokens -> Words counter
# =========================

@dataclass
class ProgressState:
    tokens_total: int = 0
    words_total: float = 0.0
    global_step: int = 0
    completed: List[str] = None
    last_branch: Optional[str] = None

    def to_dict(self):
        return {
            "tokens_total": int(self.tokens_total),
            "words_total": float(self.words_total),
            "global_step": int(self.global_step),
            "completed": list(self.completed or []),
            "last_branch": self.last_branch,
        }

    @staticmethod
    def from_file(path: Path) -> "ProgressState":
        if not path.exists():
            return ProgressState(tokens_total=0, words_total=0.0, global_step=0, completed=[])
        data = json.loads(path.read_text())
        return ProgressState(
            tokens_total=int(data.get("tokens_total", 0)),
            words_total=float(data.get("words_total", 0.0)),
            global_step=int(data.get("global_step", 0)),
            completed=list(data.get("completed", [])),
            last_branch=data.get("last_branch", None),
        )


def nonpad_token_count(attention_mask: torch.Tensor) -> int:
    # attention_mask is (B,T) 1 for real tokens, 0 for pad
    return int(attention_mask.sum().item())


# =========================
# Milestones
# =========================

def parse_word_tag(tag: str) -> int:
    """'1m' -> 1_000_000, '20m' -> 20_000_000, '100m' -> 100_000_000."""
    m = re.fullmatch(r"(\d+)\s*[mM]", tag.strip())
    if not m:
        raise ValueError(f"Invalid milestone tag: {tag}")
    return int(m.group(1)) * 1_000_000


DEFAULT_MILESTONES = [f"{i}m" for i in list(range(1, 11)) + list(range(20, 101, 10))]


class MilestoneManager:
    def __init__(self, cfg: Dict[str, Any], out_dir: Path):
        self.branch_prefix: str = cfg.get("hf_branch_prefix", "words-")
        tags = cfg.get("milestones_words", DEFAULT_MILESTONES)
        self.targets_words: List[int] = [parse_word_tag(t) for t in tags]
        self.out_dir = out_dir
        self.progress_path = out_dir / "progress.json"
        self.state = ProgressState.from_file(self.progress_path)

    def next_target(self) -> Optional[int]:
        done = set(self.state.completed or [])
        for w in self.targets_words:
            tag = f"{w//1_000_000:03d}m"
            if tag not in done and self.state.words_total < w:
                return w
        return None

    def update_counters(self, tokens_step_global: int):
        self.state.tokens_total += tokens_step_global
        # rough rule-of-thumb: 0.75 words/token for English-ish corpora
        self.state.words_total = 0.75 * self.state.tokens_total

    def mark_done(self, target_words: int):
        tag = f"{target_words//1_000_000:03d}m"
        if self.state.completed is None:
            self.state.completed = []
        if tag not in self.state.completed:
            self.state.completed.append(tag)

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(json.dumps(self.state.to_dict(), indent=2))


# =========================
# HF Hub utilities (MODEL REPO pushes)
# =========================

def ensure_repo_and_push(
    local_dir: Path,
    repo_id: str,
    branch: str,
    hf_token: Optional[str],
    create_ok: bool = True,
    private: bool = False,
):
    api = HfApi(token=hf_token)
    if create_ok:
        try:
            create_repo(repo_id, token=hf_token, private=private, exist_ok=True)
        except Exception:
            pass
    upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        path_in_repo="",
        commit_message=f"Checkpoint push @ {branch}",
        revision=branch,
        token=hf_token,
    )


def snapshot_modeling_files(project_root: Path, dest_dir: Path):
    """
    Copies the key modeling files into `dest_dir` for reproducibility.
    NOTE: directory is `model/` (singular) in this project.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_files = [
        "model/fuseformer.py",
        "model/fuseformerconfig.py",
        "model/euclidean.py",
        "model/hyperbolic.py",
        "model/hyperplanes.py",
        "model/fusion.py",
        "model/more_model_utils.py",
    ]
    for rel in model_files:
        src = project_root / rel
        if src.exists():
            shutil.copyfile(src, dest_dir / Path(rel).name)


def save_local_checkpoint(
    model,
    cfg: Dict[str, Any],
    cfg_path: Path,
    out_dir: Path,
    mixed_hf: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True, parents=True)
    shutil.copyfile(cfg_path, out_dir / "configs" / "train_config.yaml")
    # save weights
    if mixed_hf and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_dir), safe_serialization=True)
    else:
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)


# =========================
# Optimizer param splitter
# (no geoopt anchors in euclidean-param version)
# =========================

def split_params_euclid_vs_anchors(model) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    In the euclidean-parameterized variant we do not use geoopt.ManifoldParameter.
    Return all params as Euclidean; keep anchors list empty so training code
    can skip the Riemannian optimizer branch cleanly.
    """
    euclid: List[torch.nn.Parameter] = []
    anchors: List[torch.nn.Parameter] = []
    for p in model.parameters():
        # If you decide to tag anchor params, you could check p.requires_grad and a name
        # pattern here and append to `anchors`. For now: everything in `euclid`.
        euclid.append(p)
    return euclid, anchors


# =========================
# Pretty progress bar
# =========================

class ProgressMeter:
    def __init__(self, total_steps: int, desc: str = ""):
        self.bar = tqdm(total=total_steps, desc=desc, leave=True, dynamic_ncols=True)
        self._last_words = 0.0
        self._last_t = time.time()

    def update(self, step_inc: int, words_total: float, next_target: Optional[int], loss_val: Optional[float]):
        now = time.time()
        dt = max(1e-6, now - self._last_t)
        words_delta = max(0.0, words_total - self._last_words)
        it_per_s = step_inc / dt
        words_per_s = words_delta / dt

        self.bar.update(step_inc)
        words_m = words_total / 1_000_000.0
        nxt = f"{next_target/1_000_000:.0f}M" if next_target is not None else "final"
        postfix = {
            "words": f"{words_m:.2f}M",
            "next": nxt,
            "it/s": f"{it_per_s:.2f}",
            "w/s": f"{words_per_s:.0f}",
        }
        if next_target is not None and words_per_s > 0:
            remaining = max(0.0, next_target - words_total)
            eta_sec = remaining / words_per_s
            postfix["eta_next"] = f"{eta_sec/3600:.2f}h"
        if loss_val is not None:
            postfix["loss"] = f"{loss_val:.4f}"
        self.bar.set_postfix(postfix)

        self._last_words = words_total
        self._last_t = now

    def close(self):
        self.bar.close()


# =========================
# AMP preference (Hopper/Ampere)
# =========================

def prefer_bf16() -> bool:
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    # Ampere (8.x) or newer: prefer bf16
    return cap[0] >= 8