"""
Phase 2 training script — FDCA vs dot-product transformer benchmark.

Trains two matched transformers side-by-side on next-step observation prediction:
  - fdca : FDCATransformerBlock with dot-product placeholder (swap scorer after Phase 1)
  - base : identical architecture, always dot-product (the baseline)

Data
----
Uses OXEDataset from the robotdataset package.  With delta_timestamps set to T
evenly-spaced offsets, each .sample() call returns TensorDict(batch_size=[B])
with "observation" and "action" leaves shaped (B, T, ...).

The Phase 2 task is next-step observation prediction (MSE):
    x = obs[:, :-1]   (B, T-1, obs_dim)  — context
    y = obs[:, 1:]    (B, T-1, obs_dim)  — targets

action (B, T, action_dim) is extracted and kept for future use (§6 — currently unused).

Phase 1 checkpoint
------------------
Pass --phase1 phase1.pt to install the frozen FDCAScorer.
Without it the FDCA model trains with the dot-product placeholder.

Usage
-----
    python train_phase2.py                               # synthetic fallback, dot-product placeholder
    python train_phase2.py --dataset droid               # OXE dataset
    python train_phase2.py --dataset droid --phase1 p1.pt
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdca import FDCATransformerBlock
from scorers import FDCAScorer
from train_phase1 import Phase1Model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFG:
    # --- data ---
    dataset_name: str = "droid"
    split_train: str = "train"
    split_val: str = "val[:10%]"    # TFDS-style split expression
    # observation sub-keys to concatenate into the flat obs vector
    # adjust to match the actual dataset's observation TensorDict keys
    obs_keys: List[str] = field(default_factory=lambda: ["state"])
    seq_len: int = 16               # T: number of timesteps per sample
    control_frequency: float = 10.0 # Hz — for delta_timestamps conversion

    # --- dims (inferred from first batch; set manually for synthetic fallback) ---
    obs_dim: int = 16
    action_dim: int = 7

    # --- model ---
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    causal: bool = True

    # --- training ---
    epochs: int = 20
    batch_size: int = 64
    steps_per_epoch: int = 200      # .sample() calls per epoch (train)
    val_steps: int = 40             # .sample() calls per validation pass
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 42

    # --- paths ---
    run_name: str = "phase2"
    ckpt_dir: str = "checkpoints"
    log_every: int = 50


cfg = CFG()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, cfg: CFG, scorer: Optional[nn.Module] = None):
        super().__init__()
        self.input_proj = nn.Linear(cfg.obs_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            FDCATransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                obs_dim=cfg.obs_dim,
                ffn_mult=cfg.ffn_mult,
                scorer=scorer,
                causal=cfg.causal,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, obs_dim)  →  (B, T, obs_dim)"""
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.input_proj(x) + self.pos_emb(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_out(h))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_delta_timestamps(cfg: CFG) -> Dict[str, List[float]]:
    """Build a delta_timestamps dict that samples T consecutive steps at cfg.control_frequency."""
    dt = 1.0 / cfg.control_frequency
    offsets = [round(i * dt, 6) for i in range(cfg.seq_len)]
    return {
        **{f"observation/{k}": offsets for k in cfg.obs_keys},
        "action": offsets,
    }


def extract_obs_action(batch, cfg: CFG, device: torch.device):
    """
    Pull obs_keys out of batch["observation"], concatenate into (B, T, obs_dim),
    and return (obs, action) both as float tensors on device.

    batch is a TensorDict returned by OXEDataset.sample().
    """
    obs_parts = []
    for k in cfg.obs_keys:
        leaf = batch["observation", k]          # (B, T, *feature_dims)
        obs_parts.append(leaf.float().flatten(start_dim=2))  # (B, T, d_k)
    obs = torch.cat(obs_parts, dim=-1).to(device)            # (B, T, obs_dim)
    action = batch["action"].float().to(device)              # (B, T, action_dim)
    return obs, action


def make_synthetic_batch(cfg: CFG, device: torch.device):
    """Random-walk synthetic batch when OXEDataset is unavailable."""
    B, T = cfg.batch_size, cfg.seq_len
    deltas = torch.randn(B, T, cfg.obs_dim, device=device) * 0.1
    obs = deltas.cumsum(dim=1)
    action = torch.randn(B, T, cfg.action_dim, device=device) * 0.1
    return obs, action


def build_dataset(cfg: CFG, split: str):
    """Instantiate OXEDataset; returns None if unavailable (synthetic fallback)."""
    try:
        from robotdataset import OXEDataset
        ds = OXEDataset(
            dataset_name=cfg.dataset_name,
            split=split,
            batch_size=cfg.batch_size,
            delta_timestamps=make_delta_timestamps(cfg),
            control_frequency=cfg.control_frequency,
            load_str_fields=False,
        )
        return ds
    except Exception as e:
        print(f"OXEDataset unavailable ({e}), using synthetic data")
        return None


def infer_dims(ds, cfg: CFG, device: torch.device) -> None:
    """Sample one batch to update cfg.obs_dim / cfg.action_dim from real data."""
    batch = ds.sample()
    obs, action = extract_obs_action(batch, cfg, device)
    cfg.obs_dim = obs.shape[-1]
    cfg.action_dim = action.shape[-1]
    print(f"inferred obs_dim={cfg.obs_dim}  action_dim={cfg.action_dim}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    ds_train,          # OXEDataset or None
    cfg: CFG,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    for m in models.values():
        m.train()

    totals = {tag: 0.0 for tag in models}
    t0 = time.time()

    for step in range(cfg.steps_per_epoch):
        if ds_train is not None:
            batch = ds_train.sample()
            obs, _ = extract_obs_action(batch, cfg, device)
        else:
            obs, _ = make_synthetic_batch(cfg, device)

        # next-step prediction: context = obs[:, :-1], target = obs[:, 1:]
        x, y = obs[:, :-1].contiguous(), obs[:, 1:].contiguous()

        for tag, model in models.items():
            preds = model(x)
            loss = F.mse_loss(preds, y)

            optimizers[tag].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizers[tag].step()
            totals[tag] += loss.item()

        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            losses_str = "  ".join(f"{t}={totals[t]/(step+1):.4f}" for t in models)
            print(f"  epoch {epoch:3d} step {step+1:3d}/{cfg.steps_per_epoch}  {losses_str}  {elapsed:.1f}s")

    return {tag: totals[tag] / cfg.steps_per_epoch for tag in models}


@torch.no_grad()
def validate(
    models: Dict[str, nn.Module],
    ds_val,
    cfg: CFG,
    device: torch.device,
) -> Dict[str, float]:
    for m in models.values():
        m.eval()

    totals = {tag: 0.0 for tag in models}
    for _ in range(cfg.val_steps):
        if ds_val is not None:
            batch = ds_val.sample()
            obs, _ = extract_obs_action(batch, cfg, device)
        else:
            obs, _ = make_synthetic_batch(cfg, device)

        x, y = obs[:, :-1].contiguous(), obs[:, 1:].contiguous()
        for tag, model in models.items():
            totals[tag] += F.mse_loss(model(x), y).item()

    return {tag: totals[tag] / cfg.val_steps for tag in models}


# ---------------------------------------------------------------------------
# Phase 1 scorer loading
# ---------------------------------------------------------------------------

def load_fdca_scorer(ckpt_path: str, cfg: CFG, device: torch.device) -> FDCAScorer:
    """Load mlp_k, predict, tau from a Phase 1 checkpoint and return a frozen scorer."""
    ckpt = torch.load(ckpt_path, map_location=device)

    assert ckpt["obs_dim"] == cfg.obs_dim, (
        f"Phase 1 obs_dim={ckpt['obs_dim']} != cfg.obs_dim={cfg.obs_dim}. "
        "Run infer_dims() before loading the scorer."
    )

    phase1 = Phase1Model(ckpt["obs_dim"], ckpt["mlp_k_d_z"])
    phase1.mlp_k.load_state_dict(ckpt["mlp_k"])
    phase1.predict.load_state_dict(ckpt["predict"])
    phase1.tau.data.copy_(ckpt["tau"])

    scorer = FDCAScorer(phase1.mlp_k, phase1.predict, phase1.tau)
    scorer.freeze()
    return scorer.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.dataset:
        cfg.dataset_name = args.dataset

    # --- data ---
    ds_train = build_dataset(cfg, cfg.split_train)
    ds_val   = build_dataset(cfg, cfg.split_val)

    if ds_train is not None:
        infer_dims(ds_train, cfg, device)   # sets cfg.obs_dim / cfg.action_dim
    else:
        print(f"synthetic data  obs_dim={cfg.obs_dim}  action_dim={cfg.action_dim}")

    # --- scorer ---
    fdca_scorer = None
    if args.phase1:
        print(f"loading Phase 1 checkpoint: {args.phase1}")
        fdca_scorer = load_fdca_scorer(args.phase1, cfg, device)
        print("FDCAScorer loaded and frozen")
    else:
        print("no Phase 1 checkpoint — FDCA model uses dot-product placeholder")

    # --- models ---
    model_fdca = Transformer(cfg, scorer=fdca_scorer).to(device)
    model_base = Transformer(cfg, scorer=None).to(device)
    models = {"fdca": model_fdca, "base": model_base}

    print(f"FDCA  trainable params: {count_trainable(model_fdca):,}")
    print(f"base  trainable params: {count_trainable(model_base):,}")

    # --- optimizers & schedulers ---
    optimizers = {
        tag: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for tag, m in models.items()
    }
    schedulers = {
        tag: torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
        for tag, opt in optimizers.items()
    }

    # --- training loop ---
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"fdca": [], "base": []}
    best = {"fdca": float("inf"), "base": float("inf")}

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*60}\nepoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(models, optimizers, ds_train, cfg, device, epoch)
        val_loss   = validate(models, ds_val, cfg, device)

        for tag in models:
            schedulers[tag].step()
            history[tag].append({"epoch": epoch, "train_loss": train_loss[tag], "val_loss": val_loss[tag]})
            if val_loss[tag] < best[tag]:
                best[tag] = val_loss[tag]
                torch.save(
                    {"epoch": epoch, "model": models[tag].state_dict(), "val_loss": val_loss[tag]},
                    ckpt_dir / f"{cfg.run_name}_{tag}_best.pt",
                )

        print(
            f"val   fdca={val_loss['fdca']:.4f}   base={val_loss['base']:.4f}"
            f"   (best fdca={best['fdca']:.4f}  base={best['base']:.4f})"
        )

    print(f"\n{'='*60}")
    print(f"FINAL  fdca best val loss: {best['fdca']:.4f}")
    print(f"       base best val loss: {best['base']:.4f}")

    torch.save(history, ckpt_dir / f"{cfg.run_name}_history.pt")
    print(f"saved to {ckpt_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="OXE dataset name, e.g. 'droid' (default: synthetic fallback)")
    parser.add_argument("--phase1", type=str, default=None,
                        help="path to Phase 1 checkpoint (.pt)")
    args = parser.parse_args()
    main(args)
