"""
Phase 1 training — self-supervised latent forward-dynamics pretraining.

Trains mlp_k (obs → z) and predict (z → z) with a BYOL-style stop-gradient
objective on consecutive observation pairs:

    L = mean_{t}  ‖ predict(mlp_k(obs_t)) − sg(mlp_k(obs_{t+1})) ‖²

After training, saves { mlp_k, predict, tau, obs_dim, mlp_k_d_z } as a
checkpoint ready to be loaded by train_phase2.py --phase1 <ckpt>.

Usage
-----
    python train_phase1.py                          # synthetic fallback
    python train_phase1.py --dataset droid          # OXE dataset
    python train_phase1.py --dataset droid --d_z 64
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFG:
    # --- data ---
    dataset_name: str = "droid"
    split_train: str = "train"
    split_val: str = "val[:10%]"
    obs_keys: List[str] = field(default_factory=lambda: ["state"])
    seq_len: int = 16
    control_frequency: float = 10.0

    # --- dims (inferred from first batch; set manually for synthetic fallback) ---
    obs_dim: int = 16
    action_dim: int = 7

    # --- model ---
    d_z: int = 32               # latent dimension for mlp_k / predict

    # --- training ---
    epochs: int = 20
    batch_size: int = 64
    steps_per_epoch: int = 200
    val_steps: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 42

    # --- paths ---
    run_name: str = "phase1"
    ckpt_dir: str = "checkpoints"
    log_every: int = 50


cfg = CFG()


# ---------------------------------------------------------------------------
# Phase 1 model
# ---------------------------------------------------------------------------

class Phase1Model(nn.Module):
    """mlp_k + predict + learnable temperature τ."""

    def __init__(self, obs_dim: int, d_z: int):
        super().__init__()
        self.mlp_k = nn.Sequential(
            nn.Linear(obs_dim, d_z * 2),
            nn.ReLU(),
            nn.Linear(d_z * 2, d_z),
        )
        self.predict = nn.Sequential(
            nn.Linear(d_z, d_z * 2),
            nn.ReLU(),
            nn.Linear(d_z * 2, d_z),
        )
        self.tau = nn.Parameter(torch.ones(1))

    def loss(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, obs_dim)
        Pairs: (obs_t, obs_{t+1}) for t in 0..T-2.
        L = mean ‖predict(mlp_k(obs_t)) − sg(mlp_k(obs_{t+1}))‖²
        """
        obs_t  = obs[:, :-1].reshape(-1, obs.shape[-1])   # (B*(T-1), obs_dim)
        obs_tp1 = obs[:, 1:].reshape(-1, obs.shape[-1])

        z_t   = self.mlp_k(obs_t)                         # (B*(T-1), d_z)
        z_tp1 = self.mlp_k(obs_tp1).detach()              # stop-gradient on target

        pred  = self.predict(z_t)                          # (B*(T-1), d_z)
        return F.mse_loss(pred, z_tp1)


# ---------------------------------------------------------------------------
# Data helpers  (same as train_phase2.py)
# ---------------------------------------------------------------------------

def make_delta_timestamps(cfg: CFG) -> Dict[str, List[float]]:
    dt = 1.0 / cfg.control_frequency
    offsets = [round(i * dt, 6) for i in range(cfg.seq_len)]
    return {
        **{f"observation/{k}": offsets for k in cfg.obs_keys},
        "action": offsets,
    }


def extract_obs_action(batch, cfg: CFG, device: torch.device):
    obs_parts = []
    for k in cfg.obs_keys:
        leaf = batch["observation", k]
        obs_parts.append(leaf.float().flatten(start_dim=2))
    obs = torch.cat(obs_parts, dim=-1).to(device)
    action = batch["action"].float().to(device)
    return obs, action


def make_synthetic_batch(cfg: CFG, device: torch.device):
    B, T = cfg.batch_size, cfg.seq_len
    deltas = torch.randn(B, T, cfg.obs_dim, device=device) * 0.1
    obs = deltas.cumsum(dim=1)
    action = torch.randn(B, T, cfg.action_dim, device=device) * 0.1
    return obs, action


def build_dataset(cfg: CFG, split: str):
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
    model: Phase1Model,
    optimizer: torch.optim.Optimizer,
    ds_train,
    cfg: CFG,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total = 0.0
    t0 = time.time()

    for step in range(cfg.steps_per_epoch):
        if ds_train is not None:
            batch = ds_train.sample()
            obs, _ = extract_obs_action(batch, cfg, device)
        else:
            obs, _ = make_synthetic_batch(cfg, device)

        loss = model.loss(obs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total += loss.item()

        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d} step {step+1:3d}/{cfg.steps_per_epoch}"
                  f"  loss={total/(step+1):.4f}  tau={model.tau.item():.4f}"
                  f"  {elapsed:.1f}s")

    return total / cfg.steps_per_epoch


@torch.no_grad()
def validate(
    model: Phase1Model,
    ds_val,
    cfg: CFG,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(cfg.val_steps):
        if ds_val is not None:
            batch = ds_val.sample()
            obs, _ = extract_obs_action(batch, cfg, device)
        else:
            obs, _ = make_synthetic_batch(cfg, device)
        total += model.loss(obs).item()
    return total / cfg.val_steps


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def latent_variance(model: Phase1Model, obs: torch.Tensor) -> float:
    """Mean per-dim variance of mlp_k(obs) across the batch — should stay > 0."""
    z = model.mlp_k(obs.reshape(-1, obs.shape[-1]))
    return z.var(dim=0).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.dataset:
        cfg.dataset_name = args.dataset
    if args.d_z:
        cfg.d_z = args.d_z

    ds_train = build_dataset(cfg, cfg.split_train)
    ds_val   = build_dataset(cfg, cfg.split_val)

    if ds_train is not None:
        infer_dims(ds_train, cfg, device)
    else:
        print(f"synthetic data  obs_dim={cfg.obs_dim}  action_dim={cfg.action_dim}")

    model = Phase1Model(cfg.obs_dim, cfg.d_z).to(device)
    print(f"trainable params: {count_trainable(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*60}\nepoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(model, optimizer, ds_train, cfg, device, epoch)
        val_loss   = validate(model, ds_val, cfg, device)
        scheduler.step()

        # collapse check on a quick synthetic batch
        obs_check, _ = make_synthetic_batch(cfg, device)
        var = latent_variance(model, obs_check)
        print(f"val loss={val_loss:.4f}  latent_var={var:.4f}  (collapse if ≈0)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "obs_dim":    cfg.obs_dim,
                    "mlp_k_d_z":  cfg.d_z,
                    "mlp_k":      model.mlp_k.state_dict(),
                    "predict":    model.predict.state_dict(),
                    "tau":        model.tau.data.clone(),
                    "val_loss":   val_loss,
                },
                ckpt_dir / f"{cfg.run_name}_best.pt",
            )
            print(f"  saved best checkpoint (val={best_val:.4f})")

    print(f"\n{'='*60}")
    print(f"FINAL  best val loss: {best_val:.4f}")
    print(f"checkpoint: {ckpt_dir}/{cfg.run_name}_best.pt")
    print(f"\nTo use in Phase 2:")
    print(f"  python train_phase2.py --phase1 {ckpt_dir}/{cfg.run_name}_best.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="OXE dataset name, e.g. 'droid' (default: synthetic fallback)")
    parser.add_argument("--d_z", type=int, default=None,
                        help="latent dimension for mlp_k / predict (default: 32)")
    args = parser.parse_args()
    main(args)
