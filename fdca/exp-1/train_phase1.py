"""
Exp-1 Gate-1 — Full experiment: Phase 1 then Phase 2 comparison.

Phase 1: train MlpK + Predict on the LDS displaced-predecessor task.
Phase 2: compare FDCAScorer (frozen Phase 1 weights) vs DotProductScorer.

Usage
-----
    python train_phase1.py                       # full run
    python train_phase1.py --epochs 20 --steps_per_epoch 200
    python train_phase1.py --wandb_project my-project

Outputs
-------
    checkpoints/phase1_mlp_k.pt
    checkpoints/phase1_predict.pt
    checkpoints/phase1_tau.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Resolve sibling imports (fdca/ is one level up from fdca/exp-1/)
FDCA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FDCA_DIR)

from datasource import LinearDSDataSource
from scorers import FDCAScorer
from fdca import FDCATransformerBlock
from model_mlp_k import MlpK
from model_predict import Predict


# ============================================================
# Phase 1 helpers
# ============================================================

def compute_loss(
    obs: torch.Tensor,       # (B, T, obs_dim)
    pred_idx: torch.Tensor,  # (B, T) long, -1 = no predecessor
    mlp_k: MlpK,
    predict: Predict,
) -> torch.Tensor:
    """MSE of predict(z_predecessor) vs z_current over all valid pairs."""
    B, T, obs_dim = obs.shape
    z = mlp_k(obs.reshape(B * T, obs_dim)).reshape(B, T, -1)  # (B, T, d_z)

    valid = pred_idx >= 0
    bv, tv = valid.nonzero(as_tuple=True)
    if bv.numel() == 0:
        return z.sum() * 0.0

    z_cur  = z[bv, tv]
    z_pre  = z[bv, pred_idx[bv, tv]]
    return F.mse_loss(predict(z_pre), z_cur.detach())


def train_one_epoch(
    mlp_k: MlpK,
    predict: Predict,
    ds: LinearDSDataSource,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    steps_per_epoch: int,
) -> dict[str, float]:
    """Run one epoch (steps_per_epoch gradient steps). Returns averaged metrics."""
    mlp_k.train()
    predict.train()

    total_loss = 0.0
    for _ in range(steps_per_epoch):
        obs, _, pred_idx = ds.sample()
        loss = compute_loss(obs, pred_idx, mlp_k, predict)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(mlp_k.parameters()) + list(predict.parameters()),
            grad_clip,
        )
        optimizer.step()
        total_loss += loss.item()

    mlp_k.eval()
    predict.eval()

    # Compute health metrics on last batch (no grad)
    with torch.no_grad():
        B, T, obs_dim = obs.shape
        z_flat = mlp_k(obs.reshape(B * T, obs_dim))
        z_var          = z_flat.var(dim=0).mean().item()
        predict_drift  = predict.drift(z_flat)
        W_id_dist      = predict.identity_distance()

        valid = pred_idx >= 0
        bv, tv = valid.nonzero(as_tuple=True)
        mse = F.mse_loss(
            predict(z_flat.reshape(B, T, -1)[bv, pred_idx[bv, tv]]),
            z_flat.reshape(B, T, -1)[bv, tv],
        ).item() if bv.numel() > 0 else float("nan")

    return {
        "loss":           total_loss / steps_per_epoch,
        "mse":            mse,
        "z_var":          z_var,
        "predict_drift":  predict_drift,
        "W_id_dist":      W_id_dist,
    }


# ============================================================
# Phase 2 helpers
# ============================================================

class Exp1Net(nn.Module):
    """embed → FDCATransformerBlock → head : obs → predicted targets"""

    def __init__(self, obs_dim: int, d_model: int, n_heads: int, scorer=None):
        super().__init__()
        self.embed = nn.Linear(obs_dim, d_model)
        self.block = FDCATransformerBlock(d_model, n_heads, obs_dim, scorer=scorer)
        self.head  = nn.Linear(d_model, obs_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embed(obs)
        x = self.block(x)
        return self.head(x)


def train_phase2_one_epoch(
    model: Exp1Net,
    ds: LinearDSDataSource,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    steps_per_epoch: int,
    frozen_params: set,
) -> float:
    """One epoch for Phase 2. Returns mean prediction MSE."""
    model.train()
    total = 0.0
    train_params = [p for p in model.parameters() if p not in frozen_params]

    for _ in range(steps_per_epoch):
        obs, targets, _ = ds.sample()
        loss = F.mse_loss(model(obs), targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(train_params, grad_clip)
        optimizer.step()
        total += loss.item()

    return total / steps_per_epoch


# ============================================================
# Main
# ============================================================

def run(cfg: argparse.Namespace) -> None:
    device = torch.device(cfg.device)

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run,
        config=vars(cfg),
    )

    ds = LinearDSDataSource(
        d_z=cfg.d_z, obs_dim=cfg.obs_dim, seq_len=cfg.seq_len,
        n_chains=cfg.n_chains, batch_size=cfg.batch_size,
        seed=cfg.seed, device=device,
    )

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    print("\n=== Phase 1: train MlpK + Predict ===")
    mlp_k   = MlpK(cfg.obs_dim, cfg.d_z, cfg.hidden).to(device)
    predict = Predict(cfg.d_z).to(device)
    tau     = nn.Parameter(torch.ones(1, device=device))

    p1_params = list(mlp_k.parameters()) + list(predict.parameters()) + [tau]
    p1_opt    = torch.optim.AdamW(p1_params, lr=cfg.lr, weight_decay=cfg.wd)
    p1_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        p1_opt, T_max=cfg.p1_epochs * cfg.steps_per_epoch
    )

    for epoch in range(1, cfg.p1_epochs + 1):
        m = train_one_epoch(mlp_k, predict, ds, p1_opt, cfg.grad_clip, cfg.steps_per_epoch)
        p1_sched.step()

        wandb.log({"phase": 1, "epoch": epoch,
                   "p1/loss": m["loss"], "p1/mse": m["mse"],
                   "p1/z_var": m["z_var"], "p1/predict_drift": m["predict_drift"],
                   "p1/W_id_dist": m["W_id_dist"]})

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  ep {epoch:4d}  mse={m['mse']:.3e}  "
                  f"drift={m['predict_drift']:.3f}  W-I={m['W_id_dist']:.3f}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(mlp_k.state_dict(),   os.path.join(cfg.ckpt_dir, "phase1_mlp_k.pt"))
    torch.save(predict.state_dict(), os.path.join(cfg.ckpt_dir, "phase1_predict.pt"))
    torch.save(tau,                  os.path.join(cfg.ckpt_dir, "phase1_tau.pt"))
    print(f"Phase 1 checkpoints → {cfg.ckpt_dir}/")

    # ------------------------------------------------------------------
    # Phase 2a — FDCA scorer (frozen Phase 1 weights)
    # ------------------------------------------------------------------
    print("\n=== Phase 2a: FDCAScorer (frozen) ===")
    scorer_fdca  = FDCAScorer(mlp_k, predict, tau)
    scorer_fdca.freeze()
    frozen = set(scorer_fdca.parameters())

    net_fdca = Exp1Net(cfg.obs_dim, cfg.d_model, cfg.n_heads, scorer=scorer_fdca).to(device)
    train_p  = [p for p in net_fdca.parameters() if p not in frozen]
    opt_fdca = torch.optim.AdamW(train_p, lr=cfg.lr, weight_decay=cfg.wd)
    sched_fdca = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_fdca, T_max=cfg.p2_epochs * cfg.steps_per_epoch
    )

    for epoch in range(1, cfg.p2_epochs + 1):
        mse = train_phase2_one_epoch(
            net_fdca, ds, opt_fdca, cfg.grad_clip, cfg.steps_per_epoch, frozen
        )
        sched_fdca.step()
        wandb.log({"phase": 2, "epoch": epoch, "p2/fdca_mse": mse})
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  ep {epoch:4d}  mse={mse:.4e}")

    final_fdca = mse

    # ------------------------------------------------------------------
    # Phase 2b — DotProduct baseline
    # ------------------------------------------------------------------
    print("\n=== Phase 2b: DotProductScorer (baseline) ===")
    net_dot  = Exp1Net(cfg.obs_dim, cfg.d_model, cfg.n_heads, scorer=None).to(device)
    opt_dot  = torch.optim.AdamW(net_dot.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched_dot = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_dot, T_max=cfg.p2_epochs * cfg.steps_per_epoch
    )

    for epoch in range(1, cfg.p2_epochs + 1):
        mse = train_phase2_one_epoch(
            net_dot, ds, opt_dot, cfg.grad_clip, cfg.steps_per_epoch, set()
        )
        sched_dot.step()
        wandb.log({"phase": 2, "epoch": epoch, "p2/dotprod_mse": mse})
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  ep {epoch:4d}  mse={mse:.4e}")

    final_dot = mse

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pct = 100 * (final_dot - final_fdca) / final_dot
    wandb.summary["final_fdca_mse"]   = final_fdca
    wandb.summary["final_dotprod_mse"] = final_dot
    wandb.summary["fdca_improvement_pct"] = pct

    print(f"\n{'='*50}")
    print(f"Final MSE  FDCA:    {final_fdca:.4e}")
    print(f"Final MSE  DotProd: {final_dot:.4e}")
    print(f"Improvement:        {pct:+.1f}%")
    if final_fdca < final_dot * 0.9:
        print("Gate 1: PASS — FDCA beats dot-product by >10%")
    elif abs(final_fdca - final_dot) < 0.05 * final_dot:
        print("Gate 1: AMBIGUOUS — check predict_drift for identity collapse")
    else:
        print("Gate 1: FAIL — FDCA does not beat dot-product")
    print(f"{'='*50}")

    wandb.finish()


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp-1 Gate-1 full experiment")
    # Data / LDS
    p.add_argument("--d_z",            type=int,   default=16)
    p.add_argument("--obs_dim",        type=int,   default=32)
    p.add_argument("--seq_len",        type=int,   default=64)
    p.add_argument("--n_chains",       type=int,   default=8)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--seed",           type=int,   default=42)
    # Phase 1 model
    p.add_argument("--hidden",         type=int,   default=64,
                   help="MlpK hidden dim (default: 2*obs_dim if omitted, else explicit)")
    # Phase 2 model
    p.add_argument("--d_model",        type=int,   default=64)
    p.add_argument("--n_heads",        type=int,   default=4)
    # Training
    p.add_argument("--p1_epochs",      type=int,   default=30)
    p.add_argument("--p2_epochs",      type=int,   default=30)
    p.add_argument("--steps_per_epoch",type=int,   default=100)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--wd",             type=float, default=1e-4)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    # Misc
    p.add_argument("--device",         default="cpu")
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--wandb_project",  default="fdca-exp1")
    p.add_argument("--wandb_run",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
