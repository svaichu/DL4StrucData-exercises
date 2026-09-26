"""
Exp-1 Gate-1 -- Phase 1, JEPA-style training variant.

Replaces the plain-MSE "predict(z_predecessor) vs z_current" objective with a
JEPA-style (Joint-Embedding Predictive Architecture) setup:

    - Online encoder  mlp_k        (obs -> z), trained by gradient descent
    - Target encoder  mlp_k_ema    (obs -> z), EMA copy of the online encoder,
      never receives gradients directly (stop-gradient target, as in
      BYOL/JEPA/DINO-style self-distillation)
    - predict: z_online(pred) -> z_target(cur)   (one-step forward model,
      same linear-only capacity as the original exp-1 spec)

Loss = MSE(predict(mlp_k(obs[pred])), stopgrad(mlp_k_ema(obs[cur])))
       + collapse-prevention regularizer (VICReg-style variance + covariance
         terms on the online embeddings, applied per batch)

Why this design avoids "predict the mean" collapse:
  A plain online/target encoder pair *without* an explicit anti-collapse term
  can converge to a constant vector (z = c for all inputs) -- loss -> 0 trivially,
  since predict(c) can also learn to output c. JEPA/BYOL/DINO avoid this via
  either (a) different online/target *architectures* with a predictor head plus
  stop-gradient (BYOL demonstrates that with batchnorm this alone resists
  collapse) or (b) an explicit variance/covariance regularizer (VICReg). We
  don't have batchnorm here (small MLP, no batch statistics assumed), so we use
  the explicit route: a VICReg-style variance term (push per-dimension std over
  a batch above a floor) and covariance term (push off-diagonal covariance of z
  dimensions toward 0, discouraging informational collapse onto a low-rank
  subspace) on the ONLINE encoder's output. This is the standard, auditable fix
  and its own hyperparameters (var_coeff, cov_coeff) are exactly what the sweep
  varies.

Collapse instrumentation (logged every step, see health table in exp1-lds-phase1.md
plus additions here):
  - z_std_mean       : mean per-dimension std of z over the batch (VICReg "variance" proxy).
                        Near 0 => collapse.
  - z_effective_rank : effective rank of the batch covariance of z (exp(entropy of
                        normalized eigenvalues)). Near 1 => representation has
                        collapsed onto (nearly) one direction; near d_z => full rank,
                        healthy.
  - z_uniqueness     : mean pairwise cosine distance between z vectors in a batch
                        (near 0 => all embeddings point the same way => collapse).
  - predict_drift, W_id_dist : same as original (checks predict != identity).
  - online_target_mse: MSE(z_online, z_target) on the SAME token (sanity: they
                        should track each other reasonably as EMA decays, but not
                        be identical while target lags online).

Usage
-----
    python train_phase1_jepa.py
    python train_phase1_jepa.py --ema_decay 0.99 --var_coeff 25 --cov_coeff 1
    python train_phase1_jepa.py --wandb_project fdca-exp1-jepa

Outputs
-------
    checkpoints_jepa/phase1_mlp_k.pt        (online encoder -- this is what Phase 2 uses)
    checkpoints_jepa/phase1_mlp_k_ema.pt    (target encoder, for inspection)
    checkpoints_jepa/phase1_predict.pt
    checkpoints_jepa/phase1_tau.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

FDCA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FDCA_DIR)

from datasource import LinearDSDataSource
from model_mlp_k import MlpK
from model_predict import Predict


# ============================================================
# Collapse-diagnostic helpers
# ============================================================

@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Effective rank of the batch covariance of z: exp(entropy(normalized eigvals)).
    1.0 = fully collapsed onto one direction; d_z = perfectly isotropic / full rank."""
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    s = eigvals.sum()
    if s <= 1e-12:
        return 1.0
    p = eigvals / s
    p = p.clamp_min(1e-12)
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@torch.no_grad()
def mean_pairwise_cosine_distance(z: torch.Tensor, max_n: int = 256) -> float:
    """Mean pairwise cosine distance (1 - cos_sim) over a (sub-sampled) batch.
    ~0 => all vectors point the same direction (collapse); ~1 => decorrelated/orthogonal-ish."""
    if z.shape[0] > max_n:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_n]
        z = z[idx]
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.T
    n = sim.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    mean_cos = sim[off_diag_mask].mean().item()
    return 1.0 - mean_cos


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance term: hinge loss pushing per-dim std up to >= gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: push off-diagonal covariance entries toward 0."""
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / D


# ============================================================
# EMA update
# ============================================================

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.mul_(decay).add_(po.data, alpha=1 - decay)


# ============================================================
# JEPA Phase-1 loss + one epoch
# ============================================================

def compute_jepa_loss(
    obs: torch.Tensor,       # (B, T, obs_dim)
    pred_idx: torch.Tensor,  # (B, T) long, -1 = no predecessor
    mlp_k: MlpK,
    mlp_k_ema: MlpK,
    predict: Predict,
    var_coeff: float,
    cov_coeff: float,
    var_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, T, obs_dim = obs.shape

    z_online = mlp_k(obs.reshape(B * T, obs_dim)).reshape(B, T, -1)         # (B,T,d_z), grad
    with torch.no_grad():
        z_target = mlp_k_ema(obs.reshape(B * T, obs_dim)).reshape(B, T, -1)  # stop-grad target

    valid = pred_idx >= 0
    bv, tv = valid.nonzero(as_tuple=True)
    if bv.numel() == 0:
        zero = z_online.sum() * 0.0
        return zero, {"pred_mse": float("nan"), "var_loss": 0.0, "cov_loss": 0.0}

    z_pred_online = z_online[bv, pred_idx[bv, tv]]     # online embedding of predecessor
    z_cur_target  = z_target[bv, tv]                   # EMA-target embedding of current (stop-grad)

    pred_mse = F.mse_loss(predict(z_pred_online), z_cur_target)

    z_flat = z_online.reshape(B * T, -1)
    var_loss = vicreg_variance_loss(z_flat, gamma=var_gamma)
    cov_loss = vicreg_covariance_loss(z_flat)

    loss = pred_mse + var_coeff * var_loss + cov_coeff * cov_loss
    return loss, {
        "pred_mse": pred_mse.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
    }


def train_one_epoch_jepa(
    mlp_k: MlpK,
    mlp_k_ema: MlpK,
    predict: Predict,
    ds: LinearDSDataSource,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    steps_per_epoch: int,
    ema_decay: float,
    var_coeff: float,
    cov_coeff: float,
    var_gamma: float,
) -> dict[str, float]:
    mlp_k.train()
    predict.train()
    mlp_k_ema.eval()

    total_loss = total_pred_mse = total_var = total_cov = 0.0
    for _ in range(steps_per_epoch):
        obs, _, pred_idx = ds.sample()
        loss, parts = compute_jepa_loss(
            obs, pred_idx, mlp_k, mlp_k_ema, predict, var_coeff, cov_coeff, var_gamma
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(mlp_k.parameters()) + list(predict.parameters()), grad_clip
        )
        optimizer.step()
        ema_update(mlp_k_ema, mlp_k, ema_decay)

        total_loss     += loss.item()
        total_pred_mse += parts["pred_mse"]
        total_var      += parts["var_loss"]
        total_cov      += parts["cov_loss"]

    mlp_k.eval()
    predict.eval()

    with torch.no_grad():
        B, T, obs_dim = obs.shape
        z_flat        = mlp_k(obs.reshape(B * T, obs_dim))
        z_target_flat = mlp_k_ema(obs.reshape(B * T, obs_dim))

        z_var           = z_flat.var(dim=0).mean().item()
        z_std_mean      = z_flat.std(dim=0).mean().item()
        z_eff_rank      = effective_rank(z_flat)
        z_uniqueness    = mean_pairwise_cosine_distance(z_flat)
        predict_drift   = predict.drift(z_flat)
        W_id_dist       = predict.identity_distance()
        online_target_mse = F.mse_loss(z_flat, z_target_flat).item()

    return {
        "loss":               total_loss / steps_per_epoch,
        "pred_mse":           total_pred_mse / steps_per_epoch,
        "var_loss":           total_var / steps_per_epoch,
        "cov_loss":           total_cov / steps_per_epoch,
        "z_var":              z_var,
        "z_std_mean":         z_std_mean,
        "z_effective_rank":   z_eff_rank,
        "z_uniqueness":       z_uniqueness,
        "predict_drift":      predict_drift,
        "W_id_dist":          W_id_dist,
        "online_target_mse":  online_target_mse,
    }


# ============================================================
# Main
# ============================================================

def run_training(cfg: argparse.Namespace) -> dict:
    """Core training loop. Assumes wandb.init() has already been called by the
    caller (either the CLI entrypoint below, or a wandb sweep agent function).
    Returns the final summary dict so a sweep driver can also inspect it directly."""
    device = torch.device(cfg.device)

    ds = LinearDSDataSource(
        d_z=cfg.d_z, obs_dim=cfg.obs_dim, seq_len=cfg.seq_len,
        n_chains=cfg.n_chains, batch_size=cfg.batch_size,
        seed=cfg.seed, device=device,
    )

    print("\n=== Phase 1 (JEPA-style): train MlpK (online+EMA target) + Predict ===")
    mlp_k     = MlpK(cfg.obs_dim, cfg.d_z, cfg.hidden).to(device)
    mlp_k_ema = copy.deepcopy(mlp_k).to(device)
    for p in mlp_k_ema.parameters():
        p.requires_grad_(False)
    predict = Predict(cfg.d_z).to(device)
    tau     = nn.Parameter(torch.ones(1, device=device))

    p1_params = list(mlp_k.parameters()) + list(predict.parameters()) + [tau]
    p1_opt    = torch.optim.AdamW(p1_params, lr=cfg.lr, weight_decay=cfg.wd)
    p1_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        p1_opt, T_max=cfg.p1_epochs * cfg.steps_per_epoch
    )

    collapse_flag = False
    for epoch in range(1, cfg.p1_epochs + 1):
        m = train_one_epoch_jepa(
            mlp_k, mlp_k_ema, predict, ds, p1_opt, cfg.grad_clip, cfg.steps_per_epoch,
            cfg.ema_decay, cfg.var_coeff, cfg.cov_coeff, cfg.var_gamma,
        )
        p1_sched.step()

        is_collapsed = (
            m["z_std_mean"] < 0.05
            or m["z_effective_rank"] < 1.5
            or m["z_uniqueness"] < 0.02
        )
        collapse_flag = collapse_flag or is_collapsed

        wandb.log({
            "phase": 1, "epoch": epoch,
            "p1/loss": m["loss"], "p1/pred_mse": m["pred_mse"],
            "p1/var_loss": m["var_loss"], "p1/cov_loss": m["cov_loss"],
            "p1/z_var": m["z_var"], "p1/z_std_mean": m["z_std_mean"],
            "p1/z_effective_rank": m["z_effective_rank"],
            "p1/z_uniqueness": m["z_uniqueness"],
            "p1/predict_drift": m["predict_drift"], "p1/W_id_dist": m["W_id_dist"],
            "p1/online_target_mse": m["online_target_mse"],
            "p1/is_collapsed": int(is_collapsed),
        })

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  ep {epoch:4d}  loss={m['loss']:.4e}  pred_mse={m['pred_mse']:.3e}  "
                  f"std={m['z_std_mean']:.3f}  eff_rank={m['z_effective_rank']:.2f}  "
                  f"uniq={m['z_uniqueness']:.3f}  drift={m['predict_drift']:.3f}  "
                  f"W-I={m['W_id_dist']:.3f}"
                  + ("  [COLLAPSE WARNING]" if is_collapsed else ""))

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(mlp_k.state_dict(),     os.path.join(cfg.ckpt_dir, "phase1_mlp_k.pt"))
    torch.save(mlp_k_ema.state_dict(), os.path.join(cfg.ckpt_dir, "phase1_mlp_k_ema.pt"))
    torch.save(predict.state_dict(),   os.path.join(cfg.ckpt_dir, "phase1_predict.pt"))
    torch.save(tau,                    os.path.join(cfg.ckpt_dir, "phase1_tau.pt"))
    print(f"Phase 1 (JEPA) checkpoints -> {cfg.ckpt_dir}/")

    final_eff_rank = m["z_effective_rank"]
    final_std      = m["z_std_mean"]
    final_uniq     = m["z_uniqueness"]

    print(f"\n{'='*60}")
    print("Collapse verdict (Phase 1, JEPA-style)")
    print(f"  final z_std_mean       = {final_std:.4f}   (want >> 0.05)")
    print(f"  final z_effective_rank = {final_eff_rank:.3f} / {cfg.d_z}   (want >> 1.5, ideally near d_z)")
    print(f"  final z_uniqueness     = {final_uniq:.4f}   (want >> 0.02)")
    print(f"  any epoch flagged collapse: {collapse_flag}")
    if final_eff_rank < 1.5 or final_std < 0.05 or final_uniq < 0.02 or collapse_flag:
        verdict = "COLLAPSED"
    else:
        verdict = "NOT COLLAPSED"
    print(f"  VERDICT: representation is {verdict}")
    print(f"{'='*60}")

    wandb.summary["final_z_std_mean"]       = final_std
    wandb.summary["final_z_effective_rank"] = final_eff_rank
    wandb.summary["final_z_uniqueness"]     = final_uniq
    wandb.summary["any_epoch_collapsed"]    = collapse_flag
    wandb.summary["collapse_verdict"]       = verdict
    wandb.summary["final_pred_mse"]         = m["pred_mse"]
    wandb.summary["final_W_id_dist"]        = m["W_id_dist"]
    wandb.summary["final_predict_drift"]    = m["predict_drift"]

    return {
        "final_z_std_mean": final_std,
        "final_z_effective_rank": final_eff_rank,
        "final_z_uniqueness": final_uniq,
        "any_epoch_collapsed": collapse_flag,
        "collapse_verdict": verdict,
        "final_pred_mse": m["pred_mse"],
        "final_W_id_dist": m["W_id_dist"],
        "final_predict_drift": m["predict_drift"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp-1 Gate-1 Phase 1 -- JEPA-style variant")
    p.add_argument("--d_z",            type=int,   default=16)
    p.add_argument("--obs_dim",        type=int,   default=32)
    p.add_argument("--seq_len",        type=int,   default=64)
    p.add_argument("--n_chains",       type=int,   default=8)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--hidden",         type=int,   default=64)
    p.add_argument("--ema_decay",      type=float, default=0.996,
                    help="EMA decay for target encoder (higher = slower-moving target)")
    p.add_argument("--var_coeff",      type=float, default=25.0,
                    help="VICReg variance-term coefficient (collapse prevention)")
    p.add_argument("--cov_coeff",      type=float, default=1.0,
                    help="VICReg covariance-term coefficient (decorrelation)")
    p.add_argument("--var_gamma",      type=float, default=1.0,
                    help="Target per-dimension std floor for the variance term")
    p.add_argument("--p1_epochs",      type=int,   default=30)
    p.add_argument("--steps_per_epoch",type=int,   default=100)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--wd",             type=float, default=1e-4)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--ckpt_dir",       default="checkpoints_jepa")
    p.add_argument("--wandb_project",  default="fdca-exp1-jepa")
    p.add_argument("--wandb_run",      default=None)
    return p.parse_args(argv)


def main() -> None:
    cfg = parse_args()
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
    run_training(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
