import torch
import torch.nn as nn


class DotProductScorer(nn.Module):
    """Placeholder: standard scaled dot-product score.

    score(i, j) = (Wq·h_i) · (Wk·h_j) / sqrt(d_head)

    Note: q and k here are already projected to d_head by the caller.
    """

    def forward(
        self,
        q: torch.Tensor,  # (B, n_heads, T_q, d_head)
        k: torch.Tensor,  # (B, n_heads, T_k, d_head)
    ) -> torch.Tensor:    # (B, n_heads, T_q, T_k)
        d_head = q.size(-1)
        return torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)


class FDCAScorer(nn.Module):
    """Frozen FDCA score: −‖predict(mlp_k(o_j)) − mlp_k(o_i)‖² / τ

    o_i and o_j are pseudo-observations projected from the residual stream
    by Wq and Wk in FDCALayer (they live in obs_dim space, NOT d_head).

    Call freeze() after Phase 1 to lock all parameters.
    """

    def __init__(self, mlp_k: nn.Module, predict: nn.Module, tau: nn.Parameter):
        super().__init__()
        self.mlp_k = mlp_k
        self.predict = predict
        self.tau = tau

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        o_q: torch.Tensor,  # (B, n_heads, T_q, obs_dim) — query pseudo-obs
        o_k: torch.Tensor,  # (B, n_heads, T_k, obs_dim) — key pseudo-obs
    ) -> torch.Tensor:      # (B, n_heads, T_q, T_k)
        B, H, T_q, obs_dim = o_q.shape
        _, _, T_k, _ = o_k.shape

        # Flatten batch and head dims for the shared MLP
        o_q_flat = o_q.reshape(B * H * T_q, obs_dim)
        o_k_flat = o_k.reshape(B * H * T_k, obs_dim)

        z_q = self.mlp_k(o_q_flat).reshape(B, H, T_q, -1)  # (B, H, T_q, d_z)
        z_k = self.mlp_k(o_k_flat).reshape(B, H, T_k, -1)  # (B, H, T_k, d_z)

        # Forward roll on keys only
        d_z = z_k.size(-1)
        z_k_rolled = self.predict(z_k.reshape(B * H * T_k, d_z))
        z_k_rolled = z_k_rolled.reshape(B, H, T_k, d_z)      # (B, H, T_k, d_z)

        # L2 distance: expand to (B, H, T_q, T_k)
        # z_q: (B, H, T_q, 1, d_z), z_k_rolled: (B, H, 1, T_k, d_z)
        diff = z_q.unsqueeze(3) - z_k_rolled.unsqueeze(2)    # (B, H, T_q, T_k, d_z)
        dist_sq = (diff ** 2).sum(dim=-1)                     # (B, H, T_q, T_k)

        return -dist_sq / self.tau.abs().clamp(min=1e-6)
