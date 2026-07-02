"""
predict — one-step forward model for FDCA Phase 1.

Maps latent z_t → z_{t+1}.
For the LDS the true dynamics is z_{t+1} = A @ z_t (linear, no bias), so a single
bias-free linear layer is the exact right capacity. Using an MLP here risks learning
an identity map via shortcut — keep it linear and track ||W - I||_F as a health check.
"""

import torch
import torch.nn as nn


class Predict(nn.Module):
    """
    One-step forward model: Linear(d_z, d_z, bias=False)

    Learns the rotation matrix A of the LDS.

    Parameters
    ----------
    d_z : latent dimension (input = output)
    """

    def __init__(self, d_z: int):
        super().__init__()
        self.linear = nn.Linear(d_z, d_z, bias=False)
        self.d_z = d_z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (*, d_z) → z_next: (*, d_z)"""
        return self.linear(z)

    @torch.no_grad()
    def identity_distance(self) -> float:
        """||W - I||_F — should increase from sqrt(d_z) as training progresses."""
        W = self.linear.weight.data
        return (W - torch.eye(self.d_z, device=W.device)).norm().item()

    @torch.no_grad()
    def drift(self, z: torch.Tensor) -> float:
        """Mean ||predict(z) - z|| over a batch — zero for identity, > 0 for a real roll."""
        return (self(z) - z).norm(dim=-1).mean().item()
