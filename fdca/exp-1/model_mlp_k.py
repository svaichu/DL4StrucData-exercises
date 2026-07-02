"""
mlp_k — observation encoder for FDCA Phase 1.

Maps obs (obs_dim) → latent z (d_z).
In the LDS setting this inverts the linear decoder D, so a single hidden layer
with enough width is exact capacity.
"""

import torch
import torch.nn as nn


class MlpK(nn.Module):
    """
    obs encoder: Linear → GELU → Linear

    Parameters
    ----------
    obs_dim : input observation dimension
    d_z     : latent dimension (output)
    hidden  : hidden layer width (default: 2 * obs_dim)
    """

    def __init__(self, obs_dim: int, d_z: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = 2 * obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_z),
        )
        self.obs_dim = obs_dim
        self.d_z = d_z

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (*, obs_dim) → z: (*, d_z)"""
        return self.net(obs)
