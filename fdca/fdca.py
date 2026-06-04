"""
Phase 2: FDCA layer — multi-head attention with a swappable score function.

The default score function is scaled dot-product (placeholder).
After Phase 1 pretraining, replace it with FDCAScorer (frozen mlp_k + predict).

Usage
-----
# placeholder (dot-product)
layer = FDCALayer(d_model=256, n_heads=4, obs_dim=16)

# after Phase 1
scorer = FDCAScorer(mlp_k, predict, tau)
layer = FDCALayer(d_model=256, n_heads=4, obs_dim=16, scorer=scorer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from scorers import DotProductScorer


# ---------------------------------------------------------------------------
# Multi-head FDCA layer
# ---------------------------------------------------------------------------

class FDCALayer(nn.Module):
    """Multi-head FDCA attention layer.

    Drop-in replacement for nn.MultiheadAttention in a transformer block:
        x = x + layer(LN(x), causal=True)
        x = x + FFN(LN(x))

    Parameters
    ----------
    d_model  : residual stream width
    n_heads  : number of attention heads
    obs_dim  : pseudo-observation dimension (used only by FDCAScorer)
    scorer   : DotProductScorer (default) or FDCAScorer (after Phase 1)
    causal   : if True, apply an autoregressive mask
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        obs_dim: int,
        scorer: Optional[nn.Module] = None,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.causal = causal

        if scorer is None:
            scorer = DotProductScorer()
        self.scorer = scorer

        proj_dim = self.d_head if isinstance(scorer, DotProductScorer) else obs_dim
        self.Wq = nn.Linear(d_model, n_heads * proj_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * proj_dim, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def swap_scorer(self, new_scorer: nn.Module, obs_dim: int) -> None:
        """Replace scorer in-place after Phase 1; re-init Wq/Wk for new proj_dim."""
        self.scorer = new_scorer
        proj_dim = self.d_head if isinstance(new_scorer, DotProductScorer) else obs_dim
        d_model = self.Wq.weight.shape[1]
        self.Wq = nn.Linear(d_model, self.n_heads * proj_dim, bias=False).to(self.Wq.weight.device)
        self.Wk = nn.Linear(d_model, self.n_heads * proj_dim, bias=False).to(self.Wk.weight.device)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)  →  (B, T, d_model)"""
        B, T, _ = x.shape
        proj_dim = self.Wq.weight.shape[0] // self.n_heads

        # Project and split into heads: (B, n_heads, T, proj_dim)
        q = self.Wq(x).view(B, T, self.n_heads, proj_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, proj_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = self.scorer(q, k)  # (B, n_heads, T, T)

        if self.causal:
            scores = scores + self._causal_mask(T, x.device)

        attn = F.softmax(scores, dim=-1)                    # (B, n_heads, T, T)
        out = torch.matmul(attn, v)                         # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, -1)         # (B, T, d_model)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# Transformer block built around FDCALayer
# ---------------------------------------------------------------------------

class FDCATransformerBlock(nn.Module):
    """Standard pre-norm block: x = x + FDCA(LN(x)); x = x + FFN(LN(x))"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        obs_dim: int,
        ffn_mult: int = 4,
        scorer: Optional[nn.Module] = None,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FDCALayer(d_model, n_heads, obs_dim, scorer=scorer, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
