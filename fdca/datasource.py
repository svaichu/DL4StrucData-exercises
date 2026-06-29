"""
Gate 1 data source: Linear Dynamical System with displaced-predecessor retrieval.

System
------
    z_{t+1} = A @ z_t,   A orthogonal (eigenvalues on the unit circle)
    obs     = D @ z       D: obs_dim × d_z, fixed random decoder

Displaced-predecessor construction
------------------------------------
Each context window contains K independent chains interleaved in a uniformly
random order, so the dynamical predecessor of every token lands at a
*non-adjacent, random* position among distractors from the other chains.

Plain recency-based attention fails because the predecessor is never at i-1.
Dot-product attention fails as distractors grow (they share the same A-subspace).
Forward-roll FDCA is the only scorer whose inductive bias matches the task.

Batch API
---------
    ds = LinearDSDataSource(d_z=16, obs_dim=32, seq_len=64, n_chains=8,
                            batch_size=64, seed=42)
    obs, targets, pred_idx = ds.sample()

    obs      : (B, T, obs_dim)  — shuffled multi-chain observations
    targets  : (B, T, obs_dim)  — chain-next observation (what model must predict)
    pred_idx : (B, T) int       — shuffled position of each token's predecessor
                                   (-1 = first token of its chain, no predecessor
                                    in context)

Compatibility shim
------------------
    obs, action = ds.sample_compat()   # action repurposed as targets; (B, T, obs_dim)

Requirements: seq_len % n_chains == 0
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_orthogonal(d: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """Random d×d orthogonal matrix via QR of a Gaussian matrix."""
    A = torch.randn(d, d, device=device, generator=generator)
    Q, _ = torch.linalg.qr(A)
    return Q


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

class LinearDSDataSource:
    """
    Batched LDS data source for Gate 1.

    Parameters
    ----------
    d_z       : latent dimension (A is d_z × d_z)
    obs_dim   : observation dimension (obs = D @ z); must be >= d_z
    seq_len   : T tokens per sequence; must be divisible by n_chains
    n_chains  : K chains interleaved per context window (controls distractor density)
    batch_size: B
    seed      : fixed seed for A and D (reproducible across calls); sample
                randomness comes from a separate unseeded generator
    device    : torch device
    """

    def __init__(
        self,
        d_z: int = 16,
        obs_dim: int = 32,
        seq_len: int = 64,
        n_chains: int = 8,
        batch_size: int = 64,
        seed: int = 42,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        assert seq_len % n_chains == 0, (
            f"seq_len ({seq_len}) must be divisible by n_chains ({n_chains})"
        )
        assert obs_dim >= d_z, "obs_dim must be >= d_z"

        self.d_z = d_z
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.n_chains = n_chains
        self.chain_len = seq_len // n_chains   # tokens per chain per context
        self.batch_size = batch_size
        self.device = device

        # Fixed dynamics and decoder — seeded for reproducibility
        gen = torch.Generator(device=device).manual_seed(seed)
        self.A = _random_orthogonal(d_z, device, gen)       # (d_z, d_z)

        D_raw = torch.randn(obs_dim, d_z, device=device, generator=gen)
        self.D = D_raw / D_raw.norm(dim=1, keepdim=True)    # (obs_dim, d_z)

        # Alias for compatibility with CFG-based code
        self.action_dim = obs_dim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., d_z) → obs: (..., obs_dim)"""
        return z @ self.D.T

    def _roll_chain(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        z0   : (B, K, d_z)
        steps: number of forward steps to take
        Returns (B, K, steps+1, d_z) — the initial state plus `steps` successors.
        """
        zs = [z0]
        for _ in range(steps):
            zs.append(zs[-1] @ self.A.T)
        return torch.stack(zs, dim=2)   # (B, K, steps+1, d_z)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw one batch.

        Returns
        -------
        obs      : (B, T, obs_dim)  shuffled observation context
        targets  : (B, T, obs_dim)  chain-next observation for each token
        pred_idx : (B, T) long      shuffled position of predecessor (-1 = none)
        """
        B = self.batch_size
        T = self.seq_len
        K = self.n_chains
        L = self.chain_len
        device = self.device

        # ---- 1. Sample K independent chains per batch element ----
        # Initial states on the unit sphere: (B, K, d_z)
        z0 = torch.randn(B, K, self.d_z, device=device)
        z0 = z0 / z0.norm(dim=-1, keepdim=True)

        # Roll each chain L steps → (B, K, L+1, d_z)
        zs = self._roll_chain(z0, L)

        z_cur  = zs[:, :, :L, :]   # (B, K, L, d_z)  context tokens
        z_next = zs[:, :, 1:, :]   # (B, K, L, d_z)  their chain successors

        # ---- 2. Flatten to (B, T, d_z) ----
        z_flat    = z_cur.reshape(B, T, self.d_z)
        znext_flat = z_next.reshape(B, T, self.d_z)

        # ---- 3. Build pre-shuffle predecessor index ----
        # Token at flat position k*L + l has predecessor k*L + (l-1), l > 0
        pred_flat = torch.full((B, T), -1, dtype=torch.long, device=device)
        for k in range(K):
            s = k * L
            if L > 1:
                # positions s+1 … s+L-1 → predecessors s+0 … s+L-2
                pred_flat[:, s + 1 : s + L] = torch.arange(
                    s, s + L - 1, device=device
                ).expand(B, -1)

        # ---- 4. Shuffle tokens independently per batch element ----
        perm     = torch.stack([torch.randperm(T, device=device) for _ in range(B)])
        inv_perm = torch.zeros_like(perm)
        inv_perm.scatter_(1, perm, torch.arange(T, device=device).expand(B, -1))

        bidx = torch.arange(B, device=device)[:, None]          # (B, 1) broadcast
        obs_shuffled    = z_flat[bidx, perm]                     # (B, T, d_z)
        target_shuffled = znext_flat[bidx, perm]                 # (B, T, d_z)

        # ---- 5. Remap predecessor indices through the permutation ----
        # For the token now at shuffled position t:
        #   its original flat index is perm[b, t]
        #   its predecessor in flat space is pred_flat[b, perm[b, t]]
        #   its predecessor in shuffled space is inv_perm[b, pred_flat[b, perm[b, t]]]
        pred_flat_perm = pred_flat[bidx, perm]   # (B, T): flat predecessor of shuffled token t

        pred_idx = torch.full((B, T), -1, dtype=torch.long, device=device)
        valid = pred_flat_perm >= 0              # (B, T)
        bv, tv = valid.nonzero(as_tuple=True)
        pred_idx[bv, tv] = inv_perm[bv, pred_flat_perm[bv, tv]]

        # ---- 6. Decode to observation space ----
        obs     = self._decode(obs_shuffled)     # (B, T, obs_dim)
        targets = self._decode(target_shuffled)  # (B, T, obs_dim)

        return obs, targets, pred_idx

    def sample_compat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compatibility shim: returns (obs, targets) matching the (obs, action)
        convention used by train_phase1.py / train_phase2.py helpers.

            obs, action = ds.sample_compat()

        Both tensors are (B, T, obs_dim).
        """
        obs, targets, _ = self.sample()
        return obs, targets
