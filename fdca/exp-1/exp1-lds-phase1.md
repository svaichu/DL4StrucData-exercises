# Exp-1 — Gate 1, Phase 1: Train `mlp_k` + `predict` on LDS

## What this experiment tests

Can we train a good encoder (`mlp_k: obs → z`) and forward model (`predict: z → z_next`)
from the LDS displaced-predecessor data?

Phase 1 is not about attention at all. It is purely supervised reconstruction:
given the predecessor's observation, encode it, roll it forward, and match the
current token's encoding. If this fails, Phase 2 (FDCA scoring) is pointless.

---

## Data

`LinearDSDataSource` from `datasource.py`:
- Dynamics: `z_{t+1} = A @ z_t`, `A` random orthogonal (unit-circle eigenvalues)
- Observations: `obs = D @ z`, `D` fixed random normalized decoder
- Each context: `K` chains interleaved in shuffled order → predecessor of token `t`
  is at a random non-adjacent position `pred_idx[b, t]` (or -1 if first in chain)

Recommended defaults for exp-1: `d_z=16, obs_dim=32, seq_len=64, n_chains=8, B=256`

---

## Model

**`mlp_k`** — obs encoder:
```
Linear(obs_dim, hidden) → GELU → Linear(hidden, d_z)
```
Hidden size = `2 * obs_dim` (or tunable). No final activation; z lives in ℝ^{d_z}.

**`predict`** — one-step forward model:
```
Linear(d_z, d_z, bias=False)
```
Linear layer only: the true dynamics is `A` (linear), so one linear map is the
correct capacity. A MLP here would overfit to identity. Track `||W - I||` to check it's
learning rotation, not identity.

---

## Training objective

For every token `t` in the batch where `pred_idx[b, t] >= 0`:

```
z       = mlp_k(obs[b, t])                         # encode current token
z_pred  = predict(mlp_k(obs[b, pred_idx[b, t]]))   # encode predecessor, roll forward
loss   += ||z - z_pred||²
```

Divided by the number of valid pairs. This is plain MSE with exact supervision —
no stop-gradient, no negative pairs, no contrastive term needed (we know the
ground-truth predecessor).

---

## Health instrumentation (log every step)

| Metric | What it tells you |
|---|---|
| `loss_mse` | Primary objective — should decrease to near-zero for LDS |
| `predict_drift = mean ||predict(z) - z||` | > 0 → predict learned something non-identity |
| `z_var = mean var(z, dim=0).mean()` | Latent variance — collapse guard; must stay > 0 |
| `W_identity_dist = ||W_predict - I||_F` | For linear predict: how far from identity matrix |
| `grad_norm` | Training stability |

---

## Expected outcome

With LDS, `mlp_k` has to invert a linear map (`D`) and `predict` has to learn another
linear map (`A`). Both are expressible exactly by one linear layer, so:
- MSE should converge to near zero (< 1e-3 relative to initial)
- `predict_drift` >> 0 (it learns `A`, not identity)
- `z_var` stays bounded above zero

If MSE does not converge: check that `obs_dim >= d_z` (otherwise `D` is not
injective and recovery is impossible) and that LR / batch size are sensible.

---

## Outputs

- `checkpoints/phase1_mlp_k.pt` — `mlp_k` state dict
- `checkpoints/phase1_predict.pt` — `predict` state dict
- `checkpoints/phase1_tau.pt` — learned temperature (init 1.0, softplus-positive)

These are loaded in Phase 2 to construct `FDCAScorer`.
