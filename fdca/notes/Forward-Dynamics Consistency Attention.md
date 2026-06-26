# Forward-Dynamics Consistency Attention (FDCA) — Design Plan

A new attention mechanism that replaces dot-product selection with a
**latent forward-dynamics consistency** score, pretrained self-supervised on
robot trajectories and frozen into a transformer layer.

---

## 1. Core idea

Standard attention selects with directional similarity: `score(i, j) = qᵢ · kⱼ`.

FDCA selects with **causal-predecessor consistency**:

> Query `i` attends to key `j` to the degree that rolling key `j`'s state
> one step forward in latent space lands on query `i`'s state.

The affinity is an **L2 distance** between key `j`'s forward roll and query
`i`'s latent — never an inner product on learned Q/K projections. After a
row-wise softmax, the score matrix is **row-stochastic by construction**.

---

## 2. Data

| Tensor   | Shape                  | Notes                                   |
|----------|------------------------|-----------------------------------------|
| `obs`    | `(B, T, obs_dim)`      | observation = `(ee_pos, obj_pos)`       |
| `action` | `(B, T, action_dim)`  | e.g. `delta_ee_pos` (see §6 — currently unused) |

`obs_dim` and `action_dim` are set by the dataset. Transitions are formed
along the time axis: `(obs[:, t], obs[:, t+1])`.

---

## 3. Phase 1 — self-supervised pretraining (on robot trajectories)

Train two modules with no human labels (target harvested from the data stream):

- **`mlp_k`**: `obs → z`  — latent encoder, `obs_dim → d_z`
- **`predict`**: `z → z` — one-step **autonomous** latent forward model
- **`τ`**: a learnable scalar temperature

**Objective** (latent forward prediction with stop-gradient, BYOL-style):

```
L_phase1 = mean over (t)  ‖ predict( mlp_k(obs_t) ) − sg( mlp_k(obs_{t+1}) ) ‖²
```

- `sg(·)` = **stop-gradient** on the target. This is the collapse guard:
  without it, `mlp_k → const` and `predict → identity` drives the loss to
  zero trivially. Stop-grad on the target breaks that degenerate solution.
- Optionally add a tiny variance term later if collapse is still observed.

**After Phase 1:** freeze `{ mlp_k, predict, τ }`. These never update again.

> Note: `mlp_q` and `mlp_v` from earlier drafts are **gone**. The L2 form
> compares query and key in the same latent space (one shared encoder), and
> the value path is a standard learnable `Wv` in Phase 2.

---

## 4. Phase 2 — the FDCA layer (multi-head, multi-layer ready)

Inputs to the layer are residual-stream vectors `h ∈ ℝ^{d_model}`.
Per head, for query position `i` and key position `j`:

```
# learnable per-head projections into pseudo-observation space (→ obs_dim)
oᵢ = Wq · hᵢ
oⱼ = Wk · hⱼ

# frozen shared encoder (same for query and key)
zᵢ = mlp_k(oᵢ)
zⱼ = mlp_k(oⱼ)

# frozen forward roll — applied to the KEY only
# (this asymmetry, plus Wq ≠ Wk, is what makes attention directional)
ôⱼ = predict(zⱼ)

# relational scalar: distance between key's roll and query's state
score(i, j) = − ‖ ôⱼ − zᵢ ‖² / τ

# row-stochastic weights (+ causal mask if the task is autoregressive)
A = softmax_j( score )

# aggregate learnable values
Vⱼ  = Wv · hⱼ
outᵢ = Σⱼ A[i, j] · Vⱼ
```

Concatenate heads → output projection `Wo`. The layer drops into a standard
transformer block in place of multi-head attention:
`x = x + FDCA(LN(x));  x = x + FFN(LN(x))`.

**Parameter inventory**

| Component                    | Status      |
|------------------------------|-------------|
| `Wq, Wk, Wv, Wo` (per head)  | learnable   |
| `mlp_k`                      | frozen      |
| `predict`                    | frozen      |
| `τ`                          | frozen      |
| FFN, LayerNorms, embeddings  | learnable   |

The frozen scorer means the **only** handle the transformer has on the
attention *pattern* is the learnable `Wq / Wk` projections (and indirectly
`Wv`). That strong inductive bias is the central experiment.

---

## 5. Baseline & benchmark

- **Baseline:** an otherwise identical transformer with standard scaled
  dot-product attention swapped in for the FDCA layer — same `d_model`,
  number of heads, number of layers, FFN width, optimizer, schedule, and
  parameter budget (as close as the two formulations allow).
- **Metric:** prediction quality on the chosen sequence task
  (loss + task accuracy / error), plus training curves.
- **To be supplied:** the Phase-1 trajectory dataset, the Phase-2 benchmark
  task, and whether it is **causal** or **bidirectional**.

---

## 6. Open decision: is `action` used?

The locked Phase-1 objective uses **only observations**
(`predict` is an *autonomous* `z → z` roll). The `action` tensor you provided
is therefore **currently unused**. Two ways forward:

1. **Keep autonomous (current spec).** Ignore `action`. Simplest; `predict`
   learns the marginal forward dynamics. Works at inference because no action
   is needed to compute `ôⱼ`.
2. **Action-conditioned `predict`.** Pretrain `predict(z_t, enc(a_t)) ≈
   sg(mlp_k(obs_{t+1}))`, which is a more faithful dynamics model. Catch: at
   Phase-2 inference there is no real action, so `predict` would need an
   inference-time action surrogate (e.g. a learned/zero default), or we keep
   it action-free at inference and only use actions to *shape* the encoder
   during pretraining.

Recommendation: start with **(1)** for a clean first benchmark; revisit (2)
if the autonomous roll under-selects.

---

## 7. Design properties to report

- **Row-stochastic:** guaranteed by the final softmax over keys.
- **No dot-product selection:** affinity is an L2 distance between a frozen
  forward roll and a frozen encoding; no `q·k` anywhere in selection.
- **Expected behavior:** FDCA should be strongest when the benchmark rewards a
  "retrieve the causal predecessor" prior, and may trail a vanilla transformer
  when it does not. That contrast is itself a result worth reporting.

---

## 8. Implementation checklist (once data arrives)

- [ ] Load `obs (B,T,obs_dim)`, `action (B,T,action_dim)`; build `(t, t+1)` pairs
- [ ] Build `mlp_k`, `predict`, learnable `τ`; train Phase 1 with stop-grad target
- [ ] Sanity-check against collapse (latent variance > 0; non-trivial scores)
- [ ] Freeze `{mlp_k, predict, τ}`
- [ ] Implement the FDCA layer (multi-head, optional causal mask)
- [ ] Build matched FDCA-transformer and dot-product-transformer
- [ ] Train both on the benchmark task; log loss / accuracy / curves
- [ ] Report prediction-quality comparison + attention-pattern visualizations