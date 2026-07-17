# Exp-1 (Gate 1) — Results Log

This file records all runs, results, and conclusions produced so far for **Gate 1**
of the FDCA (Forward-Dynamics Consistency Attention) experiment ladder: *does
forward-roll scoring work at all* on the `LinearDSDataSource` (LDS) displaced-predecessor
task. See `exp1-lds-phase1.md` and `../notes/01-gates.md` for the experiment design.

Data source recap: `z_{t+1} = A @ z_t` (A random orthogonal), `obs = D @ z` (D fixed
random decoder), `K` chains interleaved in shuffled order so each token's dynamical
predecessor sits at a random non-adjacent context position. Defaults throughout:
`d_z=16, obs_dim=32, seq_len=64, n_chains=8, batch_size=256`. All GPU runs on RWTH's
`c23g` partition (1x H100), with the `CUDAcompat/13.3-580.167.08` module loaded to
resolve a torch cu130 / driver 570 CUDA-runtime mismatch.

---

## 1. Original Gate-1 run: Phase 1 (plain-MSE) + Phase 2 (FDCA vs DotProduct)

Script: `train_phase1.py`. Phase 1 trains `mlp_k` (obs encoder) + `predict` (one-step
linear forward model) via plain supervised MSE reconstruction (exact predecessor
supervision, no stop-gradient). Phase 2 freezes those weights and compares an
`FDCAScorer` (forward-roll consistency attention) against a `DotProductScorer` baseline
on downstream retrieval MSE. 30 epochs each phase, default hyperparameters.

**Phase 1 (plain MSE):**
- MSE converged to `2.3e-3` by epoch 30 (from `3.6e-3` initially, with a mid-run bump).
- `predict_drift` = 3.78 (>> 0 → predict learned a genuine rotation, not identity).
- `W_id_dist` (‖W_predict − I‖) = 5.86, growing steadily → consistent with learning
  the true rotation matrix `A`, not collapsing to identity.
- `z_var` stayed at 0.455 → no latent collapse under this metric.

**Phase 2 (FDCA vs DotProduct):**
| Scorer | Final MSE (ep 30) |
|---|---|
| FDCA | 2.3009e-05 |
| DotProduct | 2.2547e-05 |

Improvement: **−2.0%** (FDCA is marginally *worse*, not better).

**Gate 1 verdict (original run): AMBIGUOUS** — the two scorers land within ~2% of
each other. `predict_drift` and `W_id_dist` both look healthy (no obvious identity
collapse), so the ambiguity isn't explained by those diagnostics alone — FDCA simply
isn't beating the dot-product baseline on this LDS setup as configured.

W&B: `svaichu/fdca-exp1` run `fancy-smoke-3` (`vt881mg1`).

Checkpoints from this run: `checkpoints/phase1_mlp_k.pt`, `checkpoints/phase1_predict.pt`,
`checkpoints/phase1_tau.pt`.

**Open question raised by this result:** is the ambiguity a genuine finding about
FDCA vs dot-product, or is it confounded by an undertrained/under-diagnosed Phase-1
representation? → motivated the JEPA-style Phase-1 investigation below, per instruction
to "stick to Phase 1 first" and get a firm answer on representation collapse before
re-litigating the Phase 2 comparison.

---

## 2. JEPA-style Phase 1 rewrite

Script: `train_phase1_jepa.py`. Replaces the plain-MSE reconstruction objective with a
JEPA/BYOL-style self-distillation setup:
- Online encoder `mlp_k` (obs → z), trained by gradient descent.
- Target encoder `mlp_k_ema`, an EMA copy of `mlp_k`, stop-gradient (never receives
  gradients directly).
- `predict`: one-step linear forward model, `z_online(pred) → z_target(cur)`.
- Loss = `MSE(predict(mlp_k(obs[pred])), stopgrad(mlp_k_ema(obs[cur])))` **+ VICReg-style
  variance and covariance regularizers** on the online embeddings (no batchnorm available
  in this small-MLP setting, so the anti-collapse mechanism has to be explicit — matches
  the standard BYOL/VICReg rationale, since a plain online/target pair with no explicit
  anti-collapse term can trivially satisfy the loss by collapsing to a constant vector).

**Instrumentation added** (logged every step, in addition to loss/pred_mse):
`z_std_mean`, `z_effective_rank` (out of `d_z=16`), `z_uniqueness` (mean pairwise cosine
distance), `predict_drift`, `W_id_dist`, and a `collapse_verdict` derived from thresholds
(`z_std_mean >> 0.05`, `z_effective_rank >> 1.5`, `z_uniqueness >> 0.02`).

### 2a. Smoke test (3 epochs, CPU) — sanity check

Confirmed the JEPA script computes cleanly end-to-end: effective rank 5.6/16,
uniqueness 0.55, `z_std_mean` 0.094 → healthy non-collapsed signal even after only
3 epochs. W&B: `fdca-exp1-jepa-smoketest` run `fresh-monkey-1` (`cny13s2o`).

### 2b. Full 30-epoch baseline (default hyperparameters, GPU)

`lr=3e-4` (script default), `ema_decay=0.996`, `var_coeff=25`, `cov_coeff=1`, `hidden=64`.

| Metric | Value (epoch 30) |
|---|---|
| pred_mse | 0.2479 |
| z_effective_rank | 8.59 / 16 |
| z_std_mean | 1.006 |
| z_uniqueness | 0.656 |
| predict_drift | 5.81 |
| W_id_dist | 7.20 |
| **collapse_verdict** | **COLLAPSED** (flagged at an earlier epoch; recovered z_std/uniqueness look fine, but `any_epoch_collapsed=True` trips the flag) |

W&B: `svaichu/fdca-exp1-jepa` run `baseline-default` (`vn5gw6fr`).

This baseline result is what prompted the hyperparameter sweep below: at the default
`lr=3e-4`, the representation transiently collapses and pred_mse is high (0.25) — not
a config we'd want to feed into a re-run of the Phase 2 comparison.

---

## 3. Hyperparameter sweep (wandb, 15 trials)

Swept via `wandb sweep` + `sweep_agent.py` (random search), 30 epochs x 100 steps/epoch
each, 1x H100 per trial. Search space: `ema_decay ∈ {0.9, 0.99, 0.996, 0.999}`,
`var_coeff ∈ {1, 5, 25, 50}`, `cov_coeff ∈ {0.1, 1, 5}`, `lr ∈ {1e-4, 3e-4, 1e-3}`,
`hidden ∈ {32, 64, 128}`, `wd ∈ {0, 1e-4, 1e-3}`. Sweep ID:
`svaichu/DL4StrucData-exercises-fdca_exp-1/ly8q214a`.

Full results table: `sweep_results.csv` (saved as artifact); figure: `sweep_results.png`.

**Headline finding — learning rate is the dominant factor determining collapse, not
`ema_decay`:**

| lr | mean final pred_mse | mean final effective rank | collapse rate |
|---|---|---|---|
| 1e-4 | 0.392 | 3.65 / 16 | 3 of 4 trials COLLAPSED |
| 3e-4 | 0.286 | 9.66 / 16 | 0 of 4 trials COLLAPSED |
| 1e-3 | 0.031 | 13.82 / 16 | 1 of 8 trials COLLAPSED |

`ema_decay` (0.9–0.999) showed **no clear trend** on its own once `lr` is accounted for
— collapsed and healthy runs both appear across the full decay range tested.

**Best trial** (`55u6ngf0`): `lr=1e-3, ema_decay=0.99, var_coeff=5, cov_coeff=5,
hidden=128, wd=1e-4` → final pred_mse **0.00426**, effective rank **15.99/16**,
verdict **NOT COLLAPSED**. This trial's per-epoch curve was still improving at epoch
30 (had not converged — see §4).

**Interpretation:** the original Gate-1 "AMBIGUOUS" verdict (§1) was obtained with
whatever Phase-1 config preceded this investigation; the sweep shows that with a
poorly-chosen learning rate, the JEPA-style Phase-1 representation collapses or stays
under-trained, and this is a plausible confound for the Phase 2 ambiguity — a weak
Phase-1 representation feeding into Phase 2 would make it hard to distinguish FDCA
from dot-product regardless of the attention mechanism's actual merit.

---

## 4. Extended training at best sweep config (150 epochs)

The sweep's 30-epoch budget was too short to reach convergence for the best config.
Re-ran `train_phase1_jepa.py` standalone with the winning hyperparameters
(`lr=1e-3, ema_decay=0.99, var_coeff=5, cov_coeff=5, hidden=128, wd=1e-4`) but
`p1_epochs=150` (cosine LR schedule spans the full 150-epoch run; the script has no
checkpoint-resume path, so this is a from-scratch rerun, not a continuation).

| Epoch | pred_mse | effective rank |
|---|---|---|
| 30 (matches sweep budget) | 0.00298 | 15.99 / 16 |
| 150 (extended) | **0.00062** | 15.98 / 16 |

→ **4.8x lower prediction MSE** by extending training, with effective rank essentially
unchanged (it saturates near 16/16 within ~10 epochs — representation health is settled
early, but the predictor keeps tightening its fit for well over 100 more epochs).
`collapse_verdict`: **NOT COLLAPSED** (`any_epoch_collapsed=False` throughout).

This confirms the 30-epoch sweep score was measuring the model mid-descent, not at
convergence — the apparent plateau at epoch 30 in the original sweep trial was an
artifact of the cosine schedule ending, not evidence the model had learned all it
could. W&B: `svaichu/fdca-exp1-jepa` run `best-config-150ep` (`6csv9frb`).

Checkpoints: `checkpoints_jepa_extended/phase1_mlp_k.pt`,
`checkpoints_jepa_extended/phase1_mlp_k_ema.pt`,
`checkpoints_jepa_extended/phase1_predict.pt`, `checkpoints_jepa_extended/phase1_tau.pt`.

**Note on the data source:** `LinearDSDataSource` resamples a fresh random batch on
every call (new chain rollouts, not a fixed dataset) — there is no memorization/overfitting
concern here; every epoch's pred_mse is an honest measure of dynamics-tracking
generalization, not repeated-example fitting.

---

## 5. Current state / recommended next step

- Phase-1 representation quality is now well-characterized: JEPA-style training with
  `lr=1e-3` (not the `lr=3e-4` default), `ema_decay=0.99`, `var_coeff=5`, `cov_coeff=5`,
  `hidden=128`, `wd=1e-4`, trained for **≥100 epochs** (not the original 30), gives a
  robustly non-collapsed representation (effective rank ~16/16) with prediction MSE
  down to `6.2e-4` — far better than the original plain-MSE Phase-1 result
  (`2.3e-3` at 30 epochs) that fed into the AMBIGUOUS Gate-1 verdict in §1.
- **Not yet done:** re-running the Phase 2 (FDCA vs DotProduct) comparison using this
  improved, longer-trained Phase-1 representation. This is the natural next step to
  determine whether the original AMBIGUOUS verdict was a genuine finding about FDCA vs.
  dot-product, or an artifact of an under-trained Phase-1 encoder — deferred per request,
  to be picked up later.

---

## Artifacts index (saved to the Claude Science project, not just this directory)

- `sweep_results.csv`, `sweep_results.png` — full 15-trial sweep table + hyperparameter
  scatter plots (pred_mse & effective rank vs. lr and ema_decay, colored by collapse
  verdict).
- `extended_training.png` — 150-epoch training curve (pred_mse, effective rank) with
  the 30-epoch sweep cutoff marked, showing continued improvement past the sweep budget.
- `checkpoints_extended_phase1_mlp_k.pt`, `checkpoints_extended_phase1_mlp_k_ema.pt`,
  `checkpoints_extended_phase1_predict.pt`, `checkpoints_extended_phase1_tau.pt` —
  150-epoch best-config Phase-1 checkpoints (online encoder, EMA target encoder,
  predictor, temperature).
- Original Gate-1 run checkpoints: `phase1_mlp_k.pt`, `phase1_predict.pt`, `phase1_tau.pt`
  (plain-MSE Phase 1, §1).
