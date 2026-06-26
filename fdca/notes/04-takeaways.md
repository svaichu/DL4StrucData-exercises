# 04 — Takeaways, Risks & Prior Work

The things to remember from this session that aren't captured by the gate/data/ablation
mechanics.

---

## 1. The idea is two claims — only one is novel

**(a) Distance/RBF scoring instead of dot-product.** *Not* a contribution. Swapping
`q·k` for `−‖q−k‖²/τ` is "RBF attention" and has been run directly (a TinyStories swap
converged and was slightly better than SDPA in that constrained setup). There's a deeper
kernel-attention literature too (Transformer Dissection's kernel view; implicit/Fourier
kernel attention). Consequence: **don't sell "row-stochastic, no dot-product" as the
idea** — vanilla softmax attention is also row-stochastic, and RBF-vs-dot-product is a
known axis.

**(b) Apply a frozen pretrained one-step forward-dynamics model to the key before
comparing it to the query.** *This* is the actual bet: "attend to your causal
predecessor." The Phase-1 objective (latent next-state prediction, stop-grad target) is
itself essentially SPR / BYOL-style self-predictive representation learning — also
established. What's new is **reusing that frozen rollout as the attention scorer.** Aim
every experiment at isolating (b).

---

## 2. Prior-work map (for positioning / related-work section)

**RBF / kernel attention**
- "Scaled RBF Attention: Trading Dot Products for Euclidean Distance" (Pisoni, 2026) —
  direct SDPA→RBF swap, TinyStories.
- "Transformer Dissection: A Unified Understanding of Transformer's Attention via the
  Lens of Kernel" (2019).
- Implicit Kernel Attention; FourierFormer (Fourier integral kernels).

**Self-predictive / latent forward models (≈ the Phase-1 objective)**
- BYOL — Grill et al., 2020 (stop-grad + predictor, anti-collapse without negatives).
- SPR — Schwarzer et al., 2020 (latent dynamics model, no reconstruction; Atari 100k).
- PBL — Guo et al., 2020 (forward + reverse latent prediction).
- BYOL-Explore; BYOL-γ (Lawson et al., 2025); TD-JEPA (2025); JEPA / world-model line.
- Collapse-avoidance toolkit: stop-grad (SimSiam), EMA targets, VICReg, Barlow Twins.

**Growing depth as the task gets harder (≈ the greedy-stacking question)**
- Greedy layer-wise pretraining — Hinton et al. 2006 (DBNs), Bengio et al. 2007. Each
  layer models the representation *below* it, frozen before the next is added. **This is
  the direct precedent for the "train mlp_k/predict one layer at a time" idea.**
- Progressive Neural Networks — Rusu et al. 2016 (incremental columns, lateral
  connections; multi-task framing).
- Net2Net / progressive growing — function-preserving layer insertion.
- Depth-stacking during LLM training (freeze early layers, add + train later ones).

---

## 3. Risks, ranked by how much they bite

1. **Domain-transfer gap (biggest).** `mlp_k/predict` learn robot `(ee_pos, obj_pos)`
   dynamics; in Phase 2 they score residual-stream-derived *pseudo-observations*
   (`o = Wq·h`). The frozen forward model is a *fixed vector field*; the only adaptation
   is `Wq/Wk` contorting `h` into it. No a priori reason a robot-arm field is a good
   relational prior for an arbitrary task. If Phase-2 is in the same domain the gap
   shrinks — but then the baseline question sharpens (why not just a dynamics model?).
2. **Expressiveness bottleneck.** Standard MHA gets head/layer diversity from *learnable*
   Q/K. Here every head/layer shares the *same frozen scorer*; diversity comes only from
   linear `Wq/Wk`. One frozen one-step field serving every relational pattern → expect
   underfitting vs baseline outside the matched task.
3. **Asymmetry can collapse to symmetric distance.** Directionality needs `predict ≠
   identity`. Stop-grad training can drift it toward identity → C degenerates into plain
   RBF. (This is why the collapse instrumentation in `03-ablations.md` exists.)
4. **One-step model, possibly multi-step retrieval.** `predict` is a single step, but the
   useful antecedent may be several steps away or not a dynamics step at all.
5. **Mechanism in search of a task.** Hyper-specialized prior; needs the narrow
   intersection where "retrieve causal predecessor" is correct AND a transformer is
   natural AND the baseline is fair.

---

## 4. The multi-layer problem (don't gloss over this)

Standard transformers stack cleanly because every layer's scorer *is* the learned
similarity, in-domain by construction. FDCA breaks that the moment you pass layer 1: the
frozen scorer only knows raw-observation dynamics, but layer 2 receives *blended
attended features*. **The scorer is semantically stale by layer 2** — a structural
mismatch, not a tuning problem.

Options (in rough order of honesty/practicality):
- **One FDCA layer, rest standard** — and call it "a dynamics-informed attention module,"
  not "a new attention mechanism." Probably where you'd converge anyway. (= Gate 2.)
- **Layer-specific scorers, greedy** — pretrain each `(mlp_kˡ, predictˡ)` on the actual
  layer-ℓ input distribution. Fixes staleness *at init*; drift remains. (= Gate 3.)
- **Greedy as init, then joint fine-tune** — most practical for true depth; weakens the
  "frozen prior" story.

The doc's "multi-head, multi-layer ready" ambition is currently *ahead* of what the
mechanism supports. Honest scope for now: **one FDCA layer, single-layer ablation,
synthetic gate first.**

---

## 5. Framing (this determines whether the project "succeeds")

- Frame as **"characterize *when* this inductive bias helps,"** not "build a better
  general attention mechanism." The §7 framing in the design doc (it should win when the
  task rewards a retrieve-the-predecessor prior, and may trail otherwise — and that
  contrast is itself the result) is the right scientific stance. Keep it.
- Honest prediction: wins on tightly-matched synthetic dynamics tasks; trails vanilla
  attention on general sequence tasks; the forward-roll asymmetry needs ablations B/D to
  prove it does anything beyond the known RBF effect.
- Keep general sequence modeling (language) **off the table** until the mechanism is
  proven — the right token to attend to is rarely your dynamical predecessor.

---

## 6. Immediate next action

Build the **Gate 1** harness: an orthogonal-`A` linear dynamical system + a
displaced-predecessor retrieval task with a difficulty knob (distractor count, `d_z`,
decoder nonlinearity), wired to run the four-variant ablation ladder (A/B/C/D) with the
health instrumentation logged. A few hundred lines; minutes to train; the single most
decision-relevant thing you can do.
