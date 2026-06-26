# 03 — Ablation Strategies

The ablations are not optional polish — without them, **a win is uninterpretable**.
You won't know whether you measured the forward-roll idea or just rediscovered the
already-known RBF-attention effect.

---

## The core ablation ladder

Same task, same parameter budget, same optimizer/schedule across all four:

| # | Variant | Score function | Isolates |
| --- | --- | --- | --- |
| A | Dot-product baseline | `qᵢ · kⱼ` | the reference point |
| B | Plain RBF / distance | `−‖zⱼ − zᵢ‖² / τ` (**no forward roll**) | the RBF effect alone (known, not novel) |
| C | Full FDCA | `−‖predict(zⱼ) − zᵢ‖² / τ` | the actual idea |
| D | FDCA, `predict ≡ identity` | `−‖zⱼ − zᵢ‖² / τ` (predict pinned) | second check the roll matters |

**B and D are the crucial controls.** The novelty in FDCA is *not* distance scoring —
that's well-trodden (see `04-takeaways.md`). It's specifically applying the frozen
forward model to the key before comparing. B and D both strip the forward roll, so:

- C > B and C > D → the roll is doing real work. **This is the only outcome that
  validates the idea.**
- C ≈ B ≈ D, all > A → you've reproduced RBF attention. The roll adds nothing.
- C ≤ A → the whole approach loses even on favorable ground.

> Note: B and D are the same score function but reached differently (B never has a
> predict module; D has it pinned to identity). Running both is a cheap consistency
> check that your harness isn't doing something subtle with the predict path.

---

## Collapse / health instrumentation (log every run)

The asymmetry — and the entire "causal predecessor" story — depends on `predict` being
meaningfully **non-identity**. BYOL-with-stop-grad is exactly the regime where the
predictor can drift toward identity. If that happens, C silently degenerates into B and
nobody notices. So instrument:

- **Latent variance** of `z` (per-dim and total) — collapse guard; should stay > 0.
- **`‖predict(z) − z‖`** distribution — how far is the roll from identity? If ~0, the
  directional story is gone regardless of accuracy.
- **Attention entropy** — are scores actually selective, or near-uniform?
- **Score-matrix asymmetry** — `‖A − Aᵀ‖`; a near-symmetric `A` means the directional
  prior isn't expressing.

These turn an ambiguous "it didn't beat baseline" into a diagnosable "it didn't beat
baseline *because* predict collapsed to identity."

---

## Multi-layer / greedy-specific ablation (Gate 3 only)

Task accuracy is insufficient at depth. Add:

- **Cross-layer scorer divergence** — is `predict²` functionally different from
  `predict¹`, or did they converge to the same map? Compare on a shared probe set.
- **Per-layer attention-pattern characterization** — does layer 1 capture low-level
  dynamics and layer 2 higher-order structure? Specialization is the publishable signal;
  identical scorers mean depth bought only parameters.

---

## Reporting

For every gate, report training curves + final metric for **all** ladder variants
together, plus the health instrumentation. The contrast *between* variants is the
result — a single FDCA number in isolation says almost nothing.
