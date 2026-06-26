# 01 — Gates

The experiments form a **sequential ladder**. Each gate is a go/no-go. Do not build
the next gate until the previous one passes — a failure at Gate 1 kills the idea
cheaply, before any heavy pipeline exists.

---

## Guiding principle

Gate design is about **isolating one variable at a time**. The single most common way
to waste effort here is to confound the *mechanism* (forward-roll scoring) with
something else (a weak forward model, a recency-solvable task, the known RBF effect).
Every gate is built so that a failure has exactly one interpretation.

A second principle runs through all of them:

> **Difficulty should live in the retrieval, not in the dynamics.**
> Keep `predict` near-exact so that if FDCA fails, you know it's the *attention*
> failing, not the forward model. Make the task hard via *displaced retrieval*
> (see below), not via realistic/noisy physics.

---

## Gate 1 — Does forward-roll scoring work at all?

**Setup:** Single FDCA layer. Synthetic data where "roll the key forward, match to
query" is *literally* the ground-truth attention pattern, and `predict` is in-domain
and essentially perfect (see `02-synthetic-data.md`).

**The critical task-design subtlety — displaced predecessor.**
If the task is plain next-step prediction on a Markov sequence, the correct antecedent
is always at position `i-1`, and a vanilla transformer (or even a 1-layer conv) solves
it with a pure recency prior. You'd learn nothing — both models win for boring reasons.
**Fix:** shuffle targets so each query's dynamical predecessor sits at a *random,
non-adjacent* position in the context, surrounded by distractors. Now recency is
useless and the model must score by forward-roll consistency. That is the regime where
FDCA's prior should beat dot-product *and* where the result is interpretable.

**Run the full ablation ladder here** (see `03-ablations.md`) — dot-product /
plain-RBF / full-FDCA / FDCA-with-predict-pinned-to-identity.

**Outcomes:**
- FDCA beats dot-product **and** beats plain-RBF → forward roll is doing real work →
  proceed to Gate 2.
- FDCA ties plain-RBF, both beat dot-product → you've rediscovered RBF attention; the
  roll isn't earning its keep (likely `predict ≈ identity` — check collapse
  instrumentation). Idea in current form is dead, cheaply.
- FDCA can't beat dot-product even *here*, in the maximally favorable setting → kill it
  before writing any pretraining pipeline.

**Cost:** a few hundred lines, trains in minutes at toy dimensions. This is the single
most decision-relevant experiment available.

---

## Gate 2 — Does it contribute inside a real stack?

**Setup:** One FDCA layer embedded in an otherwise standard multi-layer transformer
(vanilla dot-product attention above and below). Sweep the depth `d` at which the FDCA
layer is inserted.

**What it tests:** whether the dynamics prior helps when it's *one specialized module*
rather than the whole architecture. This is the honest, defensible deployment of FDCA —
analogous to inserting a cross-attention or memory-retrieval layer at one depth.

**Why this and not multi-layer FDCA first:** the frozen scorer is semantically stale by
layer 2 (it only knows raw-observation dynamics, not blended attended features — see
`04-takeaways.md`). Single-layer-in-a-stack sidesteps that entirely and still tests
real utility.

---

## Gate 3 — Can FDCA layers be stacked? (greedy layer-wise)

Only attempt if Gates 1 and 2 pass.

**Setup:** Greedy layer-wise training. Train `(mlp_k¹, predict¹)` on raw observations,
freeze, wire up FDCA layer 1. Run data through, collect layer-1 output activations,
train `(mlp_k², predict²)` on *those* as if they were observations, freeze, wire up
layer 2. Repeat. (Prior art: Hinton/Bengio greedy layer-wise pretraining — see
`04-takeaways.md`.)

**The remaining tension — drift.** Each scorer is trained on the *frozen* upstream
distribution, but once Phase-2 end-to-end training moves the upstream `Wq/Wk/Wv`, that
distribution shifts and the frozen scorer goes stale. Handling options:
1. Strict greedy: never end-to-end tune. Loses stack co-adaptation (most of what makes
   depth powerful).
2. Greedy as *initialization*, then unfreeze + global fine-tune. Most practical;
   weakens the "frozen prior" story but keeps it semantically sane.
3. Periodic re-freezing every N steps. Honest but fiddly — skip at research stage.

**The measurement that matters** is NOT just task accuracy. It's whether `predictˡ`
*specializes* across depth (e.g. layer 1 = low-level proximity dynamics, layer 2 =
higher-order relational structure), measurable in attention patterns. If all scorers
converge to the same function, depth bought you nothing but parameters. Demonstrated
specialization is the genuinely publishable result here — and it's a thesis chapter,
not a weekend.

---

## Summary

| Gate | Question | Layers | Data | Status to proceed |
| --- | --- | --- | --- | --- |
| 1 | Does the mechanism work at all? | 1 FDCA | synthetic (LDS first) | beats dot-product *and* plain-RBF |
| 2 | Does it help inside a stack? | 1 FDCA + vanilla | synthetic / in-domain | net positive vs all-vanilla |
| 3 | Can FDCA stack? | N FDCA, greedy | in-domain | scorers specialize across depth |
| (later) | In-domain transfer | as needed | robot trajectories | — |
