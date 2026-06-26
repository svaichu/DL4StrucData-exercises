# FDCA — Experiment Planning Notes

Working notes for de-risking the **Forward-Dynamics Consistency Attention** idea
before committing to a full build. Companion to `Forward-Dynamics Consistency Attention.md`.

## TL;DR

- The idea is two separable claims. One (**distance/RBF scoring**) is *not* novel.
  The other (**a frozen forward-dynamics model used as the attention scorer**) is the
  actual bet, and the whole experiment should be aimed at isolating it.
- Run it — it's cheap and falsifiable — but frame it as *"characterize when this
  inductive bias helps,"* not *"beat the transformer."*
- Validate the mechanism on **synthetic data first**, single-layer, with a strict
  ablation ladder. Robot trajectories and multi-layer stacking are *later* gates,
  not the first one.

## Files

| File | Contents |
| --- | --- |
| `01-gates.md` | The sequential experiment ladder (what to run, in what order, pass/fail) |
| `02-synthetic-data.md` | Data sources for Gate 1, ranked; why robot data is the *wrong* first choice |
| `03-ablations.md` | The ablation strategy that makes any result interpretable |
| `04-takeaways.md` | Novelty map, prior work, ranked risks, the multi-layer problem, framing |

## The one-line decision rule for any benchmark

> Does the task's *correct* attention pattern correspond to "retrieve the dynamical
> antecedent"? If yes, FDCA has a shot. If no (e.g. most language), it doesn't —
> keep it off the table until the mechanism is proven.
