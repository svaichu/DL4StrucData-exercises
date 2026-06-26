# 02 — Synthetic Data Sources

Scope: choosing the **Gate 1** dataset — the one that tests whether forward-roll
scoring works at all.

---

## The principle (re-stated, because it inverts the obvious choice)

> "I want data where `predict` works very well" → that means **synthetic, not robot
> trajectories** for the first cut.

Gate 1 tests *retrieval*, so `predict` must be near-exact. Any forward-model error
confounds the result: a mediocre roll lands in the wrong place in latent space, the L2
score points at the wrong key, and you can't tell whether the *idea* is bad or just the
*dynamics model* is. So: clean dynamics, hard retrieval.

---

## Why robot trajectories are the WRONG Gate-1 choice

Two killers, the first baked into the current FDCA spec:

1. **The Markov trap.** `obs = (ee_pos, obj_pos)` is *not Markovian* for anything with
   inertia — position alone doesn't determine the next position, you need velocity.
   Feed `predict` only positions and it physically *cannot* predict the next state for a
   moving arm; best case it learns a blurry average. So `predict` is mediocre *by
   construction* and Gate 1 is confounded before you start.
   *(Worth fixing in the main design too: include velocity in the observation, or
   restrict to overdamped/first-order dynamics where position determines the next step.)*

2. **Noise, contact, partial observability.** Real trajectories carry sensor noise;
   contact dynamics (pushing an object) are discontinuous and highly nonlinear — exactly
   what a small one-step MLP fits worst. You also don't control where the predecessor
   sits, so you can't build the displaced-retrieval task cleanly.

Robot data is the right **in-domain gate later** (once the mechanism is proven, "does it
work on the actual target domain" is a real test). Just not first.

---

## Recommended ladder

### 1. Linear dynamical system (start here — cleanest possible Gate 1)

`z_{t+1} = A · z_t`, fixed random `A`.

- Use an **orthogonal / rotation `A`** (eigenvalues on the unit circle) so trajectories
  neither collapse nor blow up — they trace clean loops in latent space.
- `predict` here is *literally learning `A`*, a single linear map → converges to ~zero
  error → removes the forward-model confound entirely.
- Observations: use `z` directly, or `obs = D · z` through a fixed random decoder `D` if
  you want `obs_dim ≠ d_z`.
- A few lines of numpy. Full control over dimensionality and distractor count. Isolates
  the mechanism perfectly.

### 2. Smooth deterministic ODE (check it survives nonlinearity)

Damped pendulum / spring–mass integrated at fixed `dt`, **with velocity in the
observation** so it stays Markovian. A small MLP nails the one-step map.

- Lorenz / Rössler at small `dt` also works and gives recognizable, *visualizable*
  trajectories (nice for inspecting attention patterns). Chaos only bites at long
  horizons, so one-step retrieval is fine.
- Purpose: confirm the mechanism isn't secretly relying on linearity.

### 3. Robot trajectories (in-domain gate, later)

With velocity in the obs, ideally contact-free motion first, then add pushing/contact.
This is Gate 2 / transfer territory, not Gate 1.

---

## The difficulty knob (tune the TASK, keep the DYNAMICS easy)

Keep dynamics easy, but don't let the *whole task* go trivial or dot-product solves it
too and you get no separation. Push difficulty entirely through retrieval:

- **Predecessor displacement** — random, non-adjacent position (defeats recency).
- **Distractor count** — enough non-antecedent tokens that naive similarity fails.
- **`d_z` / `obs_dim`** — high enough that the dot-product baseline can't just memorize.
- **Decoder nonlinearity** — optional, via a nonlinear `D`, to stress the encoder.

**Calibration target:** if FDCA *and* dot-product both score ~100%, raise
distractors/dimension until **dot-product cracks but FDCA holds**. That gap is your
result.
