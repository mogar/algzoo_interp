# Plan: interpreting the AlgZoo 2nd argmax RNNs

Goal: a mechanistic explanation of the 2nd argmax RNNs from
[AlgZoo](https://github.com/alignment-research-center/alg-zoo), ideally all
the way up to `hidden_size=32, seq_len=10` (1,376 parameters). Explanations
are checked quantitatively, not just told as stories.

This file is the single source for **what we do next** ([Roadmap](#roadmap)).
Companion docs:

- [GEOMETRY.md](GEOMETRY.md): activation patterns, cones, and why the model is
  piecewise linear.
- [HYPOTHESES.md](HYPOTHESES.md): current claims about the 32x10 model, with
  evidence, status and predictions. The roadmap says when each gets tested;
  HYPOTHESES.md records the outcome.

## Background

### The models

Every 2nd argmax model is an `alg_zoo.DistRNN`:

```
h_{-1} = 0
h_t = ReLU(W_ih * x_t + W_hh @ h_{t-1})      t = 0..seq_len-1, x_t is a scalar
logits = W_out @ h_{seq_len-1}                one logit per input position
```

- No biases anywhere. Parameters: `hidden + hidden^2 + seq_len * hidden`.
- Trained on `x ~ N(0, I)` (`th.randn`) with cross-entropy against
  `x.argsort()[-2]`.
- The zoo covers `hidden_size` in {2, 3, 4, 5, 6, 8, 16, 32} and `seq_len` in
  {2, 3, 4, 5, 6, 8, 10}, five seeds each, with checkpoints at
  `n_seqs` = 0, 2^20, ..., 2^30. By default `zoo_2nd_argmax` loads the
  best-of-5 seed.
- `example_2nd_argmax()` is the 16x10 model (432 parameters) discussed in
  the blog post.
- `handcrafted_2nd_argmax(seq_len)` for seq_len 2, 3 and 10 are hand-built
  solutions. The 10-step one (22 neurons) has commented neuron meanings, such
  as shift-register copies of `x_{t-k}`, the running max, and "max minus
  current", which gives us candidate features to look for.

### Structural facts we can exploit

1. **Piecewise linear.** With ReLU and no biases, fixing which neurons fire
   at each step makes the logits an exact linear function of `x`. The input
   space splits into polyhedral cones, each with one exact linear map. See
   [GEOMETRY.md](GEOMETRY.md).
2. **Scale invariant.** For any `c > 0`, `f(c x) = c f(x)`, so the predicted
   argmax depends only on the direction of `x`.
3. **Only order matters to the task.** So the model has to approximate
   comparisons using linear maps and ReLUs, and its errors should
   concentrate where comparisons are hard, such as near-ties.

### Conventions

- TransformerLens's `HookedTransformer` doesn't fit RNNs, so we unroll the RNN
  by hand (`algzoo_interp.hooked`). `HookedTransformer` becomes relevant only
  for the longest-cycle transformers (step 8).
- No Jupyter. Experiments are plain Python scripts that save image plots and
  JSON/CSV numbers to `results/`. Interactive views, if needed, go in a
  Streamlit app under `app/`.
- Every number quoted in a doc comes from a script in `scripts/`, and the doc
  says which.

## Repo layout

```
algzoo_interp/   reusable library code (loading, data, eval, hooks, plots)
scripts/         one runnable script per experiment; writes to results/
results/         figures, CSV/JSON numbers, and per-model writeups (markdown)
tests/           pytest checks for the library
app/             (later, if needed) Streamlit viewer built on algzoo_interp
weights/         local cache of downloaded checkpoints (gitignored)
docs/            this plan and companion notes
```

## Done so far

**Setup.** `alg-zoo` installed in editable mode from the submodule; checkpoints
cached in `weights/`; accuracy baseline for every 2nd argmax model in
`results/zoo_baseline/` (`scripts/zoo_baseline.py`).

**Tools** (`algzoo_interp/`, all tested):

- `hooked.py`: hand-unrolled RNN exposing pre-activations, hidden states,
  on/off masks and logits per step, with hooks; per-neuron logit
  contributions.
- `data.py`: `randn` inputs plus probes (fixed top-two positions, chosen
  top-two gap, sorted, one position swept).
- `eval.py`: accuracy overall, by top-two positions, by top-two gap.
- `interventions.py`: zero / mean / resample ablation and activation patching
  of any neurons at any steps.
- `patterns.py`: activation-pattern counting and coverage; exact linear map
  for any pattern.

**Scripts:** `inspect_model.py` (first-look figures for any model),
`plot_regions_2d.py` (cone pictures for seq_len 2), `neuron_hypotheses.py`
(event / running-value / large-weight evidence).

**Findings so far:**

| Model | Accuracy | Distinct patterns (2^18 inputs) | Patterns covering 90% |
|---|---|---|---|
| 2x2 | 99.99% | 6 | 5 |
| 4x3 | 98.4% | 67 | 13 |
| handcrafted 10 | 99.0% | 65,851 | 39,637 |
| example 16x10 | 95.3% | 174,722 | 148,508 |
| 32x10 | 99.5% | 257,562 | 231,348 |

- Explaining region by region works for small models only. 10-step models
  need per-neuron and per-step descriptions.
- 2x2: six cones; the decision boundary runs through two of them, whose
  linear maps put it at x_1 = 0.9997 x_0 and 1.0006 x_0 (GEOMETRY.md §5).
- 4x3: neurons form two sign-flipped pairs, (0, 3) and (1, 2), and the active
  pair alternates between steps.
- 32x10: neuron 7 fires exactly when `x_t > max(0, x_0..x_{t-1})` and neuron
  16 when x_t is a positive running top-2 value (99.6-100% agreement at every
  step); always-on neurons look like linear memory for running values; the
  largest weights look like resets and vetoes. All correlational so far
  (HYPOTHESES.md).

## Method: how each model gets analysed and judged

The roadmap applies these to specific models.

**Analysis toolkit.**

- **Neuron dictionary.** For each neuron, test candidate *events* (on/off
  matches "new running max", "new top-2", ...) and candidate *running values*
  (`x_{t-k}`, max so far, 2nd max so far, gaps, steps since an event). Record
  agreement / R², and label neurons as event detectors, always-on (linear)
  memory, or other.
- **Weight structure.** Large entries (resets, vetoes), shift registers or
  delay lines in `W_hh`, low-rank pieces.
- **Readout.** Per-neuron logit contributions through `W_out`.
- **Causal checks.** Every labelled claim gets an intervention: patching a
  neuron at one step, or removing a single weight (edge ablation).
- **Generality.** The same analysis on other seeds, checkpoints
  (`n_seqs`) and nearby model sizes.

**What counts as an explanation.**

1. **Rebuild by hand.** A simplified model written from the explanation
   agrees with the original's argmax on at least 99% of random inputs, and
   the disagreements are characterised.
2. **Resample checks.** Resampling everything the explanation calls
   irrelevant leaves the output (nearly) unchanged.
3. **Estimate accuracy without sampling.** The explanation predicts the
   model's accuracy, and where its errors fall, from the mechanism (e.g. cone
   volumes and failure conditions), and the prediction matches measurement.
   This is ARC's own bar.
4. **Write-up** in `results/<model>/README.md`: figures, claims, evidence, and
   the remaining unexplained gap as a number.

## Roadmap

Steps are in order. Steps 1-4 build the 32x10 explanation; step 5 develops the
"what counts as an explanation" checks on a model small enough to do exactly,
before step 6 applies them to 32x10. Step 5 is independent of 2-4 and can run
in parallel.

### Step 1: Reusable feature library, validated on the handcrafted model

Turn the event and running-value definitions in `neuron_hypotheses.py` into
library code, and check they recover a model whose answer we know.

- [x] `algzoo_interp/features.py`: running top-2 values, events (new max,
      new top-2, with and without the `max(0, ...)` convention, sign of
      `x_{t-k}`), steps since last event, shifted inputs `x_{t-k}`, and the
      handcrafted "max0 excluding one input, minus current" family.
- [x] `algzoo_interp/dictionary.py` + `scripts/neuron_dictionary.py`: for every
      neuron, best-matching event, best single fit, greedy 3-feature fit, a fit
      on all features at once, a fit on the handcrafted model's 22 hidden
      states as a basis (seq_len 10), and mean-ablation accuracy drop. Writes
      `results/<model>/dictionary/{dictionary.md, dictionary.json, r2.png}`.
      Supports `--seed`, `--n-seqs`, `--example`, `--handcrafted`.
      `neuron_hypotheses.py` now imports its feature code from the library.
- [x] Run on `handcrafted_2nd_argmax(10)`: **all 22 neurons recovered with
      R² = 1.000**, each matching its source comment (tested in
      `tests/test_dictionary.py`).

In the handcrafted model's 0-indexed terms, after reading `x_t`: neuron `9-k`
is `relu(x_{t-k})`; neuron 19 is `max0 - max0_prev` (how much the running max
just increased); 20 is `max0_prev`; 21 is `max0 - x_t`; and neuron `19-k`
(k = 1..9) is `max0` over everything except `x_{t-k}`, minus `x_t`. Here
`max0 = max(0, x_0..x_t)`.

**Results on the trained 10-step models** (also run in this step):

| | 32x10 | 16x10 (blog example) |
|---|---|---|
| Neurons that exactly match one handcrafted neuron (single-feature R² ≥ 0.95) | 7 = hc 19 (1.000), 17 = hc 20 (0.986) | 4 = hc 19 (0.994), 2 = hc 20 (0.974), 7 = hc 21 (0.986), 6 = hc 18 (0.972) |
| Linear fit on the 22 handcrafted neurons: R² ≥ 0.99 / ≥ 0.9 / median | 3 / 10 of 32 / 0.83 | 6 / 13 of 16 / 0.95 |
| Neurons whose ablation costs ≥ 10% accuracy | 23, of which 10 have handcrafted-basis R² ≥ 0.9 | 16, of which 13 have handcrafted-basis R² ≥ 0.9 |

Both trained models rebuilt the handcrafted model's running-max register
(hc 19/20). The 16x10 model is close to a compressed version of the whole
handcrafted scheme. The 32x10 model only partly follows it. These are linear
fits on post-ReLU values, so they understate matches for neurons that clip at
zero (see step 3).

### Step 2: Causal tests of the 32x10 event detectors, resets and veto (H1, H2, H5, H6)

- [x] Tool: **edge ablation** (`ablate_edges` in `interventions.py`): a copy
      of the model with chosen individual weights zeroed. Covers the "path
      patching" item deferred from Phase 1 for the cases we need.
- [x] **H1/H2.** Single-step patching between input pairs that differ only at
      step t (`scripts/causal_tests.py`).
- [x] **H5 resets.** Target neurons at t+1 conditioned on the source firing
      at t, and with the reset edges removed.
- [x] **H6 veto.** Removed 7 → logit 9 and 12 → logit 9, and measured
      accuracy by argmax position and the rank of x_9 in the new errors.

**Done when** each of H1, H2, H5, H6 is marked causal or refuted in
HYPOTHESES.md, with the script and numbers cited. **Done:** all four are
causal. Results in `results/2nd_argmax_32x10/causal/`:

- **H1:** with everything except neuron 7 patched, a new max at step t
  changes the prediction only 1-5% of the time (t=2..8); neuron 7 is
  necessary. Neurons 7 + 16 together carry 99% of the effect at t=2, less at
  later steps.
- **H2:** neuron 16 alone carries 74-93% of the effect of a new top-2 value;
  everything except 7 and 16, 1-6%.
- **H5:** every reset target is zero at t+1 98-100% of the time after its
  source fires. Cutting the edges shows 7 causes the resets of 0, 13, 28, 29;
  targets 3, 10, 25 are also zeroed by something else. Cutting 7's or 16's
  reset edges drops accuracy to 67% / 72%.
- **H6:** cutting 7 → logit 9 drops accuracy at "max at position 9" from 99%
  to 72%, and nowhere else.
- **New, H6b:** neuron 12 = `relu(max(0, running 2nd max) - x_t)` (R² 0.998)
  vetoes position 9 when x_9 is too *small*. Cutting it causes wrong "9"
  answers where x_9 is mostly the 3rd largest. Logit 9 is bracketed by two
  vetoes.
- **New feature** `2nd0_prev` also identifies neuron 31 as the running 2nd
  max (R² 0.963).

### Step 3: Memory and timing neurons in 32x10 (H3, H4)

- [x] **H3.** Per-lag fits resolve neurons 1 and 5 as the max − 2nd gap.
      Causal test (`causal_tests.py`, memory section): pairs that differ only
      in the running max's *value*, patched at one step.
- [x] ~~Fit pre-activations on the handcrafted basis.~~ Superseded by the
      state analysis below, which handles clipping by fitting within groups
      of inputs and needs no handcrafted basis.
- [x] **H4.** `algzoo_interp/state.py` + `scripts/state_analysis.py` (how
      much of each neuron the ideal online state explains; position codes;
      readout map; position flags) and `scripts/lag_codes.py` (on/off rate
      and per-lag fits by lag of max / 2nd max).
- [x] Neuron 12 (H6b, step 2), neuron 20 (position-0 flag), neuron 8 (still
      unclear, 8% drop).

**Done when** every neuron with a non-negligible ablation effect has a
dictionary label with causal support, or is explicitly listed as unexplained.
**Done**, with a caveat: every neuron has a role in the table in
[HYPOTHESES.md §D](HYPOTHESES.md#d-neuron-roles-in-32x10), but the clock and
lag-copy roles are correlational (very clean, not yet patched). Neuron 8 is
unclear; 4, 9, 14 are unexplained but cost ≤ 2% when ablated.

Results:

- **The hidden state is mostly the minimal algorithm state:** max, 2nd max,
  and their lags, plus the current input. Median R² 0.94 for 32x10 (21 of 32
  neurons ≥ 0.9), 0.97 for 16x10.
- **Position is a lag code.** Neuron 7's resets restart a clock: from lag 2
  on, many neurons' on/off state depends only on the lag of the max (median
  variation across steps 0.02), e.g. neuron 27 on exactly at lags 3 and 7.
  Neurons 22, 19, 2, 26, 15 carry the max only at lags 2, 3, 6, 7, 8.
- **H3:** neuron 17 alone transfers 65% of a changed max value to the next
  new-max decision; 17 with 1, 5, 31 transfers 80%; everything else 20%.
- **Not yet explained:** how the readout turns these codes into logits 0-8,
  and the lag of the 2nd max, which is encoded less cleanly than the lag of
  the max. Both feed step 6 (and could start now on 32x10 if step 4 is
  deferred).

### Step 4: Generality across seeds and sizes

- [x] `scripts/compare_models.py`: role matching (best neuron per key
      feature and event), reset share, largest readout weight, clock neurons
      and state R² for 13 models: 32x10 seeds 0-4, zoo 16x10, blog 16x10,
      handcrafted 10, 8x10, 8x6, 8x5, 8x4, 4x3 → `results/compare/compare.md`.
- [x] Causal tests on 32x10 seeds 1-4 with each seed's own neuron numbers;
      `scripts/summarize_causal.py` → `results/compare/causal.md`.
- [x] Smaller and low-accuracy models (in the same table).
- [ ] Optional: training checkpoints (`n_seqs`) of 32x10 to see the order in
      which roles appear. Not done.
- [x] Compare with the handcrafted scheme (in the same table).

**Done when** there's a table of roles × models and a short note on what is
universal and what is seed-specific. **Done**: see
[HYPOTHESES.md §E](HYPOTHESES.md#e-generality-other-seeds-and-model-sizes-roadmap-step-4).
In brief:

- **Universal in all five 32x10 seeds:** an exact new-max detector
  (`max0 − max0_prev`, R² 1.00), its veto on the last position (largest
  readout weight; causal in every seed), a "below the 2nd max" neuron with
  its own last-position veto (causal in every seed), resets from the
  detector, a lag clock (4-8 neurons), and minimal-state encoding (median R²
  0.92-0.95).
- **Seed-specific:** whether the detector is the *only* carrier of a new max
  (seed 0 yes; others share it), the importance of the top-2 detector and of
  the resets, and whether the running 2nd max sits in one neuron.
- **16x10 has two solution types**: the zoo model is 32x10-like, the blog
  model is handcrafted-like. The failing 8x10 model lacks the detector
  altogether; 8x6 and 8x5 have the detector but no clock.

So an explanation of 32x10 seed 0 should transfer to other seeds at the level
of mechanisms (detector, vetoes, clock, minimal state), but not neuron by
neuron. Step 6 should describe it in those terms.

### Step 5: Complete explanation of a small model (methods track)

Use 4x3 (98.4%) or 5x3 (99.9%), where cones can be listed exactly, to build
and debug the explanation checks before applying them to 32x10.

- [x] Explain every common cone: its linear map, where the decision boundary
      falls inside it, and the role of the alternating sign-flipped pairs.
- [x] Hand-built rebuild; resample check.
- [x] **Exact accuracy from the geometry**: `algzoo_interp/cones.py`
      enumerates every cone of a seq_len-3 model exactly (spherical
      polygons) and splits each by predicted and true class.
- [x] Write-up in [`results/2nd_argmax_4x3/README.md`](../results/2nd_argmax_4x3/README.md),
      all numbers from `scripts/explain_small.py` and `scripts/exact_accuracy.py`.

**Done when** all four explanation checks from [Method](#method-how-each-model-gets-analysed-and-judged)
pass on the small model. **Three pass and one is partial:**

- **Mechanism.** For 3 inputs the 2nd largest is the median. The model is
  sign-symmetric (f(−x) = f(x) on 99.6% of inputs, as the median is). Each
  step's on-pair records one comparison (x0 > 0, x1 > x0, x2 > x1). The
  readout depends only on v/u (v = h1 + h2, u = h0 + h3) with two thresholds,
  and u ≈ |4d − e|, v ≈ |4.7d + e| in terms of consecutive differences.
- **Exact accuracy: pass.** 70 cones; exact 0.98455 vs sampled 0.98443 ±
  0.00012, and within one standard error for 3x3, 5x3 and 8x3 too. 87% of
  the error is in the 24% of inputs where a neuron pair splits ("mixed") or
  all neurons are off at the last step.
- **Resample check: pass.** Swapping final states between inputs with the
  same v/u keeps 99.76% of predictions.
- **Rebuild: partial.** The ratio readout reproduces the model on 99.78% of
  inputs; the full two-number formula on 98.84% (just short of 99%). Forcing
  pairs to switch together makes the model *more* accurate (99.15%): about
  half its error comes from mixed pairs, half is built into the design.

**Method lessons for step 6.** Exact cone enumeration doesn't scale to 10
steps (too many cones), but the pieces that worked here do: finding a
symmetry, reducing the readout to a few combined quantities and thresholds,
and attributing error to states where neurons that should switch together
don't.

### Step 6: Quantitative explanation of 32x10

- [x] Rebuild from the step 2-3 explanation (`scripts/explain_32x10.py`):
      a surrogate final state built from the ideal state, run through the
      model's readout.
- [x] Resample check over everything the explanation calls irrelevant.
- [~] Mechanistic accuracy estimate: errors attributed to mechanisms; no
      sampling-free estimate.
- [x] Write-up in [`results/2nd_argmax_32x10/README.md`](../results/2nd_argmax_32x10/README.md).

**Done when** all four explanation checks pass, with the unexplained gap
stated as a number. **Two pass, one is partial, plus the write-up:**

- **Rebuild: pass, for correct behaviour.** A surrogate final state built from
  the *positions* of the max and 2nd max alone agrees with the model on
  99.44% of fresh inputs: exactly the inputs the model gets right (the
  surrogate is 100% accurate). The unexplained gap is the model's 0.56% of
  errors.
- **Resample check: pass.** Resampling every input except the max, 2nd max
  and last leaves 99.17% of predictions unchanged (99.61% where the model was
  right). Keeping only the top-2 positions: 98.92%. Both are at the
  (accuracy)² = 98.88% ceiling expected if errors are independent near-ties.
- **Accuracy estimate: partial.** 80% of errors fall in 7% of inputs singled
  out by two mechanisms: the max(0, …) convention can't rank negative values
  (2nd largest negative: 1.08% of inputs, 21.6% error, 42% of errors), and
  comparisons have finite resolution (2nd and 3rd within 3%: 5.8% of inputs,
  3.7% error, 38% of errors). A one-neuron error model from the top-2
  detector's biased threshold over-predicts its error type tenfold (8,858
  vs 833), so near-ties are resolved jointly by several neurons. No
  sampling-free estimate yet.

**Still open** (see the write-up's "What isn't explained"): how each of
logits 0-8 combines the clock, delay-line and reference neurons; the lag code
for the 2nd max; patching the clock; and a joint model of near-tie
resolution for a sampling-free accuracy estimate.

### Step 6b: How the readout decodes positions 0-8

- [x] `scripts/readout_analysis.py` → `results/2nd_argmax_32x10/readout/`;
      findings in [HYPOTHESES.md §G](HYPOTHESES.md#g-how-the-readout-decodes-positions-0-8).

**Result:**

- **Additive code + matched filter.** The unit-normalised mean final state is
  a(max position) + b(2nd position) (R² 0.962). The readout's W·b(j) is
  highest at logit j, by 0.9-1.4; W·a slightly favours the max's own
  position, and the 2nd-max margin overrides it.
- **A delay line for the top two.** Delay neurons are on 99-100% whenever the
  input at their lag is the running max *or* 2nd, so they survive a new max
  demoting the old one. Membership flags cover the other lags.
- **Causal.** Patching 15 neurons (delay line, flags, reference, 12) moves
  the prediction to another input's 2nd-max position 80% of the time.
- **Errors.** The value-dependent part of the state usually moves the
  decisive margin by about a quarter; errors are where it outweighs the
  margin. Only 12.6% of it lies in the directions the readout can see
  (random: 28%).

- **The remaining 20%.** 8 more neurons (3, 4, 14, 8, 5, 21, 16, 23) take
  the 2nd-position transfer to 99.6%. The 15 fall short mainly at lags 1-2
  and when the max is last.
- **Max vs 2nd identity is a separate code** (H12). Swapping the max and 2nd
  leaves the membership code unchanged (delay line and flags transfer 0%).
  The reference neurons (62%), clock (35%) and new-max detector (9%) carry
  which input is the max; 10 neurons carry 94%.
- **Sufficiency.** The 25 membership, identity and last-position neurons,
  with the other 7 mean-ablated, agree with the model on 95% of inputs. The
  other 7 don't carry position (resampling them within the same position
  cell changes as many predictions as resampling the kept 25).

**Open:** a neuron-by-neuron account of how the reference neurons' lag-dependent
mix and the clock mark the max at each lag.

### Step 6c: Completion checks

- [x] **Script smoke tests** (`pytest -m scripts`) and `scripts/run_all.sh`:
      every committed result reproduces exactly from the current code.
- [x] **Clock causal test** (`clock_test.py`): patched alone at step 4-8,
      the clock moves the final prediction 34-41%; with the reference neurons
      and neuron 7, 85-92%. The clock carries information forward but
      partly re-derives from the rest of the state.
- [x] **Delay-line wiring** (`wiring.py`): a shift register (25 → 22 → 19 →
      24 → 30 → 2 → 26 → 15 → 20) written by the event detectors, with link
      signs following polarity. Most single-link cuts cost 6-16% accuracy.
- [x] **Shift prediction** (`shift_test.py`): error rates of four blind-spot
      classes at shift 0 predict the error at shifts −1.5 to +2 (e.g. 0.178
      vs 0.181 at −1, 0.0082 vs 0.0085 at +2).
- [x] **Readout generality** (`readout_generality.py`): H9-H11 hold in all
      five 32x10 seeds (all 20 shift-register links follow the sign rule);
      the 16x10 models fit less well.

### Step 7: Optional interactive viewer

Only if static plots become limiting: a Streamlit app under `app/` to step
through hidden states for chosen inputs, or to explore cones of small models.

### Step 8: Extensions

- Argmedian and median RNNs. Most tooling carries over; median uses
  `ScalarRNN` with mean-squared-error loss.
- Longest-cycle transformers: convert the weights into a `HookedTransformer`
  (attention-only, no LayerNorm; split `nn.MultiheadAttention`'s packed
  `in_proj_weight`) and use attention-pattern and OV/QK analysis.

## Milestones

1. ~~The hooked RNN matches the model exactly, and the tools recover the
   handcrafted 10-step model's neurons~~ (step 1, done).
2. ~~Causal neuron dictionary for 32x10~~ (steps 2-3, done; clock causal at
   group level via `clock_test.py`).
3. ~~Roles compared across seeds, 16x10 and smaller models~~ (step 4, done).
4. A small model passes all four explanation checks (step 5: 4x3 passes
   three; the compact rebuild reaches 98.8%, just short of 99%).
5. 32x10 passes all four explanation checks (step 6: rebuild and resample
   pass; accuracy estimate partial).
