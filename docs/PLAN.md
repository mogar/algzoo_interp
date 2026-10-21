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

- [ ] `algzoo_interp/features.py`: running top-k values, events (new max,
      new top-2, with and without the `max(0, ...)` convention), steps since
      last event, shifted inputs `x_{t-k}`.
- [ ] `scripts/neuron_dictionary.py`: for every neuron, best-matching event
      and best single and multiple-regression fit of running values; writes a
      table and figure per model. Supersedes the hard-coded parts of
      `neuron_hypotheses.py`. Supports `--seed`, `--n-seqs`, `--example`,
      `--handcrafted`.
- [ ] Run on `handcrafted_2nd_argmax(10)`.

**Done when** the dictionary labels the handcrafted neurons as their source
comments describe (shift register 0-9, the neurons 18-21 quantities, the
"max minus current" family 10-17), giving at least R² ≈ 1 / 100% agreement
where the definition is exact.

### Step 2: Causal tests of the 32x10 event detectors, resets and veto (H1, H2, H5, H6)

- [ ] Tool: **edge ablation**. Run the model with chosen individual weights
      zeroed (or their contribution removed) without touching the rest.
      Covers the "path patching" item deferred from Phase 1 for the cases we
      need.
- [ ] **H1/H2.** Patch neuron 7 (or 16) at a single step, from an input where
      the event did or didn't happen. Measure how the prediction changes and
      which neurons respond at the next step.
- [ ] **H5 resets.** Measure target neurons (29, 13, 0, ... for 7; 3, 25 for
      16) at t+1, conditioned on the source firing at t; then patch the source
      and check the targets are forced to zero.
- [ ] **H6 veto.** Remove the 7 → logit 9 contribution only and re-measure
      accuracy split by argmax position. Prediction: errors appear mainly when
      the max is at position 9. Same for 12 → logit 9 to find its role.

**Done when** each of H1, H2, H5, H6 is marked causal or refuted in
HYPOTHESES.md, with the script and numbers cited.

### Step 3: Memory and timing neurons in 32x10 (H3, H4)

- [ ] **H3.** Run the step 1 dictionary on 32x10. Fit always-on neurons
      (5, 8, 17, 31) against several running values at once. Causal test:
      does shifting neuron 17 by δ act like raising the stored max?
- [ ] **H4.** For neurons that are neither event detectors nor always on,
      plot value against "steps since last new max" and "steps since last new
      top-2". Find what encodes *position*.
- [ ] Neuron 20's drop at the last step; neuron 8; neuron 12's role.

**Done when** every neuron with a non-negligible ablation effect has a
dictionary label with causal support, or is explicitly listed as unexplained.

### Step 4: Generality across seeds and sizes

- [ ] Run the dictionary on 32x10 seeds 0-4 and on `example_2nd_argmax()`
      (16x10). Do event detectors, memory and reset neurons appear under
      different neuron numbers?
- [ ] Run on a few smaller models (hidden 4-8, seq_len 4-6) to see which
      roles survive at small sizes and what the low-accuracy models lack.
- [ ] Optional: training checkpoints (`n_seqs`) of 32x10 to see the order in
      which roles appear.
- [ ] Compare the 32x10 mechanism with the handcrafted model's scheme.

**Done when** there's a table of roles × models and a short note on what is
universal and what is seed-specific.

### Step 5: Complete explanation of a small model (methods track)

Use 4x3 (98.4%) or 5x3 (99.9%), where cones can be listed exactly, to build
and debug the explanation checks before applying them to 32x10.

- [ ] Explain every common cone: its linear map, where the decision boundary
      falls inside it, and the role of the alternating sign-flipped pairs.
- [ ] Hand-built rebuild; resample check.
- [ ] **Accuracy estimate from the mechanism**: compute cone volumes (solid
      angles) and the correct fraction within each, and compare with measured
      accuracy.
- [ ] Write-up in `results/2nd_argmax_4x3/README.md` (or 5x3).

**Done when** all four explanation checks from [Method](#method-how-each-model-gets-analysed-and-judged)
pass on the small model.

### Step 6: Quantitative explanation of 32x10

- [ ] Hand-built rebuild from the step 2-3 explanation; target ≥ 99% argmax
      agreement with the real model.
- [ ] Resample check over everything the explanation calls irrelevant.
- [ ] Mechanistic accuracy estimate, including where errors fall (the worst
      cell is argmax 8 / 2nd argmax 9, 98.5%).
- [ ] Write-up in `results/2nd_argmax_32x10/README.md`.

**Done when** all four explanation checks pass, with the unexplained gap
stated as a number.

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

1. ~~The hooked RNN matches the model exactly~~ (done), and the tools recover
   the handcrafted 10-step model's neurons (step 1).
2. Causal neuron dictionary for 32x10 (steps 2-3).
3. Roles compared across seeds, 16x10 and smaller models (step 4).
4. A small model passes all four explanation checks (step 5).
5. 32x10 passes all four explanation checks (step 6).
