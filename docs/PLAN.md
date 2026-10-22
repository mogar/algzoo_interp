# Plan: interpreting the AlgZoo 2nd argmax RNNs

Goal: a mechanistic explanation of the 2nd argmax RNNs from
[AlgZoo](https://github.com/alignment-research-center/alg-zoo), up to
`hidden_size=32, seq_len=10` (1,376 parameters). Explanations are checked
quantitatively, not just told as stories.

This file is the single source for **what we do next**
([Open questions](#open-questions)) and how far each step got
([Roadmap](#roadmap)). The findings themselves live in the write-ups and
[HYPOTHESES.md](HYPOTHESES.md); this file links to them rather than
repeating them. For repo layout, setup and conventions, see the
[README](../README.md).

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
  {2, 3, 4, 5, 6, 8, 10}, with five seeds each and checkpoints at
  `n_seqs` = 0, 2^20, ..., 2^30. By default `zoo_2nd_argmax` loads the
  best-of-5 seed.
- `example_2nd_argmax()` is the 16x10 model (432 parameters) discussed in
  ARC's blog post.
- `handcrafted_2nd_argmax(seq_len)` gives hand-built solutions for seq_len
  2, 3 and 10. The 10-step one (22 neurons) has commented neuron meanings,
  such as shift-register copies of `x_{t-k}`, the running max, and "max minus
  current", which are candidate features to look for.

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

## Method: how each model gets analysed and judged

**Analysis toolkit.**

- **Neuron dictionary.** For each neuron, test candidate *events* (does
  on/off match "new running max", "new top-2", ...?) and candidate *running
  values* (`x_{t-k}`, max so far, 2nd max so far, gaps, steps since an
  event).
- **State analysis.** How much of each neuron the ideal online state explains
  (the max, the 2nd max and their lags), fitted separately within each
  position cell.
- **Weight structure.** Large entries (resets, vetoes) and chains such as
  shift registers in `W_hh`.
- **Readout.** Per-neuron logit contributions, and position codes in the
  final state.
- **Causal checks.** Every labelled claim gets an intervention: patching
  neurons at one step, resampling, or removing individual weights.
- **Generality.** The same analysis on other seeds and nearby model sizes.

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

## Status

| check | 4x3 ([write-up](../results/2nd_argmax_4x3/README.md)) | 32x10 ([write-up](../results/2nd_argmax_32x10/README.md)) |
|---|---|---|
| Rebuild (≥ 99% agreement) | partial: ratio readout 99.78%, two-number formula 98.84% | pass for correct behaviour: 99.44%, exactly the inputs the model gets right |
| Resample check | pass: 99.76% | pass: 99.17%, at the (accuracy)² ceiling |
| Accuracy without sampling | pass: exact 0.98455 vs sampled 0.98443 ± 0.00012 | partial: 80% of errors attributed to two mechanisms, which also predict errors on shifted inputs; no sampling-free estimate |
| Write-up | done | done |

## Roadmap

Steps 1-4 built the 32x10 explanation. Step 5 developed the explanation
checks on a model small enough to do exactly, and step 6 applied them to
32x10.

### Step 0: Setup and tools — done

`alg-zoo` installed in editable mode from the submodule, with checkpoints
cached in `weights/`. The library has a hooked, unrolled RNN, input probes,
accuracy breakdowns, ablation and patching, and activation-pattern tools
(`hooked.py`, `data.py`, `eval.py`, `interventions.py`, `patterns.py`).
`zoo_baseline.py` gives the accuracy of every 2nd argmax model.

**Outcome.** For 10-step models almost every input has its own activation
pattern (GEOMETRY.md §7), so they need per-neuron and per-step
descriptions rather than cone-by-cone ones.

### Step 1: Feature library, validated on the handcrafted model — done

`features.py`, `dictionary.py` and `scripts/neuron_dictionary.py`.

**Outcome.** On `handcrafted_2nd_argmax(10)`, all 22 neurons are recovered
with R² = 1.000, each matching its source comment (`tests/test_dictionary.py`).
In the handcrafted model's 0-indexed terms, after reading `x_t`:
- neuron `9-k` is `relu(x_{t-k})`;
- neuron 19 is `max0 - max0_prev`, how much the running max just increased;
- neuron 20 is `max0_prev`;
- neuron 21 is `max0 - x_t`;
- neuron `19-k` (k = 1..9) is `max0` over everything except `x_{t-k}`,
  minus `x_t`.

Here `max0 = max(0, x_0..x_t)`. Run on the trained 10-step models
(`results/<model>/dictionary/`):

| | 32x10 | 16x10 (blog example) |
|---|---|---|
| Neurons that match one handcrafted neuron (single-feature R² ≥ 0.95) | 7 = hc 19 (1.000), 17 = hc 20 (0.986) | 4 = hc 19 (0.994), 2 = hc 20 (0.974), 7 = hc 21 (0.986), 6 = hc 18 (0.972) |
| Linear fit on the 22 handcrafted neurons: R² ≥ 0.99 / ≥ 0.9 / median | 3 / 10 of 32 / 0.83 | 6 / 13 of 16 / 0.95 |
| Neurons whose ablation costs ≥ 10% accuracy | 23, of which 10 have handcrafted-basis R² ≥ 0.9 | 16, of which 13 have handcrafted-basis R² ≥ 0.9 |

Both trained models rebuilt the handcrafted running-max register. The blog's
16x10 model is close to a compressed version of the whole handcrafted
scheme, while 32x10 only partly follows it.

### Step 2: Causal tests of the 32x10 event detectors, resets and vetoes — done

`ablate_edges` in `interventions.py`, and `scripts/causal_tests.py`.

**Outcome.** The event detectors (H1, H2), resets (H5) and both vetoes on
position 9 (H6, H6b) are causal. See
[HYPOTHESES.md §A-B](HYPOTHESES.md#a-events-and-stored-values).

### Step 3: Memory and timing neurons in 32x10 — done

`state.py`, `scripts/state_analysis.py` and `scripts/lag_codes.py`.

**Outcome.** The hidden state is mostly the minimal online state. The max's
lag is a clock restarted by every new max, and the running values sit in
always-on neurons (H3, H4). Every neuron has a role in the
[table in HYPOTHESES.md §F](HYPOTHESES.md#f-neuron-roles-in-32x10). Neurons
8, 4 and 9 are not explained.

### Step 4: Generality across seeds and sizes — done, except checkpoints

`scripts/compare_models.py` (13 models), causal tests on 32x10 seeds 1-4,
and `scripts/summarize_causal.py`.

**Outcome.** The mechanisms recur in all five 32x10 seeds, but in different
neurons. The two 16x10 models use two different solutions, and the smaller
models lack parts of the machinery. See
[HYPOTHESES.md §G](HYPOTHESES.md#g-generality-other-seeds-and-model-sizes).
An explanation of seed 0 transfers to other seeds as mechanisms, not neuron
by neuron.

- [ ] Optional: training checkpoints (`n_seqs`) of 32x10, to see the order in
      which roles appear.

### Step 5: Complete explanation of a small model (methods track) — done

`cones.py`, `scripts/explain_small.py` and `scripts/exact_accuracy.py`, on
the 4x3 model. Write-up:
[results/2nd_argmax_4x3/README.md](../results/2nd_argmax_4x3/README.md).

**Outcome.** Three of the four checks pass and the rebuild is partial (see
[Status](#status)). Exact cone enumeration matches sampling within one
standard error for the 3x3, 5x3 and 8x3 models too.

**Lessons for 32x10.** Exact cone enumeration doesn't scale to 10 steps, but
other pieces carry over:
- finding a symmetry;
- reducing the readout to a few combined quantities and thresholds;
- attributing error to states where neurons that should switch together
  don't.

### Step 6: Quantitative explanation of 32x10 — done, accuracy estimate partial

`scripts/explain_32x10.py`, then `readout_analysis.py` for the readout, and
for the completion checks `clock_test.py`, `wiring.py`, `shift_test.py` and
`readout_generality.py`. Write-up:
[results/2nd_argmax_32x10/README.md](../results/2nd_argmax_32x10/README.md).

**Outcome.**
- **Rebuild and resample checks pass** (see [Status](#status)).
- **The readout** is a matched filter on an additive position code, fed by a
  top-two shift register and a separate code for which top-two input is the
  max (H9-H12).
- **Errors** come from two blind spots, and a model of those blind spots
  predicts the error on shifted inputs (H8).
- **The readout story holds in all five seeds.**
- **Reproducibility.** `pytest -m scripts` smoke-tests every script, and
  `scripts/run_all.sh` reproduces every committed result.

### Steps 9-13: a mechanistic error estimate for 32x10 — current focus

ARC's main challenge
([blog post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/))
is a *mechanistic estimate* of a model's accuracy. Such an estimate reasons
from the model's structure rather than by sampling. It has to match random
sampling in **mean squared error versus compute**, scored over noisy copies
of the model (noise added to the weights). ARC poses it for the blog's 16x10
model; we target 32x10, where our explanation is strongest.

The plan is to split the error by input class:

**error = Σ over input classes P(class) × P(error | class)**

P(class) is an exact probability of an ordering for Gaussian inputs, e.g.
P(2nd largest < 0) = 11/1024. P(error | class) is the hard part. Near a tie
between the 2nd and 3rd largest the model is piecewise linear, so its
decision boundary sits a computable distance δ from the true tie. The error
is then roughly the density at the tie times |δ|, which turns a rare event
into a smooth quantity.

The estimators form a ladder, each more deductive than the last. **E3 and E4
are the target; E1 and E2 are stepping stones.**

| | estimator | how mechanistic |
|---|---|---|
| E0 | plain sampling | the baseline to beat |
| E1 | exact class probabilities × sampled error within each class | the error split is exact |
| E2 | for each sampled input, the error computed exactly along the tie direction, by walking the activation-pattern breakpoints on that line | exact in one dimension; the mechanism picks the direction |
| E3 | δ predicted from the weights through the explanation (top-2 detector bias, neuron 12, readout margins), integrated over the order-statistic distribution | deductive |
| E4 | a mechanistic model of the max(0, …) blind spot: what the model predicts when the 2nd largest is negative | deductive |

**Housekeeping.**
- [ ] Split `scripts/run_all.sh` into an explanatory phase (steps 1-6:
      dictionary, causal tests, readout and so on) and a challenge phase
      (steps 9-13: the estimation harness and estimators), so each can be
      rerun on its own.

**Step 9: scoring harness — done.**
- [x] `algzoo_interp/estimation.py`: a family of noisy-weight copies
      (`w + σ·|w|·ε`), high-precision ground-truth error for each copy
      (cached), compute counted in multiply-adds, and MSE against compute
      for any estimator.
- [x] `scripts/estimation_harness.py`: the sampling baseline (E0) on the
      family, with the measured line checked against p(1 − p)/N.
- [x] Check the harness on 4x3, where the exact cone method should beat
      sampling by orders of magnitude.

**Done when** the sampling line matches theory and 4x3's exact method sits
far below it.

**Outcome** (`results/<model>/estimation/summary.json`):
- **The family.** It is the model itself plus 8 noisy copies at each of
  σ = 0.001 and 0.003. 32x10 is very sensitive to weight noise: its error of
  0.56% rises to 0.60-0.87% at σ = 0.001 and 0.82-1.53% at σ = 0.003. Each
  ground truth uses 2^25 inputs, which leaves a variance floor of 2.9×10⁻¹⁰.
- **The sampling baseline matches theory.** On 32x10 the measured MSE is
  0.99-1.02 times p(1 − p)/N at every budget from 2^8 to 2^20 inputs. One
  forward pass is 10,880 multiply-adds.
- **The 4x3 check passes.** The exact cone method costs about 6×10⁴
  multiply-adds (about 830 samples' worth) and has zero error. Sampling at
  that compute has an MSE of about 2.5×10⁻⁵. The sampled ground truths
  agree with the exact values to within their standard errors.

**Step 10: E1, exact class probabilities — done.**
- [x] `algzoo_interp/order_stats.py`: exact probabilities of cells of 10
      Gaussian inputs, from a 1-D integral over the 2nd largest, with exact
      conditional samplers. There are 92 cells: max < 0, only the 2nd < 0,
      and a grid over the relative gaps g12 = (max − 2nd)/max and
      g23 = (2nd − 3rd)/2nd.
- [x] E1 (`Stratified` in `estimation.py`), scored by the harness. The
      budget is split across cells by a fixed Neyman allocation from a pilot
      on the unperturbed model.

**Outcome** (`results/2nd_argmax_32x10/estimation/{summary.json, cells.md}`):
- **E1 beats sampling by 4.4-7.5× in MSE** at every budget, on the noisy
  family as well as the original. The pilot predicted 5.7×.
- **A third blind spot.** With the classes in H8 alone, the gain would be
  only about 2.8×. The rest comes from splitting on g12: near-ties between
  the max and the 2nd also cause errors. The model then predicts the max's
  position, which fits finite resolution in the identity code (H12).
- **Where the error is,** from the unperturbed model's cell rates:

  | errors | share |
  |---|---|
  | negative values | 41% |
  | 2nd-3rd near-ties (g23 < 0.03) | 39% |
  | max-2nd near-ties only (g12 < 0.01, g23 ≥ 0.03; 1.6% of inputs) | 14% |
  | none of these | 6% |

- **What limits E1.** The two negative cells hold 41% of the error, and
  E1 can't split them further without a mechanism (E4). Within the
  near-tie cells the error is still a coin flip per sample, which is what
  E2 and E3 remove.

**Step 11: E2, exact error along the tie direction — done; no gain over E1.**
- [x] `algzoo_interp/lines.py`: the exact φ-weighted error along a line,
      found by walking the activation-pattern breakpoints (each step is a
      forward pass with derivatives). Also the tie lines through an input
      (`tie_lines`) and the part of a line inside an E1 cell
      (`cell_interval`). The integral converges to the walk's value as a
      brute-force grid is refined (`tests/test_lines.py`).
- [x] E2 (`CellLines`): E1's cells, where each cell's error comes from 0/1
      samples or from exact integration along the 2nd-3rd or max-2nd tie
      line through each sample, whichever the pilot finds cheaper. It is
      unbiased because, given the rest of the input (the line's offset y),
      the input's own position on the line is N(0, 1) restricted to the
      in-cell segment.

**Done when** E2's MSE versus compute is measured against sampling on the
noisy family.

**Outcome** (`results/2nd_argmax_32x10/estimation/summary.json`):
- **E2 beats sampling by 3.6-7.4×, the same as E1.** The pilot sends only
  15 of 92 cells to line integration.
- **Why integrating along s doesn't help:** the tie error is set by the
  rest of the input, not by the position along the line. Integrated over
  the relative near-tie window (γ = 0.03), the per-line error has a
  standard deviation 6.2 times its mean (`e2_tie_window`). Most lines have
  almost none and a few have a lot. Each line costs about 21 forward passes.
  So the remaining noise comes from *which lines* have a large boundary
  offset δ, which is exactly what E3 must predict.
- **Heavy tails broke the first design.** A few lines are wrong along their
  whole segment, or on exactly one side of the tie (the model ignores the
  pair's order). A 1,024-input pilot missed them, underestimated those
  cells' spread, and starved them of inputs, so E2's MSE stopped falling at
  large budgets. An 8,192-input pilot, and a 20% margin before choosing
  line integration, fixed it.
- **An exact-tie bug, found on 4x3.** When every neuron is off, all
  logits are 0 and `argmax` returns position 0, but the walk first counted
  such ties as correct, biasing E2. It now follows argmax's lowest-index
  rule (`tests/test_lines.py`). The 4x3 E2 numbers are from the fixed code.
  The 32x10 numbers came from the code before the fix, which only differs
  on exact logit ties; 32x10 is essentially never all-off. On 4x3, E2 gains
  1.6-2.0× from 1,024 inputs up, about the same as E1. At 256 inputs it
  loses (0.14×), because the one-input-per-cell minimum leaves cells with
  sizeable probability on a single 0/1 sample.
- **Leads for E3/E4.** Those "blind" lines are rare (about 1 in 3,000 in
  the low-error cells checked). They often have the last input in the top
  three, or the max at position 8 or 9, or a 2nd largest just above zero.

**Step 12: E3, the near-tie error from the mechanism.**
- [ ] Use E2's exact boundary offsets to fit and then *derive* a model of δ
      as a function of the explanation's quantities: the lags of the max,
      2nd and 3rd, the values, and the scale. A model of the top-2 detector
      alone over-predicts errors tenfold (H8), so several neurons decide
      near-ties together.
- [ ] Compute δ from the weights (through the detector, neuron 12 and the
      readout margins), then integrate over the order-statistic
      distribution without sampling.

**Done when** a fully deductive near-tie estimate beats sampling on MSE
versus compute.

**Progress: the offset is measured, and it isn't simple**
(`python scripts/tie_offsets.py --hidden 32 --seq-len 10`,
`results/2nd_argmax_32x10/ties/summary.json`):
- **The first-order switch point is usually exact.** One forward pass with
  derivatives at the tie gives s* = −ΔL/ΔL′. It matches the exact walk on
  75% of 2nd-3rd tie lines and captures 89% of the mean tie error. The
  rest is pattern changes near the tie or a third logit.
- **Positions explain almost nothing.** The (max, A, B) arrival positions
  explain 2% of the per-line tie error. The arrival order doesn't decide
  the error either: when the max arrives last, so that A against B is a
  new-max comparison made by an exact detector, the tie error is no smaller
  (0.00137, against 0.00142 otherwise).
- **The last input is the exception.** When the later element of the pair
  is the last input, the tie error rate is about 4× higher (0.0055 against
  0.0014). That case is 20% of inputs and about 45% of the tie error.
- **Offsets are small, roughly centred, noise-like.** The interquartile
  range is about ±0.1% relative, wider when the last input is involved. Fit
  separately within each position cell, they are not a simple function of
  the values:

  | description, fitted per position cell | R² of the offset |
  |---|---|
  | the cell's mean | 0.13 |
  | affine in max/mid, 4th/mid and last/mid | 0.33 |
  | linear in all ten inputs over mid (held out) | 0.45 |

  The offset depends on which neurons happen to be on, so a closed-form δ
  from a few explanation variables would leave most of it unexplained.
- **Integrating out the last input exactly** (given x_0..x_8, every step-9
  pre-activation is affine in x_9) leaves a third of the variance (3.0×
  reduction). Two-thirds of the error is in the piece where the last input
  is below the running 2nd: those errors are already fixed by the first
  nine inputs.

**What this means for E3.** The near-tie error is the leftover imprecision of
many neurons. It isn't one biased threshold, so it can't be derived as a
handful of per-case constants. Candidate routes, still to be chosen:

1. **Moment propagation (deductive).** Treat the offset as noise. Get its
   distribution in each position case by propagating means and
   covariances of the hidden state through the network, with the
   explanation saying which neurons are on in that case. The near-tie
   error is then Σ_cases P(case) × (tie density) × E|δ|. Start with the
   last-input case, a single-step decision: the offset is a linear readout
   of h_8 minus the true value.
2. **Last step exact, prefix sampled (semi-mechanistic).** Integrate x_9 out
   in closed form (one step, so about 1-2 forward passes), and stratify the
   first nine inputs like E1. It is a better sampler, not a deduction.
3. **A realistic bar for "deductive".** A deterministic estimate with bias
   b beats sampling wherever b² < p(1 − p)/N. For 32x10 (p = 0.0056), a
   10% bias matches sampling up to about 18,000 samples (2×10⁸
   multiply-adds), if the estimate is much cheaper than that. That would
   make routes 1 and E4 worth scoring even if they are only accurate to
   10-20%.

**Step 13: E4, the negative-value blind spot, and the full estimate.**
- [ ] Derive what the model predicts when the 2nd largest (or the max) is
      negative, and the error that follows. At shift 0 the measured error is
      67% when both are negative and 17.4% when only the 2nd is.
- [ ] Combine E3, E4 and the remaining classes into one estimate, then score
      it. Optionally, run ARC's surprise accounting.

**Done when** the combined estimate is scored, and any gap to sampling is
attributed to a named class.

### Step 7: Optional interactive viewer — not started

Only if static plots become limiting: a Streamlit app under `app/` to step
through hidden states for chosen inputs, or to explore cones of small models.

### Step 8: Extensions — not started

- Argmedian and median RNNs. Most tooling carries over; median uses
  `ScalarRNN` with mean-squared-error loss.
- Longest-cycle transformers: convert the weights into a `HookedTransformer`
  (attention-only, no LayerNorm; split `nn.MultiheadAttention`'s packed
  `in_proj_weight`) and use attention-pattern and OV/QK analysis.

## Open questions

Roughly in order of how much each would add to the 32x10 explanation:

1. **A mechanistic accuracy estimate** for 32x10, the one explanation check
   that doesn't pass. This is the current focus:
   [steps 9-13](#steps-9-13-a-mechanistic-error-estimate-for-32x10--current-focus).
2. **The identity code, neuron by neuron.** H12 shows which neurons say
   which top-two input is the max. It doesn't show how each reference
   neuron's lag-dependent mix combines with the clock to do that at each lag.
3. **The clock's internal wiring.** It re-derives itself each step from the
   reference neurons and neuron 7 (H4). How it does so isn't written out.
4. **How membership is revoked** when an input drops out of the top two
   (H11).
5. **The lag of the 2nd max**, which is encoded less cleanly than the lag of
   the max.
6. **Neurons 8, 4 and 9**, which have no explained role.
7. **Why the 16x10 models fit the readout story less well** (HYPOTHESES.md §G).
8. **Training checkpoints** (step 4), steps 7 and 8.

## Milestones

1. ~~The hooked RNN matches the model exactly, and the tools recover the
   handcrafted 10-step model's neurons~~ (step 1).
2. ~~A causal neuron dictionary for 32x10~~ (steps 2-3, plus the clock and
   shift register in step 6).
3. ~~Roles compared across seeds, 16x10 and smaller models~~ (step 4).
4. A small model passes all four explanation checks. The 4x3 model passes
   three, and its compact rebuild reaches 98.8%, just short of 99%.
5. 32x10 passes all four explanation checks. Rebuild and resample pass; the
   accuracy estimate is partial.
