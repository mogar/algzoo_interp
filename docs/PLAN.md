# Plan: interpreting the AlgZoo 2nd argmax RNNs

Goal: a mechanistic explanation of the 2nd argmax RNNs from
[AlgZoo](https://github.com/alignment-research-center/alg-zoo), ideally all
the way up to `hidden_size=32, seq_len=10` (1,376 parameters). Explanations
are checked quantitatively, not just told as stories.

## Background

### The models

Every 2nd argmax model is an `alg_zoo.DistRNN`:

```
h_0 = 0
h_t = ReLU(W_ih * x_t + W_hh @ h_{t-1})      t = 1..seq_len, x_t is a scalar
logits = W_out @ h_seq_len                    one logit per input position
```

- No biases anywhere. Parameters: `hidden + hidden^2 + seq_len * hidden`.
- Trained on `x ~ N(0, I)` (`th.randn`) with cross-entropy against
  `x.argsort()[-2]`.
- The zoo covers `hidden_size` in {2, 3, 4, 5, 6, 8, 16, 32} and `seq_len` in
  {2, 3, 4, 5, 6, 8, 10}, five seeds each, with checkpoints at
  `n_seqs` = 0, 2^20, ..., 2^30. By default `zoo_2nd_argmax` loads the
  best-of-5 seed.
- `example_2nd_argmax()` is the 16 x 10 model (432 parameters) discussed in
  the blog post.
- `handcrafted_2nd_argmax(seq_len)` for seq_len 2, 3 and 10 are hand-built
  solutions. The 10-step one (22 neurons) has commented neuron meanings, such
  as shift-register copies of `x_{t-k}`, the running max, and "max minus
  current", which gives us candidate features to look for.

### Structural facts we can exploit

1. **Piecewise linear.** With ReLU and no biases, fixing which neurons fire
   at each step makes `h_T`, and so the logits, an exact linear function of
   `x`. The input space splits into polyhedral cones, each with one exact
   linear map.
2. **Scale invariant.** For any `c > 0`, `f(c x) = c f(x)`, so the predicted
   argmax depends only on the direction of `x`. Inputs effectively live on
   the unit sphere.
3. **Only order matters to the task.** So the model has to approximate
   comparisons using linear maps and ReLUs, and its errors should
   concentrate where comparisons are hard, such as near-ties.

### Tooling choice

TransformerLens's `HookedTransformer` doesn't fit RNNs. We unroll the RNN by
hand in PyTorch and expose intermediate values, optionally through
TransformerLens's `HookPoint` and `HookedRootModule`. `HookedTransformer`
only becomes relevant for the longest-cycle transformers (Phase 5).

No Jupyter. Experiments are plain Python scripts that save image plots. If we
need interactive views, we build a Streamlit app.

## Repo layout

```
algzoo_interp/   reusable library code (loading, data, eval, hooks, plots)
scripts/         one runnable script per experiment; writes to results/
results/         figures, CSV/JSON numbers, and per-model writeups (markdown)
tests/           pytest checks for the library
app/             (later, if needed) Streamlit viewer built on algzoo_interp
weights/         local cache of downloaded checkpoints (gitignored)
docs/            this plan and other notes
```

## Phases

### Phase 0: Setup

- [x] Install `alg-zoo` in editable mode from the submodule (`-e ./alg-zoo`
      in `requirements.txt`).
- [x] Add `.gitignore`, `pyproject.toml`, and the package skeleton.
- [x] Cache weights locally under `weights/`.
- [x] Smoke test: load every 2nd argmax model and measure its accuracy;
      save a baseline CSV and heatmap to `results/zoo_baseline/`.

### Phase 1: Tools

- [x] **Hooked RNN** (`hooked.py`). Unroll by hand and expose, for each step
      t, pre-activations, `h_t`, ReLU masks and the final logits, with
      optional hooks on `pre` and `h`. Also per-neuron logit contributions.
      Tested to match `model(x)` exactly.
- [x] **Datasets** (`data.py`). `randn` inputs matching training, plus
      controlled probes: fixed (argmax, 2nd-argmax) positions, a chosen
      top-two gap (near-ties), sorted sequences, and one position swept.
- [x] **Evaluation** (`eval.py`). Accuracy overall, by (argmax position,
      2nd-argmax position), and by top-two gap.
- [x] **Interventions** (`interventions.py`). Zero, mean and resample
      ablation of any neurons at any steps; activation patching from a
      source input.
- [ ] Path patching through time (patching only one route of influence).
      Deferred until a concrete hypothesis needs it.
- [x] **Activation-pattern analysis** (`patterns.py`). Count patterns and
      their coverage, and extract the exact linear map for any pattern
      (tested to reproduce the logits exactly).
- [x] **`scripts/inspect_model.py`**: first-look figures for any model
      (weights, accuracy breakdowns, activity, pattern coverage, single-neuron
      ablations, example traces) in `results/<model>/inspect/`.

First observations (2^18 randn inputs):

| Model | Accuracy | Distinct patterns | Patterns covering 90% |
|---|---|---|---|
| 4x3 | 98.4% | 67 | 13 |
| handcrafted 10 | 99.0% | 65,851 | 39,637 |
| example 16x10 | 95.3% | 174,722 | 148,508 |
| 32x10 | 99.5% | 257,562 | 231,348 |

- Listing whole-sequence activation patterns is practical only for the small
  models. For 10-step models nearly every input has its own pattern, even for
  the handcrafted model. There we need per-step or per-neuron descriptions,
  not region-by-region ones.
- 4x3: the neurons form two sign-flipped pairs, (0, 3) and (1, 2), and the
  active pair alternates between steps.
- 32x10: `W_hh` is mostly near zero apart from strong negative columns from
  neurons 7 and 16, which inhibit many other neurons. Neuron 7 also has the
  largest input weight and a large negative weight on logit 9. Ablating 7,
  16 or 17 costs 60-90% accuracy. Ablating 9 or 10 costs almost nothing, and
  ablating 4, 14 or 24 costs little.

### Phase 2: Model ladder

Validate the tools on known answers first, then scale up.

| Step | Model | Goal |
|---|---|---|
| a | handcrafted, seq_len 2, 3, 10 | Tools recover the known neuron meanings |
| b | trained 2x2, 4x3 | Full explanation by hand; draw every region |
| c | trained, hidden 4-8, seq_len 4-6 | Find which features recur as length grows |
| d | `example_2nd_argmax()` (16x10) | Match or extend the blog's partial explanation |
| e | **32x10** | Main target |

Comparing neighbouring models in the grid shows which mechanisms are general
and which belong to one seed.

### Phase 3: Analysis for each model

- **Neuron dictionary.** Fit each neuron's `h_t` against candidate features:
  `x_{t-k}`, the running max, the running max minus the current input, "is
  position k still the max", and ReLU-clipped versions. Record R^2 and the
  leftovers. Label neurons as dead, always-on (so linear) or switching.
- **Weight structure.** Look at `W_hh` for permutations, delay lines and
  shift registers, and for low-rank pieces. Eigen-decompose the linear
  sub-system on each active neuron set.
- **Readout.** Split each final logit into one contribution per neuron
  through `W_out`.
- **Causal checks.** Every labelled claim about a neuron gets an ablation or
  patching test.
- **Seeds and checkpoints.** Use `seed=0..4` to check whether runs reach the
  same mechanism, and `n_seqs=2**20 ... 2**30` to see when features appear.

### Phase 4: Check explanations quantitatively

1. **Rebuild by hand.** Write a simplified model from the explanation and
   compare it with the original. Target: at least 99% matching argmax on
   random inputs, with failure regions described.
2. **Resample checks.** Change everything the explanation calls irrelevant
   and confirm the output holds up (causal-scrubbing style).
3. **Estimate accuracy without sampling.** Use the explanation (region
   volumes, failure conditions) to predict accuracy, and compare with the
   measured accuracy. This is ARC's own bar for an explanation.
4. **Write it up** in `results/<model>/README.md` with figures, claims,
   evidence, and the remaining unexplained gap as a number.

### Phase 5: Extensions

- Argmedian and median RNNs. Most tooling carries over; median uses
  `ScalarRNN` with mean-squared-error loss.
- Longest-cycle transformers: convert the weights into a `HookedTransformer`
  (attention-only, no LayerNorm; split `nn.MultiheadAttention`'s packed
  `in_proj_weight`) and use attention-pattern and OV/QK analysis.

## Milestones

1. The hooked RNN matches the model exactly, and the tools recover the
   handcrafted 10-step model's neurons.
2. The 2x2 and 4x3 models are fully explained, with hand-built versions
   matching at least 99%.
3. A neuron dictionary for 16x10, compared with the blog's account.
4. A 32x10 explanation with a rebuilt model and an accuracy estimate made from
   the mechanism.
