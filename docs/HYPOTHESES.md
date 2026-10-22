# Hypotheses and evidence: 2nd argmax 32x10

The detailed evidence for every claim about how the 32x10 model (1,376
parameters, 99.5% accurate) works. For the story without the evidence, read
[results/2nd_argmax_32x10/README.md](../results/2nd_argmax_32x10/README.md)
first. For what happens next, see the roadmap in [PLAN.md](PLAN.md#roadmap).

Neuron numbers refer to the default 32x10 model (seed 0). Other seeds put the
same roles in different neurons (§G). The H numbers are stable labels in the
order the hypotheses were proposed, so they aren't in order within each
section.

**Status key.** *Causal*: confirmed by ablation, patching or removing
weights. *Correlational*: the neuron's behaviour matches the claim on random
inputs, but we haven't shown the model *uses* it that way.

## At a glance

| | claim | status |
|---|---|---|
| [H1](#h1-neuron-7-detects-a-new-positive-running-max) | Neuron 7 fires at a new positive running max and outputs how much it rose | causal |
| [H2](#h2-neuron-16-detects-a-new-positive-top-2-value) | Neuron 16 fires at a new positive top-2 value | causal |
| [H3](#h3-always-on-neurons-hold-running-values) | Always-on neurons hold the running max (17), running 2nd (31) and their gap (1, 5) | causal for 17; correlational for the rest |
| [H4](#h4-a-reset-started-clock-encodes-how-long-ago-the-max-arrived) | A clock restarted by every new max encodes how long ago the max arrived | causal at group level |
| [H5](#h5-neurons-7-and-16-reset-stale-state) | Neurons 7 and 16 zero stale state through large negative recurrent weights | causal |
| [H6](#h6-neuron-7-vetoes-position-9) | Neuron 7 → logit 9 vetoes the last position when it is the new max | causal |
| [H6b](#h6b-neuron-12-vetoes-position-9-when-x_9-is-too-small) | Neuron 12 → logit 9 vetoes the last position when it is below the 2nd max | causal |
| [H7](#h7-why-the-important-weights-are-the-large-ones) | Large weights implement the discrete logic | rationale, not a test |
| [H8](#h8-errors-come-from-three-blind-spots) | Errors come from three blind spots: negative values, 2nd-3rd near-ties and max-2nd near-ties | correlational; predicts errors on shifted inputs |
| [H9](#h9-the-final-state-is-an-additive-position-code-read-by-a-matched-filter) | The final state is an additive position code, read by a matched filter | correlational, plus surrogate test |
| [H10](#h10-errors-are-value-dependent-noise-overwhelming-the-position-margin) | Errors are value-dependent noise that outweighs the position margin | correlational |
| [H11](#h11-a-top-two-shift-register-carries-the-2nd-maxs-position) | A top-two shift register carries the 2nd max's position | causal |
| [H12](#h12-a-separate-code-says-which-top-two-input-is-the-max) | A separate code says which top-two input is the max | causal at group level |
| [H13](#h13-the-errors-are-crosstalk-from-irrelevant-inputs) | The errors are crosstalk: irrelevant inputs perturb hard comparisons through many neurons | causal (redraw test) |

## Sources

Every number below comes from one of these scripts, run with
`--hidden 32 --seq-len 10` unless noted. Outputs are under
`results/2nd_argmax_32x10/`.

| script | output | used for |
|---|---|---|
| `neuron_hypotheses.py` | `hypotheses/summary.json` | event agreement and firing rates (H1, H2), weight statistics (H5-H7) |
| `neuron_dictionary.py` | `dictionary/dictionary.md` | value fits (H1, H3, H6b), ablation drops (§F) |
| `causal_tests.py` | `causal/` | patching and weight removal (H1-H3, H5, H6, H6b) |
| `state_analysis.py` | `state/summary.json` | state R² (H4, §F), final-step flags (H11) |
| `lag_codes.py` | `lag_codes/` | per-lag on-rates and fits (H3, H4) |
| `clock_test.py` | `clock/summary.json` | clock patching (H4) |
| `explain_32x10.py` | `explain/summary.json` | surrogate, resample and error analysis (H8) |
| `shift_test.py` | `shift/summary.json` | error prediction on shifted inputs (H8) |
| `estimation_harness.py` | `estimation/{cells.md, summary.json}` | error by exact cell, including max-2nd near-ties (H8) |
| `tie_offsets.py` | `ties/summary.json` | where the model switches near ties, and the last-step vote (H13) |
| `crosstalk.py` | `crosstalk/summary.json` | redraw test, linearity and carriers of crosstalk (H13) |
| `readout_analysis.py` | `readout/summary.json` | readout codes and patching (H9-H12) |
| `wiring.py` | `wiring/summary.json` | shift-register weights and link cuts (H11) |
| `compare_models.py`, `summarize_causal.py` (no flags) | `results/compare/{compare,causal}.md` | other seeds and sizes (§G) |
| `readout_generality.py` (no flags) | `results/compare/readout_generality.md` | readout story on other models (§G) |

## A. Events and stored values

![Activity Plot](../results/2nd_argmax_32x10/inspect/activity.png)

The activity plot (`inspect_model.py`) shows, for each neuron and step, the
fraction of inputs where the neuron is on. Step 0 is uniform (all ≈ 0.5):
with no previous state, `pre_0 = W_ih * x_0`, so each neuron is on exactly
when `x_0` has the same sign as its input weight. From step 1 on, some
neurons fire for rarer and rarer events, some are almost always on, and the
rest switch with a pattern that depends on the step.

### H1. Neuron 7 detects a new positive running max

**Claim.** Neuron 7 is on at step t exactly when `x_t > max(0, x_0, ..., x_{t-1})`,
and its value is `max0 − max0_prev`, where `max0 = max(0, x_0..x_t)`. That is,
it outputs how much the positive running max just increased.

**Evidence.**
- On/off agreement with the event is 99.9-100% at every step, and the firing
  rate falls with the event's probability:

  | t | 0 | 1 | 2 | 3 | 5 | 9 |
  |---|---|---|---|---|---|---|
  | P(neuron 7 on) | 0.498 | 0.376 | 0.292 | 0.234 | 0.165 | 0.101 |
  | P(event) | 0.498 | 0.376 | 0.292 | 0.233 | 0.164 | 0.100 |

- The "and > 0" matters: without it, agreement drops to 87% at t=1.
  Comparing against `max(0, ...)` is what a ReLU unit with no bias does
  naturally, and the handcrafted model uses the same convention.
- The value fit `h_t[7] = max0 − max0_prev` has R² = 1.000, the same
  quantity as handcrafted neuron 19.
- It has the largest input weight in the model (`W_ih[7]` = 8.8; the median
  |W_ih| is 0.14).
- **Patching.** Pairs of inputs differ only at step t, where the source's
  `x_t` is raised to a new positive max. Later inputs are identical, so
  patching every neuron at step t gives the source's prediction exactly.
  Patching subsets shows which neurons carry the change. Flip rate is how
  often the patched prediction becomes the source's:

  | patched at step t (t = 2..8) | add a new max | remove a new max |
  |---|---|---|
  | everything except neuron 7 | 1-5% | 16-18% |
  | neuron 7 alone | 51-57% | 29-59% |
  | neurons 7 and 16 | 58-99% | 30-76% |

  Without neuron 7, the rest of the state almost never carries a new max, so
  7 is **necessary**. It isn't sufficient alone: a new max is also a new
  top-2 value, so neuron 16 fires too, and 7 + 16 carry 99% of the effect at
  t=2. Their share falls at later steps as other neurons carry more:
  patching everything except 7 and 16 transfers 12% at t=2 and 45% at t=8.

**Status.** Causal: necessary for the downstream effect of a new max.

### H2. Neuron 16 detects a new positive top-2 value

**Claim.** Neuron 16 is on at step t exactly when `x_t > 0` and `x_t` is one
of the two largest of `x_0..x_t`.

**Evidence.**
- Agreement is 99.6-100% at every step. The firing rate is 0.203 against an
  event probability of 0.199 at t=9.
- It has the second-largest input weight (`W_ih[16]` = 5.7).
- **Patching**, using the same design with the source's `x_t` moved above or
  below the running 2nd max without becoming a new max:
  - patching neuron 16 alone reproduces the source's prediction 83-93% of
    the time at t=2..7 when adding a top-2 value, and 74-89% at t=2..8 when
    removing one;
  - patching everything except 7 and 16 manages only 1-6%;
  - neuron 7 doesn't change in these pairs, so patching it does nothing (0%).

**Status.** Causal: necessary and largely sufficient for the effect of a new
top-2 value.

### H3. Always-on neurons hold running values

**Claim.** Neurons that are on for almost every input act as plain linear
units, which suits carrying a value forward. They hold:
- the running max (17): `max0_prev = max(0, x_0..x_{t-1})`, the same
  quantity as handcrafted neuron 20;
- the running 2nd max (31): `2nd0_prev = max(0, 2nd largest of x_0..x_{t-1})`;
- the gap between the two (1, 5).

**Evidence.**
- Neurons 5, 8, 17 and 31 are on for at least 95% of inputs at the last step.
- Value fits pooled over steps 1-9: neuron 17 ≈ `max0_prev` (R² 0.986);
  neuron 31 ≈ `2nd0_prev` (R² 0.963).
- Neurons 1 and 5 have no single-feature match. Fitted separately for each
  lag of the max (`lag_codes/coefs_lag_max.png`), they are close to a fixed
  multiple of the gap at every lag: neuron 1 ≈ 4.4·max − 4.1·2nd; neuron 5 ≈
  3·max − 2.7·2nd at lags 2-9.
- **Patching.** Pairs of inputs differ only in the *value* of the running max
  of `x_0..x_t`, keeping its position, so every running-order event up to t
  is unchanged. Patching at step t, the table shows how often neuron 7's
  new-max decision at t+1 and the final prediction match the source, over
  pairs where they differ:

  | patched at step t (t = 2..8) | neuron 7 at t+1 matches source | final prediction matches source |
  |---|---|---|
  | neuron 17 alone | 65% (53% → 72% as t grows) | 58% |
  | value neurons 1, 5, 17, 31 | 80% | 77% |
  | everything except 1, 5, 17, 31 | 20% | 21% |

  The remaining 20% is spread over neurons that hold the max at particular
  lags (H4, H11).

**Status.** Causal: neuron 17 is the main carrier of the running max's
value, and together with 1, 5 and 31 it carries 80%. The roles of 31, 1 and
5 individually are correlational.

## B. Resets, vetoes and large weights

Typical weights are small: the median |W_hh| is 0.095 and the median |W_out|
is 1.1. A few are far larger (up to about 320x the median in W_hh, and 60x in
W_out), and almost all of them involve neurons 7 and 16.

### H5. Neurons 7 and 16 reset stale state

**Claim.** When a new max arrives, neuron 7 forces a set of neurons to zero,
clearing state that the new max makes stale. Neuron 16 does the same for a
smaller set when a new top-2 value arrives.

**Evidence.**
- **46.5%** of all recurrent weight magnitude (sum of |W_hh|) leaves neuron
  7, and **10.4%** leaves neuron 16. The next-largest neuron accounts for
  2.6%.
- The largest recurrent weights are all negative and come from these two:
  7→29 (−30.7), 7→13 (−30.5), 7→0 (−24.7), 16→3 (−13.7), 7→28 (−13.7), 7→3
  (−11.6), 7→11 (−10.5), 16→25 (−9.9), and so on. When neuron 7 fires, its
  value is typically about 5, so it pushes its targets' pre-activations down
  by roughly 50-150.
- **Every target is zeroed.** Targets are the neurons with
  `W_hh[target, source] < −5` (`causal/resets.png`). P(target = 0 at t+1 |
  source on at t) is 0.98-1.00 for all 10 targets of neuron 7 and both
  targets of neuron 16, against 0.004-0.76 when the source is off.
- **Cutting the edges shows which resets they cause.** With 7's reset edges
  removed:
  - targets 0, 13, 28 and 29 fall back to their "source off" rates
    (P(0 | on) = 0.30, 0.15, 0.01, 0.23), so neuron 7 causes those resets;
  - targets 3, 10 and 25 stay zeroed (0.998-1.000), so something else zeroes
    them too. For 3 and 25 it is partly neuron 16, which also fires at every
    new max;
  - targets 11, 23 and 27 are in between.
- **The resets matter.** Cutting 7's reset edges drops accuracy from 99.5%
  to 67.4%; cutting 16's two drops it to 72.3%.

**Status.** Causal.

### H6. Neuron 7 vetoes position 9

**Claim.** The largest weight in the model, `W_out` from neuron 7 to logit 9
(**−65.1**), is a veto. If the last input is a new max, it is the largest
value, so it can't be the 2nd largest.

**Evidence.** Removing that one weight (`causal/veto.png`) drops accuracy
from 99.5% to 96.8%. On inputs whose max is at position 9, accuracy falls
from 99.2% to 72.4%, and accuracy at every other max position is unchanged.
86% of the new errors have the max at position 9, and 88% predict position 9.

**Status.** Causal.

### H6b. Neuron 12 vetoes position 9 when x_9 is too small

**Claim.** Neuron 12 computes `relu(2nd0_prev − x_t)`, how far the current
input falls below the running 2nd max. Its readout weight to logit 9
(**−39.7**) rules out position 9 when `x_9` is too small to be the 2nd
largest. Together with H6, logit 9 is vetoed exactly when `x_9` is above the
max (neuron 7) or below the 2nd max (neuron 12).

**Evidence.**
- The value fit has R² = 0.998. The neuron's input weight is −3.1, so it
  reacts to *small* inputs.
- Removing the 12 → logit 9 weight drops accuracy to 94.8%. 94% of the new
  errors wrongly predict position 9, but almost none have the max at 9 (1%).
- In those wrong "9" predictions, `x_9` is usually just below the true 2nd
  max: it is the 3rd largest in 2,250 of 3,219 cases, the 4th in 717 and the
  5th in 192.

**Status.** Causal for the veto; the value fit is correlational but near
exact.

### H7. Why the important weights are the large ones

This is a reason to expect the pattern in H5-H6b, not a claim about one
neuron:

1. **Large gains turn ReLUs into switches.** A big weight makes a neuron
   jump from 0 to large over a tiny input change. That approximates if/then
   logic such as "is x_t bigger than the stored max?", and the task is
   logical.
2. **Cross-entropy rewards confident margins,** so weights behind rules that
   are always right, such as vetoes and resets, keep growing.
3. **Weight decay (0.001) shrinks weights that don't earn their size,** so a
   large weight that survives training is probably doing real work. This
   fits the ablations: removing neuron 7 or 16 costs 75-90% accuracy.

## C. Where the max is: a clock and lag-tagged neurons

### H4. A reset-started clock encodes how long ago the max arrived

**Claim.** The hidden state is close to the minimal state an exact online
algorithm needs: the max and 2nd max so far, and their lags (how many steps
ago each arrived). The lag of the max is encoded by a clock that each new
max restarts.

**Evidence.**
- **Minimal state.** The state is fitted by grouping inputs by (step, lag of
  max, lag of 2nd max), and fitting linearly in the two values and the
  current input within each group. This explains each neuron's value with a
  median R² of 0.94: 21 of 32 neurons reach 0.9, against 0.59 for positions
  alone. The blog's 16x10 model scores higher (0.97, 15 of 16 neurons at
  0.9). The handcrafted model scores lower (13 of 22) because its shift
  register stores every past input, not just the top two.
- **A clock.** Neuron 7's resets (H5) put their targets in a fixed state one
  step after a new max. From then on, many neurons switch on and off by the
  lag alone: for lags 2 and above, P(on | lag of max = k) varies by a median
  of only 0.02 across steps. P(on) by lag k = 0..9:
  - neuron 27 is on only at lags 3 and 7 (1.00), and at most 0.04 elsewhere,
    apart from lags 0, 6 and 9 (0.39, 0.37, 0.51);
  - neuron 11 is on at lags 3, 5 and 9 (1.00, 1.00, 0.97) and at lag 7
    (0.86), and off at lags 1, 2, 4 and 6;
  - neuron 10 is on at lags 4-5 (0.66, 0.78), and off at lags 1-3 and 6-8;
  - neuron 28 is off only at lag 1 (0.10), the state right after a reset;
  - neuron 3 ramps up with lag (0.07 at lag 2, then 0.49 and 0.72, then
    0.81-0.88).

  Neurons 24, 25 and 30 also look like this: they are off at a single lag
  (4, 1 and 5). They turn out to be top-two membership flags for that lag
  (H11), and so is the delay line below.
- **Lag-tagged neurons.** Some neurons carry the max's value only when it
  arrived a specific number of steps ago. The coefficients on the max at
  that lag, at the final step, are: neuron 22 at lag 2 (2.0), 19 at lag 3
  (1.6), 2 at lag 6 (2.6), 26 at lag 7 (4.0) and 15 at lag 8 (3.6). This is
  why the dictionary partly matched them to delayed copies of the input
  (`relu(x_{t-k})`, R² 0.63-0.80). H11 shows they mark top-two membership.
- **Reference neurons.** Neurons 0, 6, 13, 18 and 29 are ≈ `b·max + c·2nd −
  1.7·x_t`. `b + c` is roughly constant, and the split between max and 2nd
  switches with the lag. For example, neuron 13 is mostly the max at lags 2,
  5, 7 and 8, and mostly the 2nd at lags 3, 4, 6 and 9. The model is
  scale-free, so the difference between two such neurons is proportional to
  `max − 2nd`, with a sign set by the lag.
- **Patching the clock.** Swapping the max's and 2nd max's values changes how
  long ago the max arrived, but not which positions are in the top two. A
  group is patched from the swapped input at an intermediate step t, after
  both have arrived, and the recurrence then runs on. The table shows how
  often the final prediction moves to the swapped answer:

  | patched at step t | t = 4 | t = 6 | t = 8 |
  |---|---|---|---|
  | clock (3, 10, 11, 27, 28) | 34% | 38% | 41% |
  | reference (0, 6, 13, 18, 29) | 16% | 33% | 49% |
  | clock + reference | 53% | 68% | 73% |
  | clock + reference + new-max detector (7) | **85%** | **92%** | **91%** |
  | membership neurons (control) | 0% | 0% | 0% |

  Patched alone five steps before the end, the clock still moves the
  prediction 34% of the time, so it carries information forward. But it
  isn't a self-contained oscillator. Its own final on/off pattern follows the
  patched source only 9-15% of the time, and mostly reverts to the original
  (47-67%), so each step partly re-derives it from the rest of the state. The
  self-sustaining memory of which input is the max is the clock, the
  reference neurons and neuron 7 together.

**Status.** Causal at group level for the clock, which works together with
the reference neurons and neuron 7. The lag code itself is correlational
but very clean (exact 0/1 on-rates, the same at every step).

## D. The readout

All numbers are from `readout_analysis.py` on 2^19 fresh float64 inputs.
Groups are the roles in §F: "delay line" = 2, 15, 19, 22, 26;
"membership flags" = 20, 24, 25, 30; "reference" = 0, 6, 13, 18, 29;
"clock" = 3, 10, 11, 27, 28.

### H9. The final state is an additive position code, read by a matched filter

**Claim.** Averaged over inputs, the final hidden state is a sum of two
codes: a(i) for the max's position and b(j) for the 2nd max's position. The
state is normalised to unit length, which doesn't change the prediction. The
readout acts as a matched filter on b: W·b(j) is highest for logit j.

**Evidence.**
- The mean states for each (i, j) cell fit a(i) + b(j) with R² = 0.962.
- A surrogate final state a(i) + b(j) is 100% accurate, and agrees with the
  model on 99.44% of inputs, which is every input the model gets right.
- **B = W·b is diagonal** (`readout/codes.png`). For positions 0-8, logit j
  beats every other logit by 0.9-1.4 (on the unit-normalised scale) when the
  2nd max is at j. Logit 9 is suppressed by about 9.5 unless the 2nd max is
  at 9.
- **A = W·a is weakly diagonal for positions 0-8** (+0.3 to +0.8). Being the
  max slightly *favours* a position, and the 2nd-max code's larger margin
  overrides it. Position 9 is the exception: a vetoes it (−8.3) when the max
  is there, which is neuron 7's veto (H6).

**Status.** Correlational (fits), plus the surrogate test.

### H10. Errors are value-dependent noise overwhelming the position margin

**Claim.** Per input, the state also has a large part that depends on the
values rather than the positions: a(i) + b(j) explains only 55% of each
normalised final state. The readout mostly ignores this part, and the model
errs when it doesn't.

**Evidence.**
- The final state is exactly a(i) + b(j) plus this part, so the model errs
  exactly when it outweighs the position code's margin. That holds by
  construction and isn't a test. What can be measured is its size:
  typically it moves the decisive margin by 26% of the margin (99th
  percentile: 207%).
- The readout sees only 9 of the 32 dimensions, the ones that change logit
  differences. A random direction would have 28% of its variance there; the
  value-dependent part has only 12.6%.

**Status.** Correlational. It links to H8: near-ties and negative values
make the margin small relative to this noise.

### H11. A top-two shift register carries the 2nd max's position

**Claim.** For each lag k, a neuron flags whether the input k steps ago is
among the running top two. The flag doesn't depend on whether that input is
the max or the 2nd, so it survives when a new max demotes the old one. The
flags are passed from lag to lag by a shift register that the event
detectors write into.

**Evidence.**
- **The delay line.** Neurons 22, 19, 2, 26 and 15 cover lags 2, 3, 6, 7
  and 8. Pooled over all steps, each is on 99.1-100% of the time when the
  input at its lag is the running max *or* the running 2nd. Otherwise it is
  on 24-82% of the time, at lower amplitude.
- **Membership flags** cover other lags by switching *off* when the input at
  that lag is in the top two. As classifiers, neuron 25 (lag 1) is 87%
  accurate, 24 (lag 4) 91%, 30 (lag 5) 93% and 20 (lag 9) 94%, against
  majority baselines of 57-80%. At the final step, where lag k is position
  9 − k, they are clean position flags (`state/summary.json`):

  | neuron | position | P(on) if that position is in the top two | P(on) otherwise |
  |---|---|---|---|
  | 25 | 8 | 0.009 | 0.973 |
  | 24 | 5 | 0.002 | 0.940 |
  | 20 | 0 | 0.265 | 0.996 |

  Neuron 14 is a second lag-1 flag at the final step: it is on when
  position 8 is in the top two (P(on) 1.00, against 0.17 otherwise). Lags 7 and 8 have no on/off
  flag better than the baseline. There, the delay neurons' amplitude carries
  membership.
- **Final-step ablations.** Mean-ablating at the last step only cuts
  accuracy as follows. "Before" and "after" refer to whether the 2nd max
  arrived before or after the max:

  | ablated | 2nd after max | 2nd before max |
  |---|---|---|
  | membership flags | 73% | 72% |
  | delay line | 90% | 72% |
  | reference neurons | 44% | 59% |

- **Patching the 2nd max's position.** The pairs share the max's position
  but have a different 2nd-max position. Copying a group's final state from
  the other input moves the prediction to the other input's answer:
  - delay line: 14% (23% in the "before" case);
  - membership flags: 9%;
  - delay line + flags: 46% (63% before);
  - adding the reference neurons: 67%;
  - adding neuron 12: **80%**, with 15 neurons in total.

  No other group moves it more than 2%.
- **Where the 15 fall short** (`patch_2nd_position.base_15_by_position`).
  They transfer the position least well when the 2nd max moves to position
  8 (54%) or 7 (59%), lags 1-2, and when the max is at position 9 (58%).
  Greedily adding neurons:

  | added | 3 | 4 | 14 | 8 | 5 | 21 | 16 | 23 |
  |---|---|---|---|---|---|---|---|---|
  | transfer | 85% | 89% | 92% | 93% | 94% | 94% | 96% | **99.6%** |

  That makes 23 neurons in total. Neuron 14 is the second lag-1 flag, and 21
  and 23 cover the cases where the last input is involved.
- **The wiring is a shift register.** The per-lag neurons form a chain from
  lag 1 to lag 9:

  25 → 22 → 19 → 24 → 30 → 2 → 26 → 15 → 20

  For most, the strongest incoming recurrent weight comes from the previous
  lag's neuron. The sign follows the polarity:
  - **positive** between neurons of the same polarity: 22 → 19 (+1.19),
    24 → 30 (+0.86), 2 → 26 (+0.73), 26 → 15 (+0.86);
  - **negative** where an "on when in the top two" neuron meets an "off when
    in" one: 25 → 22 (−1.16), 19 → 24 (−1.01), 30 → 2 (−2.09), 15 → 20
    (−0.63).

  The lag-1 neuron 25 is written by the event detectors (−9.9 from 16, −7.9
  from 7). Cutting each link:

  | link cut | downstream flag accuracy | model accuracy |
  |---|---|---|
  | detectors 16, 7 → 25 | 0.87 → 0.72 | 99.4% → 79.9% |
  | 25 → 22 | 0.83 → 0.50 | 99.2% |
  | 22 → 19 | 0.71 → 0.47 | 86.6% |
  | 19 → 24 | 0.91 → 0.72 | 91.8% |
  | 24 → 30 | 0.93 → 0.94 (no effect) | 99.1% |
  | 30 → 2 | 0.81 → 0.27 | 83.1% |
  | 2 → 26 | (26's on/off isn't a flag) | 93.3% |
  | 26 → 15 | (15's on/off isn't a flag) | 89.5% |
  | 15 → 20 | 0.94 → 0.80 | 92.2% |

  This is the handcrafted model's shift-register idea, applied to event
  flags rather than raw inputs. Two links are redundant: 25 → 22 for
  accuracy, and 24 → 30 entirely.

**Status.** Causal. 23 neurons carry 99.6% of the 2nd max's position at the
readout, and 15 of them (the delay line, flags, reference neurons and 12)
carry 80%. The shift-register links are causal for most lags.

### H12. A separate code says which top-two input is the max

**Claim.** The membership code (H11) is the same whether an input is the
max or the 2nd. A separate code says which of the two is the max, carried by
the reference neurons and the clock.

**Evidence.**
- **Swapping the max and 2nd** (`swap_max_and_2nd`) keeps the set of top-two
  positions and flips only which one is the max. The correct answer moves
  from the 2nd's position to the old max's. Patching a group's final state
  from the swapped input moves the prediction there:

  | group patched | prediction moves |
  |---|---|
  | reference (0, 6, 13, 18, 29) | 62% |
  | clock (3, 10, 11, 27, 28) | 35% |
  | new-max detector (7) | 9% |
  | last-max (21, 23) | 3% |
  | **delay line, membership flags** | **0%** |
  | everything else | ≤ 0.5% |

  The two codes are cleanly separate: the membership neurons carry no
  max/2nd information. Greedy patching reaches 94% with 10 neurons: 13, 6,
  3, 7, 0, 29, 11, 28, 21, 23.
- **Sufficiency** (`role_sets.sufficiency`, final step, unit-normalised).
  Three groups are kept:
  - the membership neurons: the delay line, the flags and 14;
  - the identity neurons: the reference neurons, the clock and 7;
  - the last-position neurons: 12, 16, 21 and 23.

  With the other 7 neurons (1, 4, 5, 8, 9, 17, 31) mean-ablated, the result
  agrees with the model on 95.0% of inputs. Membership and identity alone
  agree on 75%; either alone, 22-28%.
- **The other 7 don't carry position.** Resampling them from other inputs
  with the same max and 2nd positions changes 3.8% of predictions. That is
  about as many as resampling the *kept* 25 the same way (4.2%). The changed
  inputs have smaller margins (median 0.38, against 0.56 overall), so the
  remaining gap is value-dependent sensitivity on low-margin inputs.
- **Refuted along the way:** the value neurons don't cancel the reference
  neurons' value-dependent part. Their contributions to the margin within a
  cell are uncorrelated (−0.02).

**Status.** Causal at group level.

## E. Errors

### H8. Errors come from three blind spots

**Claim.** Most errors come from three mechanisms. First, comparisons are
against running values clipped at 0 (the max(0, …) convention in H1-H3), so
negative values can't be ranked. Second, comparisons have finite relative
resolution, so near-ties between the 2nd and 3rd largest are missed. Third,
the same finite resolution applies to which top-two input is the max (H12),
so near-ties between the max and the 2nd are missed too.

**Evidence** (`explain_32x10.py`, 2^20 inputs):

| inputs | share of inputs | error rate | share of errors |
|---|---|---|---|
| 2nd largest is negative | 1.08% | 21.6% | 42% |
| 2nd and 3rd within 3% of each other (2nd positive) | 5.76% | 3.7% | 38% |
| everything else | 93.2% | 0.12% | 20% |

66% of all errors predict the 3rd largest.

**The third blind spot** (`estimation_harness.py`, `cells.md`). Splitting
inputs into exact cells by both relative gaps, g12 = (max − 2nd)/max and
g23 = (2nd − 3rd)/2nd, accounts for most of the errors left in "everything
else" above:

| errors | share |
|---|---|
| negative values | 41% |
| 2nd-3rd near-ties (g23 < 0.03) | 39% |
| max-2nd near-ties only (g12 < 0.01, g23 ≥ 0.03; 1.6% of inputs) | 14% |
| none of these | 6% |

In the max-2nd near-ties, errors mostly predict the max's position.

**Predicting errors on shifted inputs** (`shift_test.py`). Adding a constant
to every input leaves the true answer unchanged, but changes how often each
blind spot occurs. The prediction uses only measurements at shift 0: the
error rate of each of four input classes there is re-weighted by how often
that class occurs at each shift. The class frequencies come from the inputs
alone. The four error rates at shift 0 are:
- max and 2nd both negative: 67%;
- only the 2nd negative: 17.4%;
- relative 2nd-3rd gap below 3%: 3.6%;
- everything else: 0.13%.

| shift | −1.5 | −1.0 | −0.5 | 0 | +0.5 | +1.0 | +2.0 |
|---|---|---|---|---|---|---|---|
| predicted | 0.397 | 0.178 | 0.038 | 0.0056 | 0.0046 | 0.0058 | 0.0082 |
| measured | 0.439 | 0.181 | 0.039 | 0.0056 | 0.0038 | 0.0049 | 0.0085 |

The prediction includes the non-obvious rise for positive shifts, where
larger values make relative near-ties more common. With three classes (both
negative not separated from only the 2nd negative), large negative shifts
were under-predicted: 0.19 against 0.44 at −1.5. The "both negative" case is
rare at shift 0 but dominant at −1.5.

**One neuron doesn't explain the near-tie errors.** The top-2 detector's
threshold is biased: it fires up to about 1% below the running 2nd max. But
predicting errors from that bias alone gives 8,858 errors against 833
measured, so several neurons resolve near-ties together.

**Status.** Correlational; the shift prediction is confirmed. The max-2nd
near-tie blind spot was found during step 10 and isn't in the shift test's
classes.

### H13. The errors are crosstalk from irrelevant inputs

**Claim.** In all three blind spots, the model's decision depends on
inputs that don't affect the true answer. They reach the decision through
activation-pattern switches spread over many neurons. So the errors are
resolution limits set by crosstalk, not rules on the top values.

**Evidence** (`crosstalk.py`, `tie_offsets.py`):
- **Redraw test.** Keeping the top three inputs and redrawing the rest below
  them changes 18.2% of predictions when only the max is positive (error
  17.0%), 31.4% when all inputs are negative (error 66.4%), and 20.8% at
  2nd-3rd near-ties (error 21.8%). In a typical cell it changes 0.02%
  (error 0.02%).
- **Not linear.** The gradient with respect to the redrawn inputs doesn't
  predict the change in the 2nd-vs-3rd logit difference (R² far below 0),
  and it over-predicts flips (26% against 15% actual, only the max
  positive).
- **Spread out.** Copying the 12 best neurons' final states from the
  redrawn input reproduces 49% (only the max positive) and 54% (near-tie)
  of the flips. The largest single carrier is neuron 12 (26-29%), which
  reads the last input even when it is irrelevant.
- **Near ties have no fixed offset.** The model's switch point near a
  2nd-3rd tie is offset from the true tie by a roughly centred, noise-like
  amount, about ±0.1% relative. The arrival positions of the max and the
  pair explain 2% of the per-line error. A fit linear in all ten inputs
  within each position cell explains 45% of the offset (held out).
- **The last step is a vote.** Given the first nine inputs, the x_9 at
  which the model switches between position 9 and the earlier 2nd lies
  within about ±0.4% of the true 2nd. It is uncorrelated with each single
  neuron's threshold (|r| ≤ 0.06): the running 2nd is read off by
  cancellation across many neurons.
- **Negative values are ranked coarsely, not by a rule.** With only the
  max positive, 83% of wrong answers point at the 3rd largest. The error
  falls smoothly from about 50% to 2-6% as the 2nd-3rd gap grows from
  under 1% to over 30% of the max.

**Consequence.** No deductive accuracy estimate is in reach without a
statistical model of this crosstalk (results/2nd_argmax_32x10/README.md,
"Estimating the error").

**Status.** Causal for the redraw test. The neuron-level account is partial.

## F. Neuron roles in 32x10

*Drop* is the accuracy lost under mean ablation at all steps
(`dictionary/dictionary.md`). *State R²* is the share of the neuron's value
explained by the ideal state plus the current input (`state/summary.json`).

| neuron | role | evidence | drop | state R² |
|---|---|---|---|---|
| 7 | new positive max: `max0 − max0_prev` (H1); resets (H5); vetoes position 9 (H6) | causal | 0.88 | 1.00 |
| 16 | new positive top-2 value (H2); resets 3 and 25 (H5) | causal | 0.75 | 0.93 |
| 17 | running max (H3) | causal | 0.58 | 1.00 |
| 31 | running 2nd max (H3) | correlational (R² 0.96) | 0.33 | 0.96 |
| 1, 5 | gap between max and 2nd max (H3) | correlational (per-lag fits) | 0.25, 0.31 | 0.96, 0.96 |
| 12 | how far x_t is below the running 2nd max; vetoes position 9 (H6b) | causal (veto) | 0.44 | 1.00 |
| 0, 13, 29 | reference − current input, with the max/2nd mix set by the lag (H4); reset by 7 | causal as a group (H4, H12) | 0.37, 0.45, 0.46 | 0.99 |
| 6, 18 | the same, not reset by 7 | causal as a group (H4, H12) | 0.27, 0.20 | 0.99 |
| 3, 10, 11, 27, 28 | clock: on or off at fixed lags of the max (H4) | causal as a group (H4, H12) | 0.16, 0.003, 0.14, 0.07, 0.11 | 0.90-0.98 |
| 22, 19, 2, 26, 15 | delay line: on when the input at lag 2, 3, 6, 7, 8 is in the top two (H11) | causal (patching, link cuts) | 0.33, 0.13, 0.19, 0.13, 0.06 | 0.86-0.96 |
| 25, 24, 30, 20 | membership flags: off when the input at lag 1, 4, 5, 9 is in the top two (H11) | causal (patching, link cuts) | 0.19, 0.08, 0.33, 0.08 | 0.79-0.89 |
| 14 | second lag-1 flag at the final step (H11) | causal (greedy patching) | 0.02 | 0.66 |
| 21, 23 | mostly on when x_t is a new max; carry last-position cases (H11, H12) | correlational role; in both greedy patching sets | 0.18, 0.23 | 0.98, 0.89 |
| 8 | always on; carries the max with a weight that grows with lag | unclear | 0.08 | 0.88 |
| 4, 9 | unexplained | — | 0.005, 0.003 | 0.83, 0.81 |

## G. Generality: other seeds and model sizes

Neuron numbers differ between models, so roles were matched by behaviour,
not index. The causal tests on seeds 1-4 use each seed's own neuron numbers
(see `scripts/run_all.sh`), and `readout_generality.py` finds its flags and
greedy sets per model.

**In all five 32x10 seeds:**

- **An exact new-max detector** (H1): one neuron whose value is
  `max0 − max0_prev` (R² 1.00) and whose on/off agrees with "new positive
  max" on at least 99.6% of inputs. The zoo's 16x10 model has one too (99.8%).
- **The detector vetoes the last position** (H6). In every seed and in the
  zoo's 16x10, the largest readout weight is detector → last logit (−22 to
  −71). Removing it drops accuracy on inputs whose max is last from 0.99 to
  0.67-0.72 (seed 2: 0.03).
- **A "below the 2nd max" neuron vetoes the last position** (H6b). Every
  seed has a neuron matching `relu(2nd0_prev − x_t)` with R² of at least
  0.95. Removing its weight to the last logit causes wrong "last position"
  answers (72-100% of new errors), and costs from 1% (seed 3) to 76%
  (seed 1) accuracy.
- **Resets from the detector** (H5). The detector has the largest share of
  outgoing recurrent weight in every seed (14-47%), and its strong targets
  are zero after it fires (P ≥ 0.98). How much accuracy depends on them
  varies: cutting them costs 0% (seed 2) to 35% (seed 4).
- **A lag clock** (H4): 4-8 neurons that are always on at some lag of the
  max (2 or more) and always off at another.
- **Minimal-state encoding** (H4): median state R² 0.92-0.95.
- **The readout story** (H9-H12):
  - the additive code has R² 0.95-0.96, and its surrogate agrees with the
    model on all the model's correct inputs;
  - the matched filter has a positive margin at every position (minimum
    0.46-0.87);
  - the readout sees 7-13% of the residual, against 28% for a random
    direction;
  - 5-7 of the 9 lags have clear membership flags;
  - **all 20 shift-register links found follow the polarity sign rule**;
  - the membership flags carry at most 3% of the max/2nd identity, and 7-10
    neurons carry 90% of it.

**Varies by seed:**

- **How exclusively the detector carries a new max.** Patching everything
  *except* the detector transfers the effect of a new max 3% of the time in
  seed 0, where the detector is necessary, but 16-67% in seeds 1-4, where
  other neurons also carry it.
- **The top-2 detector** (H2): event agreement of at least 99% in seeds 0-3
  and 90% in seed 4. Patched alone, it transfers 86% (seed 0) down to 0%
  (seed 4).
- **A single running-2nd-max neuron** exists only in seed 0 (R² 0.96; other
  seeds 0.46-0.71). Every seed has a single running-max neuron (R²
  0.92-1.00), but how much of the max's value it carries alone varies
  (38-78%).

**Other sizes:**

- **16x10 has two solution types.** The zoo's best 16x10 seed looks like
  32x10: a detector, a veto of −37.6, a 64% reset share and a clock neuron.
  The blog's 16x10 model looks like the handcrafted model: no detector veto,
  no clock, a low reset share (0.17), and a near-exact `max0 − x_t` neuron
  (R² 0.99, like handcrafted neuron 21).
- **Both 16x10 models fit the readout story less well.** In the zoo's
  16x10, the surrogate agrees with the model on only 83% of inputs, against
  an accuracy of 95%. Only 3 lags have clear flags, and identity is spread
  over 15 of its 16 neurons. The blog's 16x10 has the additive code
  (surrogate agreement 95%), but also spreads identity over 15 neurons.
- **The failing 8x10 model (43%) lacks the mechanism**: no event detector
  (best agreement 0.39), no veto, and a state R² of only 0.76.
- **8x6 (79%) and 8x5 (88%)** have the detector (at least 97%) but no clock
  (0-1 clock neurons), since there aren't enough neurons for both.
- **8x4 and 4x3 (98.5% or more)** reach high accuracy with only a weak
  detector (85% or less): short sequences don't need this machinery.

## Open questions

These are also the open items in the [PLAN.md roadmap](PLAN.md#open-questions).

- **The identity code, neuron by neuron.** H12 shows which neurons say which
  top-two input is the max. It doesn't show how each reference neuron's
  lag-dependent mix of max and 2nd (`lag_codes/coefs_lag_max.png`) combines
  with the clock to do that at each lag.
- **The clock's internal dynamics.** Patching shows the clock re-derives
  itself each step from the reference neurons and neuron 7 (H4). The wiring
  between the three isn't written out.
- **How membership is revoked.** H11 explains how a flag enters the shift
  register and moves along it. It doesn't explain how a flag is cleared when
  its input drops out of the top two.
- **The lag of the 2nd max** is encoded less cleanly than the lag of the
  max. P(on | lag of 2nd) has a median spread across steps of 0.06 for lags
  of 2 or more (90th percentile 0.28), against 0.02 for the max.
- **Neurons 8, 4 and 9.** Neuron 8 costs 8% when ablated and carries the max
  with a weight that grows with lag. Its role is unclear. Neurons 4 and 9
  cost less than 1%.
- **A statistical model of the crosstalk** (H13): how irrelevant inputs
  flip which neurons, well enough to predict the error rate of each blind
  spot. That would give the deductive accuracy estimate that ARC's
  challenge asks for. Neuron 12's reading of an irrelevant last input is
  the most structured piece.
