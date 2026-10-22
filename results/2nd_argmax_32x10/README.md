# 2nd argmax 32x10: how it works

The zoo's 2nd argmax RNN with 32 hidden neurons and 10 inputs (1,376
parameters, 99.44% accurate). It reads `x_0..x_9` one at a time and outputs a
logit for each position, and has to point at the 2nd largest input.

This write-up collects roadmap steps 1-6. Evidence for each claim, under
the H numbers given in section headings, is in
[docs/HYPOTHESES.md](../../docs/HYPOTHESES.md), and every number comes from a
script (listed under [Reproducing](#reproducing)). Neuron numbers refer to
this model (seed 0); other seeds use different numbers for the same roles.

## Summary

The network runs an online algorithm close to the minimal one: after each
input it effectively knows the largest and 2nd-largest values so far and how
many steps ago each arrived.

1. **Detect events.** One neuron fires exactly when the input is a new
   positive running max, and outputs how much the max just increased (neuron
   7, R² 1.00). Another fires when the input is a new positive top-2 value
   (neuron 16, 99.6-100% agreement).
2. **Store values.** Neurons hold the running max (17), the running 2nd max
   (31), and the gap between them (1, 5).
3. **Reset.** When a new max arrives, the detector's large negative recurrent
   weights (up to −31) zero about ten neurons on the next step.
4. **Keep time.** From that reset, a set of neurons runs a fixed on/off
   sequence that depends only on how long ago the max arrived (a clock). Other
   neurons each watch one lag and mark whether the input that many steps back
   is among the top two (a delay line). Together these encode *where* the max
   and 2nd max are.
5. **Read out.** The final state is (on average) a sum of a code for the
   max's position and a code for the 2nd max's position, and the readout is
   a matched filter on the second. Two separate codes feed it: a delay line
   marks which recent inputs are in the top two (surviving the reset), and
   the reference neurons and clock say which of those two is the max.
   The last position gets two hard vetoes: "x_9 is the new max" (neuron 7,
   weight −65) and "x_9 is below the running 2nd max" (neuron 12, weight −40).
6. **Compare against max(0, …).** All comparisons are against running values
   clipped at 0, which a ReLU without biases gets for free. This is also the
   model's main blind spot.

The errors (0.56%) come from three blind spots: negative values, 2nd-3rd
near-ties and max-2nd near-ties. In all three, the decision is perturbed by
inputs that shouldn't matter, through many neurons at once. That crosstalk
is why the model's error can be estimated about 5× more efficiently than by
sampling, but not yet deduced from its weights (see
[Estimating the error](#estimating-the-error-arcs-challenge)).

## Mechanism

### 1. Events (H1, H2)

| neuron | computes | match | causal test |
|---|---|---|---|
| 7 | `max(0, x_0..x_t) − max(0, x_0..x_{t-1})`: on exactly at a new positive max | R² 1.000; on/off agreement 99.9-100% at every step | Patching every neuron *except* 7 transfers a new max's effect only 1-5% of the time: 7 is necessary |
| 16 | on when x_t > 0 and is among the two largest so far | agreement 99.6-100% | Patching 16 alone transfers a new top-2 value's effect 74-93% of the time |

### 2. Stored values (H3)

| neuron | computes | match |
|---|---|---|
| 17 | `max(0, x_0..x_{t-1})` (running max, same as handcrafted neuron 20) | R² 0.986 |
| 31 | `max(0, 2nd largest of x_0..x_{t-1})` | R² 0.963 |
| 1, 5 | a fixed multiple of max − 2nd max, at every lag | per-lag fits (`lag_codes/coefs_lag_max.png`) |
| 12 | `relu(max(0, running 2nd) − x_t)`: how far the input falls below the 2nd max | R² 0.998 |

Changing only the running max's *value* and patching neuron 17 alone carries
65% of the change into the next new-max decision; 17 with 1, 5 and 31
carries 80%.

### 3. Resets (H5)

46.5% of all recurrent weight magnitude leaves neuron 7. After 7 fires, each
of its ten strong targets is zero on the next step 98-100% of the time.
Cutting those edges shows 7 directly causes the resets of neurons 0, 13, 28
and 29. Without its reset edges, accuracy falls from 99.5% to 67.4%.

### 4. Position: a clock and lag-tagged neurons (H4)

The minimal online algorithm needs the max, the 2nd max and their lags.
Grouping inputs by (step, lag of max, lag of 2nd max) and fitting linearly in
the two values and the current input explains a median **R² of 0.94** of each
neuron's value (21 of 32 neurons ≥ 0.9). Positions alone give 0.59.

- **Clock.** From two steps after a new max, many neurons' on/off state depends
  only on the lag. It varies by a median of just 0.02 across steps. For
  example, neuron 27 is on exactly at lags 3 and 7, neuron 11 at lags 3, 5 and
  9, and neuron 28 is off only right after a reset. It is causal: swapping which top-two input is the max and
  patching only the clock at step 4 moves the final prediction 34% of the time
  (`clock/summary.json`). Patching the clock together with the reference
  neurons and neuron 7 moves it 85-92%.
- **Delay line for the top two.** Neurons 22, 19, 2, 26 and 15 carry the max
  value when it arrived 2, 3, 6, 7 or 8 steps ago. They also stay on when
  the input at that lag is the 2nd max (see section 5).
- **Reference minus current input.** Neurons 0, 6, 13, 18 and 29 compute
  roughly `b·max + c·2nd − 1.7·x_t`, where the split between max and 2nd
  switches with the lag.

![position codes](state/position_codes.png)

### 5. Readout: a matched filter on an additive position code (H9-H11)

From `python scripts/readout_analysis.py --hidden 32 --seq-len 10`
(`readout/summary.json`).

- **The final state is an additive position code.** Averaged over inputs
  and normalised to unit length (which doesn't change the prediction), the
  final state is a(max position) + b(2nd position), with R² = 0.962. A
  surrogate built from a(i) + b(j) alone is 100% accurate.
- **The readout is a matched filter on b.** W·b(j) is highest for logit j,
  beating every other logit by 0.9-1.4 for positions 0-8 (unit-normalised
  scale). W·a(i) slightly *favours* the max's own position (+0.3 to +0.8),
  which the larger 2nd-max margin overrides.
- **A delay line for the top two.** Neurons 22, 19, 2, 26 and 15 are on
  99-100% of the time when the input 2, 3, 6, 7 or 8 steps back is the
  running max *or* the running 2nd, so they keep marking the old max after a
  new max demotes it. Neurons 25, 24, 30 and 20 flag the same for lags 1, 4,
  5 and 9 by switching off (87-94% accurate).
- **It's a shift register.** The per-lag neurons form a chain (25 → 22 → 19
  → 24 → 30 → 2 → 26 → 15 → 20) whose link signs follow their polarity, and
  the event detectors write into its first neuron. Cutting single links costs
  up to 16% accuracy, and cutting the detectors' input costs 20%
  (`wiring/summary.json`).
- **Causal check.** For inputs that differ only in where the 2nd max is,
  copying the final state of the delay line, membership flags, reference
  neurons and neuron 12 (15 neurons) from the other input moves the
  prediction to the other input's answer 80% of the time. No other group
  moves it more than 2%. Adding 8 more neurons (3, 4, 14, 8, 5, 21, 16, 23)
  brings it to 99.6%.
- **A separate code says which top-two input is the max.** Swapping the max's
  and 2nd max's values leaves the membership code unchanged, and patching the
  delay line or flags from the swapped input moves the prediction 0% of the
  time. The identity comes from the reference neurons (62%), the clock (35%)
  and the new-max detector (9%). Ten neurons carry 94%.
- **Sufficiency.** Keeping only the membership, identity and last-position
  neurons (25) at the final step and mean-ablating the other 7 agrees with the
  model on 95% of inputs. The other 7 don't carry position: resampling them
  within the same position cell changes about as many predictions as
  resampling the kept 25.
- **Errors.** Per input, about 45% of the normalised final state depends on
  the values rather than the positions. The readout mostly ignores it: only
  12.6% of it lies in the 9 directions the readout can see, against 28% for a
  random direction. Typically it shifts the decisive margin by about a
  quarter of the margin, and the model's errors are the inputs where it
  outweighs the margin.

![position codes and readout](readout/codes.png)

The last position is handled by two hard vetoes:

- Removing **7 → logit 9** (−65) drops accuracy on inputs whose max is last
  from 99.2% to 72.4%, and changes nothing elsewhere.
- Removing **12 → logit 9** (−40) causes wrong "position 9" answers when x_9
  is just below the 2nd max (mostly when x_9 is the 3rd largest).

## Quantitative checks

All from `python scripts/explain_32x10.py --hidden 32 --seq-len 10`
(`explain/summary.json`, 2^20 fresh float64 inputs).

### Rebuild from the explanation

Replace the model's final hidden state with a prediction from the ideal state
alone: one affine fit per (lag of max, lag of 2nd max) cell, fitted on a
separate sample. Then apply the model's own readout.

| surrogate final state built from | held-out R² of final state | agrees with model | accuracy |
|---|---|---|---|
| positions of max and 2nd max only | 0.38 | **99.44%** | 100.00% |
| + max and 2nd max values | 0.84 | 99.43% | 99.90% |
| + current input | 0.94 | 99.41% | 99.84% |

With positions alone, the surrogate agrees with the model on exactly the
inputs the model gets right: it reproduces all of the model's correct
behaviour and none of its errors. **The explanation accounts for what the
model does right; the remaining 0.56% is its errors**, analysed below.

### Resample check

Resample what the explanation calls irrelevant, keeping the true answer
fixed:

| resampled | predictions unchanged | unchanged where the model was right |
|---|---|---|
| every input except the max, the 2nd max and the last input (redrawn below the 2nd max) | 99.17% | 99.61% |
| all values, keeping only the positions of the max and 2nd max | 98.92% | 99.48% |

If the model's errors are independent near-ties, the best possible is about
(accuracy)² = 98.88%. Both resamples are at that ceiling.

### Where the errors are

The model makes 5,885 errors in 2^20 inputs. Two mechanisms predict where
most of them fall:

| inputs | share of inputs | error rate | share of errors |
|---|---|---|---|
| **2nd largest is negative.** Comparisons use max(0, …), so values below 0 can't be ranked. Probability 11/1024 for 10 inputs. | 1.08% | 21.6% | **42%** |
| **2nd and 3rd within 3% of each other** (2nd positive). Comparisons have finite resolution. | 5.76% | 3.7% | **38%** |
| everything else | 93.2% | 0.12% | 20% |

Among errors, the median gap between the 2nd and 3rd largest is 0.018,
against 0.258 overall, while the gap between max and 2nd max is typical
(0.32 vs 0.41). 66% of errors predict the 3rd largest.

Most of the remaining 20% are a third blind spot: near-ties between the max
and the 2nd, where the model points at the max. Splitting inputs into exact
cells by both gaps (`estimation/cells.md`) puts 41% of errors on negative
values, 39% on 2nd-3rd near-ties, 14% on max-2nd near-ties, and 6% elsewhere. The error rate rises
from under 1% to about 40% as the relative gap shrinks from 1% to 0.03%:

![errors](explain/errors.png)

### Predicting errors on shifted inputs

Adding a constant to every input changes how often each blind spot occurs
but not the true answer. Re-weighting the shift-0 error rates of four input
classes by their frequency at each shift predicts the model's error closely,
including the rise for positive shifts, where near-ties become more common
(`python scripts/shift_test.py --hidden 32 --seq-len 10`):

| shift | −1.5 | −1.0 | −0.5 | 0 | +0.5 | +1.0 | +2.0 |
|---|---|---|---|---|---|---|---|
| predicted error | 0.397 | 0.178 | 0.038 | 0.0056 | 0.0046 | 0.0058 | 0.0082 |
| measured error | 0.439 | 0.181 | 0.039 | 0.0056 | 0.0038 | 0.0049 | 0.0085 |

![shift](shift/shift.png)

## Estimating the error: ARC's challenge

ARC's main challenge for AlgZoo is a *mechanistic estimate* of a model's
accuracy. The estimate should reason from the model's structure rather than
sample it, and it should match random sampling in **mean squared error for
the same compute**, scored over noisy copies of the model. ARC poses it for
its 16x10 model; we tried it on this one, where our explanation is
strongest. The plan and every step's details are in
[docs/PLAN.md](../../docs/PLAN.md) (steps 9-13).

**In short: we beat sampling about 5× by using exact probabilities, but no
deductive estimate is in reach.** Every one of the model's errors turns out
to be crosstalk: inputs that don't affect the true answer perturb a
comparison through many neurons switching on and off.

### The scoring harness

`python scripts/estimation_harness.py --hidden 32 --seq-len 10`
(`estimation/summary.json`). The family is the model plus 16 copies with
relative noise on every weight (`w + σ·|w|·ε`, σ = 0.001 and 0.003). The
model is sensitive to this: the error rates range from 0.56% to 1.53%.
Each copy's ground truth uses 2^25 inputs, and compute is counted in
multiply-adds (10,880 per forward pass). Plain sampling's measured MSE
matches p(1 − p)/N at every budget, which is the line any estimator has to
beat.

![MSE versus compute](estimation/mse_vs_compute.png)

### What works: exact probabilities (E1)

The inputs are split into 92 cells: max < 0, only the 2nd < 0, and a grid
over the relative gaps max-2nd and 2nd-3rd. Each cell's probability is
computed exactly from a 1-D integral over the order statistics of 10
Gaussians. Only the error within each cell is sampled, with the budget
spread by a fixed Neyman allocation. **E1 beats sampling by 4.4-7.5× at
every budget,** on the noisy copies as well as the model itself.

Splitting on the max-2nd gap found a third blind spot. Near-ties between
the max and the 2nd cause errors too, usually predicting the max's position
(`estimation/cells.md`):

| errors | share |
|---|---|
| negative values | 41% |
| 2nd-3rd near-ties | 39% |
| max-2nd near-ties | 14% |
| none of these | 6% |

### What doesn't help: integrating along tie lines (E2)

Along a line through an input in the tie direction, the model is piecewise
linear, so its error on that line can be integrated exactly
(`algzoo_interp/lines.py`). E2 does this inside E1's cells where the pilot
finds it cheaper. **It does no better than E1 (3.6-7.4×).** Whether a line
has errors depends on the rest of the input, not on the position along the
line. Over the near-tie window, the per-line error has a standard deviation
6.2× its mean, and each line costs about 21 forward passes.

### Why a deductive estimate isn't in reach

`python scripts/tie_offsets.py` (`ties/summary.json`) and
`python scripts/crosstalk.py` (`crosstalk/summary.json`):

- **Near-tie errors aren't a biased threshold.** The point where the model
  switches between the 2nd and 3rd can be found exactly. A first-order
  formula at the tie matches the exact walk on 75% of lines and captures
  89% of the mean error. But the offset of that switch point from the true
  tie is noise-like and roughly centred, within about ±0.1% of the values.
  The arrival positions of the max and the pair explain 2% of it. A fit
  linear in all ten inputs, within each position cell, explains 45%
  (held out).
- **One exception: when the later element of the pair is the last input,**
  the tie error rate is 4× higher (0.0055 against 0.0014). That case is
  20% of inputs and 45% of the tie error.
- **The last step is a vote.** For a given first nine inputs, the value of
  x_9 at which the model switches between position 9 and the earlier 2nd
  lies within about ±0.4% of the true 2nd. It is uncorrelated with every
  single neuron's threshold (|r| ≤ 0.06). Near the crossing it equals a
  weighted average of all active neurons' thresholds, so the running 2nd
  is read off by cancellation across many neurons.
- **The negative-value blind spot isn't a rule.** With only the max
  positive, 83% of wrong answers point at the 3rd largest, and the error
  falls smoothly as the gap between the two negatives grows (about 50% at
  a gap under 1% of the max, 2-6% above 30%). The model ranks negative
  values, just coarsely, without its explicit running-2nd memory, which
  holds max(0, 2nd) = 0.
- **Irrelevant inputs change the hard decisions.** The test keeps the top
  three inputs, redraws every other input below them, and counts changed
  predictions:

  | cell | error | predictions changed |
  |---|---|---|
  | only the max positive | 17.0% | 18.2% |
  | all negative | 66.4% | 31.4% |
  | 2nd-3rd near-tie | 21.8% | 20.8% |
  | typical | 0.02% | 0.02% |

- **The crosstalk is nonlinear and spread out.** The gradient with respect
  to the redrawn inputs doesn't predict it (R² far below 0). The 12 most
  responsible neurons reproduce about half the flips (49% and 54%). The
  biggest single carrier is neuron 12 (26-29%), which reads the last input
  even when it is irrelevant and whose readout weights reach every logit.

So a deductive estimate would need a statistical model of how irrelevant
inputs flip neurons across the whole state. That is a deeper explanation
than the one in this write-up. The circuit above explains what the model
does right; its errors are the residue of imperfect cancellation across
many neurons.

**An earlier, related finding.** A one-neuron error model fails in the
same way. The new-top-2 detector has a biased threshold: it still fires
when x_t is up to about 1% *below* the running 2nd max (47% firing at
0.75-1% below). Predicting errors from that bias alone (for inputs where
the 3rd largest arrives after the max and 2nd) gives 8,858 errors where
the model makes 833. Several neurons decide near-ties together.

## Generality

Across all five 32x10 seeds (`../compare/compare.md`, `../compare/causal.md`):

- **In every seed:** the exact new-max detector, its veto on the last
  position, the "below the 2nd max" veto, resets, a clock of 4-8 neurons, and
  minimal-state encoding (median R² 0.92-0.95).
- **Varies by seed:** whether the detector is the *only* carrier of a new max,
  the importance of the top-2 detector and the resets, and whether the
  running 2nd max sits in one neuron.

The readout story (additive code, matched filter, delay-line flags, a
separate max/2nd identity code) also holds in all five seeds, and all 20
delay-line links found follow the same sign rule
(`../compare/readout_generality.md`).

Of the other sizes, the zoo's 16x10 model works the same way. The blog's
16x10 model resembles the handcrafted solution (no veto, no clock). The
failing 8x10 model has no new-max detector at all.

## What isn't explained

- **Neuron-by-neuron identity code.** We know which neurons say which top-two
  input is the max (reference and clock), but not how each one's lag-dependent
  mix combines to do it at each lag.
- **The lag of the 2nd max** is encoded less cleanly than the lag of the max
  (median spread of P(on) across steps 0.06 vs 0.02).
- **The clock's internal dynamics.** Patching the clock at an intermediate
  step carries information forward (34-41% of predictions follow it), but its
  own on/off pattern mostly re-derives from the rest of the state. The exact
  wiring between the clock, reference neurons and neuron 7 isn't described.
- **How membership is revoked.** We know how a top-two flag enters the
  shift register and moves along it, but not how it is cleared when its
  input drops out of the top two.
- **Neurons 4, 8, 9** have no clear role (ablation costs under 1%, or 8% for
  neuron 8).
- **The errors themselves.** They are crosstalk: irrelevant inputs perturb
  hard comparisons through many neurons switching (see "Estimating the
  error"). There is no statistical model of that crosstalk, so no
  deductive accuracy estimate. Exact cell probabilities (E1) beat sampling
  about 5×.

## Reproducing

```bash
python scripts/inspect_model.py --hidden 32 --seq-len 10     # inspect/
python scripts/neuron_dictionary.py --hidden 32 --seq-len 10 # dictionary/
python scripts/neuron_hypotheses.py --hidden 32 --seq-len 10 # hypotheses/
python scripts/causal_tests.py --hidden 32 --seq-len 10      # causal/
python scripts/state_analysis.py --hidden 32 --seq-len 10    # state/
python scripts/lag_codes.py --hidden 32 --seq-len 10         # lag_codes/
python scripts/explain_32x10.py --hidden 32 --seq-len 10     # explain/
python scripts/readout_analysis.py --hidden 32 --seq-len 10  # readout/
python scripts/clock_test.py --hidden 32 --seq-len 10        # clock/
python scripts/wiring.py --hidden 32 --seq-len 10            # wiring/
python scripts/shift_test.py --hidden 32 --seq-len 10        # shift/
python scripts/estimation_harness.py --hidden 32 --seq-len 10 # estimation/
python scripts/tie_offsets.py --hidden 32 --seq-len 10       # ties/
python scripts/crosstalk.py --hidden 32 --seq-len 10         # crosstalk/
python scripts/compare_models.py && python scripts/summarize_causal.py  # ../compare/
python scripts/readout_generality.py                         # ../compare/
```

`summarize_causal.py` needs the causal tests for seeds 1-4 first, with each
seed's own neuron numbers. `bash scripts/run_all.sh` runs everything in the
right order.
