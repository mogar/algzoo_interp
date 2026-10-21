# Working hypotheses: 2nd argmax 32x10

Hypotheses about how the 32x10 model (1,376 parameters, 99.5% accuracy) works,
drawn from the activity plot (`results/2nd_argmax_32x10/inspect/activity.png`)
and the weight plot (`.../inspect/weights.png`). Each entry gives the claim,
the evidence so far, its status, and how we'll test it.

All numbers come from `python scripts/neuron_hypotheses.py --hidden 32 --seq-len 10`
on 2^17 `randn` inputs, saved to `results/2nd_argmax_32x10/hypotheses/summary.json`.
Neuron numbers refer to this model only: other models and seeds will number
their neurons differently.

**Status key.** *Correlational*: the neuron's behaviour matches the claim on
random inputs, but we haven't yet shown the model *uses* it that way.
*Causal*: confirmed by ablation or patching. *Speculative*: a guess to guide
what we look at next.

## A. Activity over time: two kinds of neuron

![Activity Plot](../results/2nd_argmax_32x10/inspect/activity.png)

The activity plot shows, for each neuron and step, the fraction of inputs
where the neuron is on. Rows differ in two ways.

**Step 0 is uniform (all ≈ 0.5): explained.** At t=0 there is no previous
state, so `pre_0 = W_ih * x_0` and each neuron is on exactly when `x_0` has the
same sign as its input weight. That happens half the time for every neuron.

From step 1 on, neurons split into two groups.

### H1. Neuron 7 detects a new positive running max

**Claim.** Neuron 7 is on at step t exactly when `x_t > max(0, x_0, ..., x_{t-1})`.

**Evidence.** Its on/off state agrees with that event on 99.9-100% of inputs
at every step. Its firing rate falls like the event's probability:

| t | 0 | 1 | 2 | 3 | 5 | 9 |
|---|---|---|---|---|---|---|
| P(neuron 7 on) | 0.498 | 0.376 | 0.292 | 0.234 | 0.165 | 0.101 |
| P(event) | 0.498 | 0.376 | 0.292 | 0.233 | 0.164 | 0.100 |

The "and > 0" matters at early steps (e.g. P = 3/8 at t=1, not 1/2). Without
it, agreement drops to 87% at t=1. Comparing against `max(0, ...)` is what a
ReLU unit with no bias does naturally, and the handcrafted model uses the same
convention.

It also has the largest input weight in the model (`W_ih[7]` = 8.8; the median
|W_ih| is 0.14).

**Status.** Correlational. Very strong.

### H2. Neuron 16 detects a new positive top-2 value

**Claim.** Neuron 16 is on at step t exactly when `x_t > 0` and `x_t` is one
of the two largest of `x_0..x_t`.

**Evidence.** Agreement 99.6-100% at every step; firing rate 0.203 against
event probability 0.199 at t=9. Second-largest input weight (`W_ih[16]` = 5.7).

**Status.** Correlational. Very strong.

### H3. Always-on neurons are linear memory for running values

**Claim.** Neurons that are on for (almost) every input stop switching, so
they act as plain linear units. That suits carrying a continuous value
forward through time. They store running statistics of the sequence.

**Evidence.** Neurons 5, 8, 17 and 31 are on for at least 95% of inputs at the
last step (most from step 2 or 3 on). Fitting each neuron's final value
against single candidate quantities (R² of the best straight-line fit):

| neuron | best match | R² | runner-up |
|---|---|---|---|
| 17 | max so far | 0.84 | 2nd max so far, 0.52 |
| 31 | 2nd max so far | 0.79 | max so far, 0.43 |
| 5 | max − 2nd max | 0.52 | max so far, 0.35 |
| 8 | (none fits) | ≤ 0.14 | |

**Status.** Correlational, partial. The single-feature fits leave a lot
unexplained, so these neurons probably mix several quantities. Neuron 8 is
unexplained.

### H4. The variable neurons encode *where*, not just *what* (speculative)

**Claim.** The task asks for the **position** of the 2nd largest value, but
H1-H3 only track values and events. Something has to remember *when* the
relevant events happened. The neurons whose activity changes from step to step
(neither event-like nor always on) may encode time since the last new max or
new top-2, for example by growing or decaying at a steady rate until reset.

**Evidence.** Indirect only. The readout has to separate 10 positions, and
`W_out` spreads moderate weights over many neurons and positions (e.g. neuron
21 has weights of both signs across positions 4-7: +8.0, −9.8, +12.8, −7.4), which fits a distributed
"when" code.

**Status.** Speculative.

**Also open.** Neuron 20 is on for 75-99% of inputs from steps 2-8 but drops
to 85% at t=9. Something specific happens at the last step.

## B. Large weights implement discrete logic

Typical weights are small: median |W_hh| is 0.095 and median |W_out| is 1.1.
A few are far larger (up to about 320x the median in W_hh and 60x in W_out),
and almost all of them involve neurons 7 and 16.

### H5. Neurons 7 and 16 send "reset" signals

**Claim.** When a new max arrives, neuron 7 forces a set of neurons to zero,
clearing state that the new max makes stale. Neuron 16 does the same for a
smaller set when a new top-2 value arrives.

**Evidence.**
- **46.5%** of all recurrent weight magnitude (sum of |W_hh|) leaves neuron 7,
  and **10.4%** leaves neuron 16. The next-largest neuron accounts for 2.6%.
- The largest recurrent weights are all negative and come from these two:
  7→29 (−30.7), 7→13 (−30.5), 7→0 (−24.7), 16→3 (−13.7), 7→28 (−13.7),
  7→3 (−11.6), 7→11 (−10.5), 16→25 (−9.9), ...
- When neuron 7 fires, its value is typically about 5 (median when on), so it
  pushes its targets' pre-activations down by roughly 50-150. That's far more
  than anything else feeds them, so they are forced off.

**Status.** Plausible from the weights, untested.

**Prediction.** Neurons 29, 13 and 0 are (almost) exactly zero at step t+1
whenever neuron 7 is on at step t. Patching neuron 7 "on" at step t on an
input where no new max occurred should zero them, and change the prediction
in the way a new max would.

### H6. Neuron 7 vetoes position 9 in the readout

**Claim.** The largest weight in the model, `W_out` from neuron 7 to logit 9
(**−65.1**), is a veto: if the last input is a new max, position 9 is the
largest value, so it cannot be the 2nd largest.

**Evidence.** The logic fits H1 exactly, and this is a rule that is always
right, so cross-entropy training would keep strengthening it. Neuron 12 →
logit 9 (**−39.7**) is a second large negative weight on the same logit. Its
role is unknown; it may be a related veto (neuron 12's input weight is −3.1,
so it reacts to *small* final inputs).

**Status.** Plausible, untested.

**Prediction.** Removing just this one weight's contribution (leaving neuron 7
in place elsewhere) causes errors mainly on inputs whose max is at position 9,
predicting 9 instead of the true 2nd argmax.

### H7. Why the important weights are the large ones

Not a claim about one neuron but a reason to expect this pattern:

1. **Large gains turn ReLUs into switches.** A big weight makes a neuron jump
   from 0 to large over a tiny input change, which approximates if/then logic
   such as "is x_t bigger than the stored max?" The task is logical, so the
   model needs switches.
2. **Cross-entropy rewards confident margins,** so weights behind
   always-correct rules (vetoes, resets) keep growing.
3. **Weight decay (0.001) shrinks weights that don't earn their size,** so a
   large weight that survives training is probably doing real work. This fits
   the ablations: removing neuron 7 or 16 costs 75-90% accuracy.

## C. Overall sketch (speculative)

Putting H1-H6 together, the model may be running an online algorithm much like
the handcrafted 10-step model:

1. **Detect events** each step: new max (neuron 7), new top-2 (neuron 16).
2. **Hold running values** in always-on neurons: max so far (17), 2nd max so
   far (31), their gap (5).
3. **Reset stale state** when an event fires (large negative W_hh from 7 and 16).
4. **Track when** things happened, in the time-varying neurons (H4).
5. **Read out** the position, with hard vetoes for impossible answers (H6).

The hand-built model (`handcrafted_2nd_argmax(10)`) uses a different but
related scheme: a shift register of past inputs, plus running max and
"max minus current" neurons. Comparing the two is part of Phase 2.

## Next tests, in order

1. **Causal check of H1/H2.** Patch neuron 7 (or 16) at a single step from an
   input where the event did or didn't happen, and measure how the prediction
   changes and which neurons respond.
2. **H5 resets.** Measure target neurons' values at t+1 conditioned on neuron 7
   / 16 firing at t. Then patch.
3. **H6 veto.** Subtract the 7→logit 9 contribution and re-measure accuracy,
   split by argmax position.
4. **H3 memory.** Fit the always-on neurons against several quantities at once
   (multiple regression) and test causally: does shifting neuron 17 by δ act
   like raising the stored max by δ/slope?
5. **H4 timing.** For the time-varying neurons, plot value against "steps
   since last new max" and "steps since last new top-2".
6. **Generality.** Rerun `neuron_hypotheses.py` on other seeds
   (`--seed 0..4`, needs a flag) and on 16x10, to see whether the same roles
   appear under different neuron numbers.
