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

**Update (dictionary, roadmap step 1).** Its *value* is pinned down too:
`h_t[7] = max0 - max0_prev` with R² = 1.000, where `max0 = max(0, x_0..x_t)`.
That is, it outputs how much the positive running max just increased, which is
zero exactly when there is no new positive max. This is the same quantity as
handcrafted neuron 19 (`results/2nd_argmax_32x10/dictionary/dictionary.md`).

**Causal test (roadmap step 2).** Pairs of inputs differing only at step t,
where the src input's `x_t` is raised to a new positive max (`results/2nd_argmax_32x10/causal/` (`python scripts/causal_tests.py --hidden 32 --seq-len 10`)).
Because later inputs are identical, patching every neuron at step t gives
src's prediction exactly; patching subsets shows which neurons carry the
change. Flip rate = how often the patched prediction becomes src's:

| patched at step t (t = 2..8) | add a new max | remove a new max |
|---|---|---|
| everything except neuron 7 | 1-5% | 16-18% |
| neuron 7 alone | 51-57% | 29-59% |
| neurons 7 and 16 | 58-99% | 30-76% |

Without neuron 7 the rest of the hidden state almost never carries the effect
of a new max, so neuron 7 is **necessary**. It isn't sufficient alone: a new
max is also a new top-2 value, so neuron 16 fires too, and 7 + 16 together
carry 99% of the effect at t=2. Their share falls at later steps as other
neurons carry more (patching everything except 7 and 16 rises from 12% at t=2
to 45% at t=8). At the last step (t=9) the readout reads every neuron directly,
so no small subset dominates.

**Status.** Causal: necessary for the downstream effect of a new max.

### H2. Neuron 16 detects a new positive top-2 value

**Claim.** Neuron 16 is on at step t exactly when `x_t > 0` and `x_t` is one
of the two largest of `x_0..x_t`.

**Evidence.** Agreement 99.6-100% at every step; firing rate 0.203 against
event probability 0.199 at t=9. Second-largest input weight (`W_ih[16]` = 5.7).

**Causal test (roadmap step 2).** Same design, with src's `x_t` moved
above or below the running 2nd max without becoming a new max. Patching
neuron 16 alone reproduces src's prediction 83-93% of the time at t=2..7
when adding a top-2 value, and 74-89% at t=2..8 when removing one. Patching
everything except neurons 7 and 16 manages only 1-6% at t=2..7. Neuron 7 doesn't change in these pairs (no new max), so
patching it alone does nothing (0%).

**Status.** Causal: necessary and largely sufficient for the effect of a new
top-2 value.

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

**Update (dictionary, roadmap step 1).** Pooling the fit over steps 1-9
(rather than the last step only) and adding the `max0` convention sharpens
neuron 17: `h_t[17] ≈ max0_prev = max(0, x_0..x_{t-1})` with R² = 0.986, the
same quantity as handcrafted neuron 20. Neurons 5, 8 and 31 still have no
single-feature match (best R² 0.46, 0.40, 0.64), but linear fits on all
candidate features reach 0.93, 0.94 and 0.93.

**Update (step 2).** Adding the feature `2nd0_prev = max(0, 2nd largest of
x_0..x_{t-1})` identifies neuron 31: R² = 0.963. So the model keeps both a
running max (17) and a running 2nd max (31).

**Status.** Correlational. Neurons 17 and 31 are strong; 5 and 8 look like
mixtures of several running values.

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

**Prediction.** Neurons 29, 13 and 0 are (almost) exactly zero at step t+1
whenever neuron 7 is on at step t.

**Test (roadmap step 2).** Targets are neurons with `W_hh[target, source] < -5`
(`results/2nd_argmax_32x10/causal/` (`python scripts/causal_tests.py --hidden 32 --seq-len 10`), `resets.png`):

- **The prediction holds for every target.** P(target = 0 at t+1 | source on
  at t) is 0.98-1.00 for all 10 targets of neuron 7 and both targets of neuron
  16, against 0.004-0.76 when the source is off.
- **Cutting the edges shows which resets the edges cause.** With 7's
  reset edges removed, targets 0, 13, 28 and 29 fall back to their "source
  off" rates (P(0 | on) = 0.30, 0.15, 0.01, 0.23): those resets are caused by
  neuron 7. Targets 3, 10 and 25 stay zeroed (0.998-1.000): something else
  also zeroes them. For 3 and 25 that's partly neuron 16, which fires at
  every new max too. Targets 11, 23 and 27 are in between.
- **The resets matter.** Cutting 7's reset edges drops accuracy from 99.5% to
  67.4%; cutting 16's two drops it to 72.3%.

**Status.** Causal.

### H6. Neuron 7 vetoes position 9 in the readout

**Claim.** The largest weight in the model, `W_out` from neuron 7 to logit 9
(**−65.1**), is a veto: if the last input is a new max, position 9 is the
largest value, so it cannot be the 2nd largest.

**Evidence.** The logic fits H1 exactly, and this is a rule that is always
right, so cross-entropy training would keep strengthening it. Neuron 12 →
logit 9 (**−39.7**) is a second large negative weight on the same logit. Its
role is unknown; it may be a related veto (neuron 12's input weight is −3.1,
so it reacts to *small* final inputs).

**Prediction.** Removing just this one weight's contribution (leaving neuron 7
in place elsewhere) causes errors mainly on inputs whose max is at position 9,
predicting 9 instead of the true 2nd argmax.

**Test (roadmap step 2).** `results/2nd_argmax_32x10/causal/` (`python scripts/causal_tests.py --hidden 32 --seq-len 10`), `veto.png`. Removing the 7 → logit 9
weight drops accuracy from 99.5% to 96.8%, and accuracy on inputs whose max is
at position 9 from 99.2% to 72.4%. 86% of the new errors have the max at
position 9, and 88% predict position 9. Accuracy at every other max position
is unchanged.

**Status.** Causal. Prediction confirmed.

### H6b. Neuron 12 vetoes position 9 when x_9 is too small

**Claim.** Neuron 12 computes `relu(max(0, 2nd largest of x_0..x_{t-1}) - x_t)`,
how far the current input falls below the running 2nd max, and its readout
weight to logit 9 (**−39.7**) rules out position 9 when `x_9` is too small to
be the 2nd largest. Together with H6, logit 9 is vetoed exactly when `x_9` is
above the max (neuron 7) or below the 2nd max (neuron 12).

**Evidence.** Dictionary fit R² = 0.998 (feature `relu(2nd0_prev-x_t)`,
`results/2nd_argmax_32x10/dictionary/dictionary.md`). Removing the 12 → logit 9
weight drops accuracy to 94.8%. 94% of the new errors wrongly predict position
9, but almost none have the max at 9 (1%). Among wrong predictions of 9, `x_9`
is usually just below the true 2nd max: it is the 3rd largest in 2,250 of
3,219 cases, then 4th (717), 5th (192).

**Status.** Value: correlational (very strong). Veto: causal.

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
   far (31), and a mixture including their gap (5).
3. **Reset stale state** when an event fires (large negative W_hh from 7 and 16).
4. **Track when** things happened, in the time-varying neurons (H4).
5. **Read out** the position, with hard vetoes for impossible answers: for
   the last position, "too big" (neuron 7, H6) and "too small" (neuron 12, H6b).

Steps 1, 3 and 5 now have causal support (roadmap step 2).

The hand-built model (`handcrafted_2nd_argmax(10)`) uses a different but
related scheme: a shift register of past inputs, plus running max and
"max minus current" neurons. Roadmap step 1 found the overlap is concrete:
32x10 neurons 7 and 17 compute exactly the handcrafted running-max register
(handcrafted neurons 19 and 20), and a linear fit on all 22 handcrafted
neurons explains a median R² of 0.83 of each 32x10 neuron's value (0.95 for
the 16x10 blog model). See the step 1 results in [PLAN.md](PLAN.md#roadmap). Comparing the two is part of roadmap step 4 in [PLAN.md](PLAN.md#roadmap).

## Testing

The order in which these hypotheses get tested, alongside the rest of the
project, lives in the roadmap in [PLAN.md](PLAN.md#roadmap). Update each
hypothesis's **Status** here as results come in.
