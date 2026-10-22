# Lessons for mechanistic interpretability

What we learned from interpreting AlgZoo's 2nd argmax RNNs that should carry
over to other small networks. Each lesson comes with the example from this
project that taught it. The examples' numbers come from the scripts named
in brackets, run on the 32x10 model (seed 0) unless the example says
otherwise. The full evidence is in the write-ups
([32x10](../results/2nd_argmax_32x10/README.md),
[4x3](../results/2nd_argmax_4x3/README.md)) and
[HYPOTHESES.md](HYPOTHESES.md).

The lessons roughly follow the order of a project: set up, find what the
neurons compute, test it causally, judge the explanation, and then explain
the errors.

## 1. Before looking at neurons

**Use the architecture's structure before using data.** A ReLU network
without biases is piecewise linear and scale invariant: fix which neurons
are on and the whole model is one matrix, and scaling an input never
changes the prediction. That turns questions about the model into geometry
(regions, walls, solid angles) and tells you that only the direction of an
input matters ([GEOMETRY.md](GEOMETRY.md)). Work out what the architecture
guarantees before you run an experiment.

**Look for symmetries the task implies.** If the task is unchanged under a
transformation, the model may have learned the matching symmetry, and
finding it halves the work. The median of three numbers is unchanged when
every input is negated. The 4x3 model's neurons form two sign-flipped
pairs, and its prediction is the same for x and −x on 99.57% of inputs
(`explain_small.py`).

**Count activation patterns to choose the level of description.** How many
distinct on/off patterns occur tells you whether a region-by-region
explanation is possible. In 2^18 inputs, 13 patterns cover 90% for 4x3, so
an exact region-by-region account worked. For 32x10 almost every input has
its own pattern (257,562 distinct), so the explanation had to be per neuron
and per step (`inspect_model.py`).

**Write down the minimal algorithm, and use it as the hypothesis space.**
Ask what state *any* solution must keep. For an online 2nd argmax, that is
the max, the 2nd max and how long ago each arrived. Fitting each neuron
within cells of that state explained a median R² of 0.94 per neuron, against
0.59 for positions alone (`state_analysis.py`). This one framing organised
most of what followed: the event detectors, the clock and the shift
register are all pieces of it.

**A hand-built solution suggests features, but the trained model may not
use them.** ARC's handcrafted 10-step model gave candidate features to test
(running max, "max minus current", delayed copies). The blog's 16x10 model
is close to a compressed handcrafted model, but 32x10 only partly follows
it: it adds a reset-started clock that the handcrafted model doesn't have
(`neuron_dictionary.py`, `compare_models.py`).

**Validate your tools on a model where you know the answer.** Before
trusting the neuron dictionary on trained models, we ran it on the
handcrafted model, and it recovered all 22 neurons with R² 1.000, each
matching its source comment (`neuron_dictionary.py`, with
`tests/test_dictionary.py`). An analysis tool that has never been checked
against ground truth can't tell a real finding from an artefact.

## 2. Finding what the network computes

**Start with the largest weights.** A large weight turns a ReLU into a
switch, which is how a smooth network implements if/then logic. Training
with cross-entropy keeps growing weights that are always right, and weight
decay shrinks the ones that don't earn their size (H7). In 32x10, 46.5% of
all recurrent weight magnitude leaves one neuron, the new-max detector, and
the largest readout weight is that detector's veto on the last position
(`neuron_hypotheses.py`). Both turned out to be central.

**Check both what a neuron does and how strongly.** A ReLU neuron has an
on/off state and a value. Neuron 7's on/off matches "new positive max" on
99.9-100% of inputs, and its value is exactly how much the max rose (R²
1.000) (`neuron_hypotheses.py`, `neuron_dictionary.py`). Testing either one
alone would have missed half the description.

**The architecture's free operations predict both features and blind
spots.** A ReLU without a bias compares against 0 for free, so every
comparison in 32x10 is against a running value clipped at 0, such as
max(0, running 2nd). That is efficient, and it is also the model's largest
blind spot: negative values can't be ranked properly, and inputs whose 2nd
largest is negative (1.08% of inputs) hold 42% of the errors
(`explain_32x10.py`).

**Many variables are codes spread across neurons.** No single neuron says
"the max arrived 5 steps ago". A clock of neurons runs a fixed on/off
sequence after each reset, and a chain of neurons passes a "this input is
in the top two" flag along like a shift register (H4, H11). A
one-neuron-one-feature dictionary finds only some of these. To find the
rest, group inputs by the hypothesised variable and see which neurons'
on/off pattern depends on it alone (`lag_codes.py`, `state_analysis.py`).

**Fit within cells, not globally.** A neuron can be exactly linear in the
values *given* the positions, and look like noise when fitted across all
positions. Fitting separately in each (lag of max, lag of 2nd) cell is
what made the state model work (`state_analysis.py`).

**Averages give the mechanism; residuals give the errors.** Averaged over
inputs, 32x10's final state is a code for the max's position plus a code
for the 2nd's (R² 0.962), read out by a matched filter. Per input, about
45% of the state depends on the values instead
(`readout_analysis.py`). The average picture explains everything the model
gets right. The per-input residual is where its errors come from, and
explaining it is a different and much harder job (section 5).

**Look at the weights between the neurons you've labelled.** Once the
per-lag flags were known, the recurrent weights between them showed a
chain (25 → 22 → 19 → 24 → 30 → 2 → 26 → 15 → 20) whose link signs follow
each neuron's polarity, written into by the event detectors
(`wiring.py`). Labelling the neurons first is what made the weight matrix
readable.

## 3. Testing claims causally

**Every claim gets an intervention.** Agreement on random inputs only shows
that a neuron *tracks* something, not that the model *uses* it. Keep a
status for each claim (causal or correlational), as HYPOTHESES.md does, so
the difference stays visible.

**Test necessity and sufficiency separately.** Patching a neuron alone tests
whether it is enough; patching *everything except* it tests whether it is
needed. Patching every neuron except the new-max detector carries the
effect of a new max only 1-5% of the time, so the detector is necessary. In
other seeds the same test gives 16-67%, so other neurons share the job
there (`causal_tests.py`, `summarize_causal.py`).

**Remove single weights to test single mechanisms.** Ablating a neuron
removes everything it does. Removing one edge tests one claim. Removing only
the detector's veto on the last logit drops accuracy on inputs whose max is
last from 99.2% to 72.4%, and changes nothing elsewhere (`causal_tests.py`).

**Expect group-level causality, and report the curve.** Patching the clock
alone moves the prediction 34% of the time; together with the reference
neurons and the detector, 85-92% (`clock_test.py`). Many real mechanisms
are like this. A greedy curve (add the most useful neuron, then the next)
shows how concentrated a mechanism is, which a single number can't.

**Know the ceiling of a resample test.** Resampling what the explanation
calls irrelevant should leave predictions unchanged, but the model's own
errors put a ceiling on that. If errors are independent near-ties, the best
possible agreement is about accuracy², 98.88% for 32x10. The resample check
reached 99.17%, so it is at the ceiling, not 0.8% short of perfect
(`explain_32x10.py`).

**Record what you refute.** Several hypotheses failed: for example, the
value neurons don't cancel the value-dependent part of the reference
neurons (correlation −0.02, `readout_analysis.py`). Recording refutations
stops you from retrying them, and tells readers which obvious stories are
wrong.

## 4. Judging an explanation

**Decide what counts as an explanation before you start.** We used four
checks: a rebuild that agrees with the model on at least 99% of inputs, a
resample check, an accuracy estimate made without sampling, and a write-up
([PLAN.md](PLAN.md#method-how-each-model-gets-analysed-and-judged)). Fixed
bars stop the explanation from being declared done because the story is
convincing.

**A surrogate that matches only on correct inputs is a precise result.** A
surrogate final state built only from the positions of the max and 2nd
agrees with 32x10 on 99.44% of inputs, and is 100% accurate: it reproduces
every correct prediction and none of the errors (`explain_32x10.py`). That
tells you exactly what the explanation covers (correct behaviour) and what
it doesn't (the errors). Measuring agreement and accuracy separately is
what reveals it.

**Test the explanation's predictions off the training distribution.**
Adding a constant to every input doesn't change the true answer but
changes how often each blind spot occurs. Reweighting the per-class error
rates by their frequency at each shift predicted the model's error, for
example 0.178 against 0.181 measured at a shift of −1
(`shift_test.py`). That is a prediction the explanation could have failed.

**Generality is at the level of mechanisms, not neurons.** Every 32x10
seed has the exact detector, its veto, the resets, a clock and the shift
register, in different neurons (`compare_models.py`,
`readout_generality.py`). Match roles by behaviour, not by index. Models of
the same size can also find different solutions: the zoo's 16x10 looks like
32x10, while the blog's 16x10 looks like the handcrafted model.

**Smaller models show which parts are needed.** The 8x6 and 8x5 models have
the detector but no clock, because there aren't enough neurons for both.
The failing 8x10 model (43% accurate) has no detector at all
(`compare_models.py`). Watching which parts disappear as capacity shrinks
tells you which ones the task requires.

**Do one small model exactly.** On 4x3, every activation-pattern region
could be enumerated and its probability computed, giving the exact
accuracy, 0.98455, against 0.98443 ± 0.00012 sampled
(`explain_small.py`). An exact case calibrates the methods you then
approximate on large models, and it can surprise you: forcing 4x3's paired
neurons to switch together makes the model *more* accurate (99.15%), so
about half its error comes from imperfect pairing.

## 5. Explaining the errors

**Errors live where comparisons are hard; stratify by those variables.**
For an ordering task those variables are the gaps between ranks, relative
to scale. Splitting inputs by the max-2nd gap as well as the 2nd-3rd gap
found a third blind spot, max-2nd near-ties, which holds 14% of the errors
(`estimation_harness.py`, `estimation/cells.md`). It had been hidden in
"everything else".

**Exact class probabilities are a cheap win.** If you can compute the
probability of each input class exactly, you only need to sample the error
within each class. For 32x10 that beats plain sampling 4.4-7.5× for the
same compute (`estimation_harness.py`). It needs only knowing which
variables matter, not how the model computes them.

**Explaining correct behaviour isn't explaining errors.** A clean circuit
can coexist with errors that it doesn't describe. In 32x10 the errors come
from imperfect cancellation across many neurons. Near a tie, the point where
the model switches answers is a weighted vote over the thresholds of all
active neurons, uncorrelated with any single neuron's threshold (|r| ≤
0.06) (`tie_offsets.py`).

**Test whether "irrelevant" inputs really are.** Keep the inputs that
decide the true answer, redraw the rest, and count changed predictions. In
every error-prone cell of 32x10, predictions changed about as often as the
model was wrong. For example, with only the max positive, the error is
17.0% and 18.2% of predictions changed. In typical inputs both are 0.02%
(`crosstalk.py`). The errors are crosstalk.

**A one-neuron error model can be badly wrong.** The top-2 detector's
threshold is biased by about 1%, which looked like the cause of the
near-tie errors. Predicting errors from that bias alone gives 8,858 errors
where the model makes 833 (`explain_32x10.py`). When several neurons decide
together, a flaw in one of them is corrected by the others.

**Gradients don't predict switches.** The gradient of the deciding logit
difference with respect to the irrelevant inputs doesn't predict how it
changes when they are redrawn (R² far below 0), and over-predicts flips,
26% against 15% (`crosstalk.py`). In a piecewise linear network the effects
that matter pass through neurons switching on or off, which a local
linearisation can't see.

**A mechanistic error estimate needs a model of the residue.** ARC's
challenge asks for an accuracy estimate that reasons from the model and
beats sampling. For 32x10, the explanation of correct behaviour gave the
classes (and the 5× gain), but a deduction of the error rate within each
class would need a statistical account of how irrelevant inputs flip
neurons across the whole state. Decide early whether the target is the
algorithm or the errors, because they need different explanations.

## 6. Practice

**Every number comes from a script, and the doc names it.** It makes every
claim reproducible, and when a script changes, `git status results/` after
`run_all.sh` shows which claims to recheck. Several numbers in our docs
were corrected this way.

**Check your baseline against theory first.** Before comparing estimators,
the harness's sampling MSE had to match p(1 − p)/N (it is within 0.99-1.02×)
and the exact method had to win clearly on 4x3 (`estimation_harness.py`).
Otherwise a "gain" may be a bug in the scoring.

**Exactness has edge cases.** Some we hit:
- float32 rounding flips switches near a wall; use float64 when exactness
  matters ([GEOMETRY.md §8](GEOMETRY.md#8-caveats));
- when every neuron is off, all logits are 0 and `argmax` returns position
  0, and an exact calculation has to follow the same tie-break
  (`tests/test_lines.py`);
- grid inputs that land exactly on a wall get spurious patterns.

**Heavy tails break pilots.** An estimator that sizes its sampling from a
small pilot run starves cells whose errors are rare but large. E2's
1,024-input pilot missed lines that were wrong along their whole length;
8,192 inputs fixed it (`estimation_harness.py`).

**Batch large evaluations.** Caching hidden states for 2^22 inputs at once
used over 10 GB and crashed the machine. Batching at 2^18 keeps the peak
at 3.5 GB.

**Keep a hypotheses ledger with stable labels.** Number hypotheses in the
order they were proposed and never renumber them, so that commit messages,
scripts and docs can all refer to "H11". Give each one a claim, evidence
and status, and keep the open questions in one place.
