# algzoo Experiments

Mechanistic interpretability of the tiny neural networks in ARC's
[AlgZoo](https://github.com/alignment-research-center/alg-zoo). Our focus has
been the **2nd argmax RNNs**: one-layer ReLU RNNs that read a sequence of
numbers and point at the position of the second-largest one. The main target
is the largest model, with 32 hidden neurons and sequence length 10 (1,376
parameters).

## What we found

The 32x10 model runs an online algorithm close to the minimal one. After
each input it effectively knows the largest and 2nd-largest values so far and
how many steps ago each arrived.

- **Event detectors.** One neuron outputs exactly how much the positive
  running max just increased. Another fires when the input is a new top-2
  value.
- **Stored values.** Neurons hold the running max, the running 2nd max and
  the gap between them.
- **Resets and a clock.** Each new max zeroes about ten neurons, which then
  run a fixed on/off sequence that encodes how long ago the max arrived.
- **A position code and a matched filter.** On average the final state is a
  sum of a code for where the max is and a code for where the 2nd max is. A
  shift register, written by the event detectors, marks which recent inputs
  are in the top two. A separate code says which of the two is the max, and
  the readout picks out the 2nd-max code.
- **Vetoes.** The two largest readout weights rule out the last position when
  it is too big (the new max) or too small (below the 2nd max).
- **Checks.** A surrogate built only from the positions of the max and 2nd
  max reproduces all of the model's correct predictions. 94% of its errors
  fall in three identified blind spots: negative values, and two kinds of
  near-tie. The first two also predict how the error rate changes when a
  constant is added to every input.

These mechanisms recur across all five training seeds. We also have a
complete, exact explanation of a 4-neuron, 3-input model, including its exact
accuracy computed from geometry rather than sampling.

## Where to start reading

1. **[results/2nd_argmax_32x10/README.md](results/2nd_argmax_32x10/README.md):**
   the main write-up. How the 32x10 model works, the quantitative checks, and
   what isn't explained yet.
2. **[results/2nd_argmax_4x3/README.md](results/2nd_argmax_4x3/README.md):**
   a small model explained completely, with exact accuracy from geometry.
   Short, and a good introduction to the methods.
3. **[docs/GEOMETRY.md](docs/GEOMETRY.md):** background on why these
   networks are piecewise linear, and what "activation patterns" and "cones"
   are. Read this if the write-ups use terms you don't recognise.
4. **[docs/HYPOTHESES.md](docs/HYPOTHESES.md):** the detailed evidence for
   every claim about 32x10, with a status table up front, a table of every
   neuron's role (§F), and how the findings generalise across seeds and
   model sizes (§G).
5. **[docs/PLAN.md](docs/PLAN.md):** how explanations are judged, how far
   each step got, and the list of open questions.

## Getting started

Requires Python 3.10+. A GPU isn't needed: everything runs on a CPU in
seconds to a minute.

```bash
git clone --recurse-submodules https://github.com/mogar/algzoo_interp.git
cd algzoo_interp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # includes the alg-zoo submodule, in editable mode
pip install -e .                  # this repo's algzoo_interp package
pytest                            # about 2 seconds
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`
first.

`requirements.txt` lists only the direct dependencies. On Linux, plain
`pip install torch` pulls several GB of CUDA libraries the code doesn't need.
To get the much smaller CPU build, install torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Model checkpoints download on first use from ARC's public storage bucket (no
credentials needed) and are cached in `weights/`.

To check everything works and get a first look at a model:

```bash
python scripts/inspect_model.py --hidden 32 --seq-len 10
```

This writes figures (weights, accuracy breakdowns, neuron activity,
ablations) to `results/2nd_argmax_32x10/inspect/`.

Most scripts accept the same model flags: `--hidden H --seq-len T` for a zoo
model, `--seed N` for a training seed other than the best, `--example` for the
16x10 model from ARC's blog post, and `--handcrafted 10` for ARC's
hand-built solution.

## Tests and reproducing results

```bash
pytest                       # library tests, about 2 seconds
pytest -m scripts            # runs every script end to end on small inputs, about 30 seconds
bash scripts/run_all.sh      # regenerates every committed result, a few minutes
git status results/          # after run_all: lists any result that changed
```

The script smoke tests write to a temporary directory, so they never touch
`results/`. Set `ALGZOO_RESULTS_DIR` to send any script's output somewhere
else.

## Repo layout

```
algzoo_interp/   library code
scripts/         one runnable script per experiment; each writes to results/
results/         figures, JSON/CSV numbers and write-ups, one folder per model
docs/            plan, hypotheses and background notes
tests/           pytest checks for the library, plus opt-in script smoke tests
alg-zoo/         ARC's AlgZoo (git submodule)
weights/         downloaded checkpoints (gitignored)
```

### Library (`algzoo_interp/`)

| module | what it does |
|---|---|
| `hooked.py` | Runs the RNN step by step, recording every hidden state, pre-activation and on/off mask, with optional hooks to edit them |
| `interventions.py` | Ablation, activation patching, and removal of individual weights |
| `features.py`, `dictionary.py` | Candidate features (running max, new-max events, ...) and matching each neuron against them |
| `state.py` | How much of each neuron is explained by the ideal online state (the max, 2nd max and their positions) |
| `patterns.py`, `cones.py` | Activation patterns, the exact linear map for each, and exact cone enumeration and accuracy for 3-input models |
| `loading.py`, `data.py`, `eval.py`, `plots.py` | Cached model loading, input generators, accuracy breakdowns, shared plot style |

### Scripts (`scripts/`)

| script | question it answers |
|---|---|
| `zoo_baseline.py` | How accurate is every 2nd argmax model in the zoo? |
| `inspect_model.py` | First look at any model |
| `neuron_dictionary.py` | What does each neuron compute? |
| `causal_tests.py` | Do the event detectors, resets and vetoes actually matter? (patching and weight removal) |
| `state_analysis.py`, `lag_codes.py` | How does the model encode *where* the max and 2nd max are? |
| `compare_models.py`, `summarize_causal.py` | Do the mechanisms recur across seeds and sizes? |
| `explain_small.py`, `exact_accuracy.py` | Full explanation and exact accuracy for the 4x3 model |
| `explain_32x10.py` | Rebuild, resample and error checks for 32x10 |
| `readout_analysis.py` | How does the readout decode the 2nd max's position? |
| `clock_test.py` | Does a patched clock state carry forward through the recurrence? |
| `wiring.py` | How is the top-two delay line wired? |
| `shift_test.py` | Do the error mechanisms predict errors on shifted inputs? |
| `readout_generality.py` | Does the readout story hold for other seeds and sizes? |
| `estimation_harness.py` | How well does an error estimator do against sampling, per unit of compute? |
| `plot_regions_2d.py`, `neuron_hypotheses.py` | Figures for GEOMETRY.md, and early evidence for HYPOTHESES.md |
| `run_all.sh` | Regenerate every committed result |

Each script's docstring explains its method and outputs (`python scripts/<name>.py --help`).

## Conventions

- **No notebooks.** Experiments are plain scripts that save image plots and
  JSON/CSV numbers. If interactive views become necessary, they'll go in a
  Streamlit app under `app/`.
- **Every number in a doc comes from a script**, and the doc says which.
- **Neuron numbers are model-specific.** "Neuron 7" means neuron 7 of the
  default 32x10 model (seed 0); other seeds put the same roles in different
  neurons.
- **Plain PyTorch, not TransformerLens.** TransformerLens is installed, but its
  `HookedTransformer` doesn't fit RNNs, so the RNN is unrolled by hand in
  `hooked.py`. It would become useful for AlgZoo's longest-cycle
  transformers.

## Background

- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) (ARC blog post)
- [Jacob Hilton's thread](https://x.com/JacobHHilton/status/2110915999356817588)
- [AlgZoo on GitHub](https://github.com/alignment-research-center/alg-zoo)
