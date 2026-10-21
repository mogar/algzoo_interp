# algzoo Experiments

Experiments in interpreting the NNs from [algzoo](https://github.com/alignment-research-center/alg-zoo).


## Setup

For environment:

```bash
git submodule update --init
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs the `alg-zoo` submodule in editable mode. `pip install -e .` installs this repo's `algzoo_interp` package. Downloaded checkpoints are cached in `weights/`.

Run the tests with `pytest`. Experiments are plain scripts in `scripts/` that write figures and numbers to `results/`, e.g.:

```bash
python scripts/zoo_baseline.py
```

## Plan

See [docs/PLAN.md](docs/PLAN.md) for the full plan. In short: unroll the RNNs by hand with hooks, validate the tools on the handcrafted models, explain trained models from smallest to largest, and check each explanation quantitatively (hand-rebuilt model, resampling checks, accuracy estimate made from the mechanism).

## Starting Experiments

Algzoo contains small networks that model various functions. Can we figure out what each neuron is doing?

Some context:

* https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
* https://x.com/JacobHHilton/status/2110915999356817588
* https://github.com/alignment-research-center/alg-zoo


The 2nd argmax models are ReLU RNNs, so we unroll them by hand in PyTorch, using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) hook points where they help. TransformerLens's `HookedTransformer` becomes relevant for the longest-cycle transformers. We will start with the 2nd argmax model. Ideally we will explain the largest model (hidden size 32), but it's ok to start with a smaller one.
