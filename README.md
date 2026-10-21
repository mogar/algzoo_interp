# algzoo Experiments

Experiments in interpreting the NNs from [algzoo](https://github.com/alignment-research-center/alg-zoo).


## Setup

For environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


```




## Starting Experiments

Algzoo contains small networks that model various functions. Can we figure out what each neuron is doing?

Some context:

* https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
* https://x.com/JacobHHilton/status/2110915999356817588
* https://github.com/alignment-research-center/alg-zoo


The plan is to explore these using the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) project. We will start with the 2nd argmax model. Ideally we will explain the largest model (hidden size 32), but it's ok to start with a smaller one.
