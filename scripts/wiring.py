"""How the top-2 delay line is wired: a shift register written by the event detectors.

For the chain of per-lag membership neurons (lag 1 → 9), reports:
- each neuron's largest incoming recurrent weights;
- the chain weights W_hh[lag k+1 neuron, lag k neuron], whose signs follow
  the neurons' polarity (positive between same-polarity neurons, negative
  where an "on when in the top two" neuron feeds an "off when in" one);
- a causal test: cut each link (and the event detectors' input to the lag-1
  neuron) and measure the downstream neuron's on/off accuracy as a top-two
  membership flag at its lag, and the model's accuracy.

The chain defaults to the 32x10 seed-0 neurons (docs/HYPOTHESES.md §G).
On/off accuracy is a crude measure for neurons whose amplitude, not on/off
state, carries membership (26, 15).

Writes results/<model>/wiring/summary.json.

Example:
    python scripts/wiring.py --hidden 32 --seq-len 10
"""

import argparse
import json

import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import ablate_edges, get_weights, randn_inputs, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR

# (neuron, lag, on when the input at that lag is in the top two?)
CHAIN = [(25, 1, False), (22, 2, True), (19, 3, True), (24, 4, False), (30, 5, False),
         (2, 6, True), (26, 7, True), (15, 8, True), (20, 9, False)]
EVENT_DETECTORS = [16, 7]


def membership(x, T):
    """For each lag k: list over t = k..T-1 of [batch] bools, x_{t-k} in the running top two."""
    out = {}
    for k in range(1, T):
        rows = []
        for t in range(k, T):
            o = x[:, : t + 1].argsort(-1, descending=True)
            rows.append((o[:, 0] == t - k) | (o[:, 1] == t - k))
        out[k] = rows
    return out


def flag_accuracy(cache, neuron, lag, on_when_in, member):
    T = cache.mask.shape[1]
    hits = total = 0
    for i, t in enumerate(range(lag, T)):
        on = cache.mask[:, t, neuron]
        hits += ((on if on_when_in else ~on) == member[lag][i]).sum().item()
        total += len(on)
    return hits / total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**17)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "wiring"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Wiring of {name} -> {out}")
    W_hh = get_weights(model).W_hh.detach()
    T = model.seq_len

    incoming = {}
    for n, k, _ in CHAIN:
        top = W_hh[n].abs().argsort(descending=True)[:6]
        incoming[n] = {"lag": k, "largest_incoming": [(int(j), round(W_hh[n, j].item(), 3)) for j in top]}
    links = [{"from": a, "to": b, "same_polarity": pa == pb, "weight": W_hh[b, a].item()}
             for (a, _, pa), (b, _, pb) in zip(CHAIN[:-1], CHAIN[1:])]

    x = randn_inputs(args.n, T, dtype=th.float64)
    target = task_2nd_argmax(x)
    member = membership(x, T)
    with th.no_grad():
        base = run_with_cache(model, x)
    acc0 = (base.logits.argmax(-1) == target).double().mean().item()
    cuts = {f"{CHAIN[0][0]} <- event detectors {EVENT_DETECTORS}": (CHAIN[0], [(CHAIN[0][0], d) for d in EVENT_DETECTORS])}
    for (a, _, _), node in zip(CHAIN[:-1], CHAIN[1:]):
        cuts[f"{node[0]} <- {a}"] = (node, [(node[0], a)])
    cut_results = {}
    for label, ((n, k, pol), edges) in cuts.items():
        with th.no_grad():
            cut = run_with_cache(ablate_edges(model, hh=edges), x)
        cut_results[label] = {
            "flag_accuracy_before": flag_accuracy(base, n, k, pol, member),
            "flag_accuracy_after": flag_accuracy(cut, n, k, pol, member),
            "model_accuracy_after": (cut.logits.argmax(-1) == target).double().mean().item(),
        }
        r = cut_results[label]
        print(f"cut {label:32s}: flag accuracy {r['flag_accuracy_before']:.3f} -> {r['flag_accuracy_after']:.3f};"
              f" model accuracy {acc0:.4f} -> {r['model_accuracy_after']:.4f}")

    S = {"model": name, "n_inputs": args.n, "chain": CHAIN, "incoming": incoming, "links": links,
         "model_accuracy": acc0, "cuts": cut_results}
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=2)
    for link in links:
        print(f"link {link['from']:2d} -> {link['to']:2d}: weight {link['weight']:+.3f} (same polarity: {link['same_polarity']})")


if __name__ == "__main__":
    main()
