"""Causal test of the clock: does a patched clock state carry forward? (H4, H12)

Swap the max's and 2nd max's values. The set of top-two positions is unchanged;
only which one is the max (and so how long ago it arrived) flips, and the
correct answer moves from the 2nd's position to the old max's. For pairs where
both have arrived by step t, patch a neuron group at step t from the swapped
input and let the recurrence run on (later inputs are identical, so patching
every neuron reproduces the swapped answer exactly). Reports how often the
final prediction moves, and whether the clock's own final on/off pattern
follows the patched source or reverts to the original.

Neuron groups default to the 32x10 seed-0 roles (docs/HYPOTHESES.md §F).

Writes results/<model>/clock/summary.json.

Example:
    python scripts/clock_test.py --hidden 32 --seq-len 10
"""

import argparse
import json

import torch as th

from algzoo_interp import randn_inputs, replace_h, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--clock", nargs="+", type=int, default=[3, 10, 11, 27, 28])
    parser.add_argument("--reference", nargs="+", type=int, default=[0, 6, 13, 18, 29])
    parser.add_argument("--max-neuron", type=int, default=7)
    parser.add_argument("--membership", nargs="+", type=int, default=[2, 15, 19, 22, 26, 20, 24, 25, 30, 14])
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--n", type=int, default=2**18)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "clock"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Clock test for {name} -> {out}")

    x = randn_inputs(args.n, model.seq_len, seed=1, dtype=th.float64)
    order = x.argsort(-1, descending=True)
    p1, p2 = order[:, 0], order[:, 1]
    idx = th.arange(len(x))
    xw = x.clone()
    xw[idx, p1], xw[idx, p2] = x[idx, p2], x[idx, p1]
    with th.no_grad():
        c, cw = run_with_cache(model, x), run_with_cache(model, xw)
    informative = (c.logits.argmax(-1) == p2) & (cw.logits.argmax(-1) == p1)
    last_arrival = th.maximum(p1, p2)

    groups = {
        "clock": args.clock,
        "reference": args.reference,
        "clock + reference": args.clock + args.reference,
        "clock + reference + max detector": args.clock + args.reference + [args.max_neuron],
        "membership (control)": args.membership,
        "all neurons": list(range(model.hidden_size)),
    }
    S = {"model": name, "n_inputs": args.n, "groups": groups, "by_step": {}}
    for t in args.steps:
        sel = (last_arrival <= t) & informative
        src = run_with_cache(model, xw[sel])
        orig_clock = c.mask[sel][:, -1][:, args.clock]
        res = {"n_pairs": int(sel.sum())}
        for g, ns in groups.items():
            patched = replace_h(model, x[sel], src.h, neurons=ns, steps=[t])
            final_clock = patched.mask[:, -1][:, args.clock]
            res[g] = {
                "prediction_moves": (patched.logits.argmax(-1) == p1[sel]).double().mean().item(),
                "final_clock_matches_source": (final_clock == src.mask[:, -1][:, args.clock]).all(-1).double().mean().item(),
                "final_clock_matches_original": (final_clock == orig_clock).all(-1).double().mean().item(),
            }
        S["by_step"][t] = res
        print(f"step {t} ({res['n_pairs']} pairs):")
        for g in groups:
            r = res[g]
            print(f"   {g:34s} moves {r['prediction_moves']:.3f}   final clock = source {r['final_clock_matches_source']:.3f},"
                  f" = original {r['final_clock_matches_original']:.3f}")

    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=2)


if __name__ == "__main__":
    main()
