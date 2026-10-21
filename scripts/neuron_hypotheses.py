"""Evidence for the neuron-role hypotheses in docs/HYPOTHESES.md.

For one model, measures:
- how well each neuron's on/off state matches candidate events
  ("x_t is a new running max", "x_t is in the running top 2"), per step;
- which neurons are almost always on, and how well single running quantities
  (max so far, 2nd max so far, their gap, ...) linearly predict them;
- the largest weights in W_hh and W_out relative to typical magnitudes.

Prints a report and writes results/<model>/hypotheses/summary.json.

Example:
    python scripts/neuron_hypotheses.py --hidden 32 --seq-len 10
"""

import argparse
import json
from pathlib import Path

import torch as th

from algzoo_interp import get_weights, load_2nd_argmax, randn_inputs, run_with_cache
from algzoo_interp.features import events, running_top2

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def named_events(x):
    """The four running-order events (leaves out the per-lag sign events)."""
    ev = events(x)
    return {k: ev[k] for k in ["new_max", "new_max_pos", "new_top2", "new_top2_pos"]}


def quantities(x):
    """Continuous [batch, seq_len] running values a 2nd argmax algorithm might store."""
    top1, top2 = running_top2(x)
    top2 = th.where(th.isinf(top2), th.zeros_like(top2), top2)
    prev = th.cat([th.zeros_like(x[:, :1]), x[:, :-1]], 1)
    return {
        "x_t": x,
        "x_{t-1}": prev,
        "max_so_far": top1,
        "2nd_max_so_far": top2,
        "max-2nd": top1 - top2,
        "max-x_t": top1 - x,
    }


def r2(y, f):
    """R^2 of the best affine fit y ~ a*f + b."""
    X = th.stack([f, th.ones_like(f)], 1)
    coef = th.linalg.lstsq(X, y[:, None]).solution
    return (1 - ((X @ coef)[:, 0] - y).var() / y.var()).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--n", type=int, default=2**17)
    parser.add_argument("--always-on", type=float, default=0.95,
                        help="min fraction active at the last step to count as always on")
    parser.add_argument("--top-weights", type=int, default=12)
    args = parser.parse_args()

    model = load_2nd_argmax(args.hidden, args.seq_len)
    name = f"2nd_argmax_{args.hidden}x{args.seq_len}"
    x = randn_inputs(args.n, args.seq_len)
    with th.no_grad():
        cache = run_with_cache(model, x)
        w = get_weights(model)
    mask = cache.mask
    T, H = args.seq_len, args.hidden

    # Event detectors: for each event, the neuron whose on/off best matches it at the last step
    ev = named_events(x)
    detectors = {}
    print("== Event detectors (agreement of neuron on/off with event) ==")
    for ev_name, e in ev.items():
        agree_last = (mask[:, -1] == e[:, -1, None]).float().mean(0)
        best = int(agree_last.argmax())
        per_step = (mask[:, :, best] == e).float().mean(0)
        detectors[ev_name] = {
            "neuron": best,
            "agreement_by_step": [round(v, 4) for v in per_step.tolist()],
            "p_active_by_step": [round(v, 4) for v in mask[:, :, best].float().mean(0).tolist()],
            "p_event_by_step": [round(v, 4) for v in e.float().mean(0).tolist()],
        }
        print(f"{ev_name}: best neuron {best}")
        for t in range(T):
            print(f"  t={t}: agree={per_step[t]:.3f}  P(active)={mask[:, t, best].float().mean():.3f}"
                  f"  P(event)={e[:, t].float().mean():.3f}")

    # Always-on neurons and what running quantity they track
    p_last = mask[:, -1].float().mean(0)
    always_on = [int(i) for i in th.where(p_last >= args.always_on)[0]]
    q = quantities(x)
    trackers = {}
    print(f"\n== Always-on neurons (active on >= {args.always_on:.0%} of inputs at the last step) ==")
    for n in always_on:
        fits = {k: round(r2(cache.h[:, -1, n], v[:, -1]), 3) for k, v in q.items()}
        best = max(fits, key=fits.get)
        trackers[n] = {
            "p_active_by_step": [round(v, 3) for v in mask[:, :, n].float().mean(0).tolist()],
            "r2_last_step": fits,
            "best": best,
        }
        print(f"  neuron {n:2d}: best single fit {best} (R^2 {fits[best]:.2f}); all: {fits}")

    # Large weights
    def top_entries(W, k, row_name, col_name):
        vals, idx = W.abs().flatten().topk(k)
        return [{row_name: int(i // W.shape[1]), col_name: int(i % W.shape[1]),
                 "weight": round(W.flatten()[i].item(), 2)} for i in idx]

    big = {
        "median_abs": {"W_ih": round(w.W_ih.abs().median().item(), 3),
                       "W_hh": round(w.W_hh.abs().median().item(), 3),
                       "W_out": round(w.W_out.abs().median().item(), 3)},
        "W_ih_top": top_entries(w.W_ih[:, None], 5, "neuron", "_"),
        "W_hh_top": top_entries(w.W_hh, args.top_weights, "to", "from"),
        "W_out_top": top_entries(w.W_out, args.top_weights, "logit", "neuron"),
    }
    # How much of each neuron's outgoing recurrent weight mass it accounts for
    col_mass = w.W_hh.abs().sum(0)
    big["W_hh_outgoing_share"] = {int(n): round((col_mass[n] / col_mass.sum()).item(), 3)
                                  for n in col_mass.argsort(descending=True)[:5]}
    print("\n== Large weights ==")
    print("  median |w|:", big["median_abs"])
    print("  largest W_ih:", [(e["neuron"], e["weight"]) for e in big["W_ih_top"]])
    print("  largest W_hh (to <- from):", [(e["to"], e["from"], e["weight"]) for e in big["W_hh_top"]])
    print("  largest W_out (logit <- neuron):", [(e["logit"], e["neuron"], e["weight"]) for e in big["W_out_top"]])
    print("  share of total |W_hh| leaving each neuron:", big["W_hh_outgoing_share"])

    out = RESULTS_DIR / name / "hypotheses"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"model": name, "n_inputs": args.n, "event_detectors": detectors,
                   "always_on": trackers, "large_weights": big}, f, indent=2)
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
