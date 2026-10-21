"""Does the 32x10 seed-0 readout story hold for other models? (H9-H12; docs/HYPOTHESES.md §G)

For each 10-step model, without using any neuron numbers from seed 0:
- additive position code: R² of mean final state ≈ a(max position) +
  b(2nd position), and agreement of the a(i) + b(j) surrogate with the model;
- matched filter: the smallest margin by which W·b(j) favours logit j
  (positions 0-8), and the mean diagonal of W·a(i) (positions 0-8);
- value-dependent residual: its median effect on the decisive margin relative
  to the margin, and its share in the readout-visible subspace vs a random
  direction;
- membership flags: per lag, the best on/off neuron for "the input k steps
  ago is in the top two", and how many lags have a flag clearly better than
  the majority baseline (accuracy - baseline > 0.05);
- shift register: for consecutive lags with clear flags, whether the
  recurrent weight from one to the next follows the polarity rule (positive
  between same-polarity flags, negative otherwise);
- identity vs membership: swap the max's and 2nd max's values, and patch (a)
  the clear membership flags only, (b) greedily chosen neurons until the
  prediction moves to the swapped answer for 90% of pairs.

Writes results/compare/readout_generality.{md,json}.

Example:
    python scripts/readout_generality.py
    python scripts/readout_generality.py 32x10 32x10:s1 example
"""

import argparse
import json

import torch as th

from algzoo_interp import randn_inputs, run_with_cache
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.readout import additive_codes, greedy, membership_flags

DEFAULT_MODELS = ["32x10", "32x10:s1", "32x10:s2", "32x10:s3", "32x10:s4", "16x10", "example"]


def load(spec):
    # Same model specs as scripts/compare_models.py
    from algzoo_interp import load_2nd_argmax, load_example_2nd_argmax
    if spec == "example":
        return load_example_2nd_argmax(), "example 16x10"
    shape, _, seed = spec.partition(":s")
    hidden, seq_len = map(int, shape.split("x"))
    seed = int(seed) if seed else None
    label = f"{hidden}x{seq_len}" + (f" seed {seed}" if seed is not None else " (best seed)")
    return load_2nd_argmax(hidden, seq_len, seed=seed), label


def analyse(model, n, greedy_target=0.9, max_greedy=24):
    model = model.double().eval()
    T, H = model.seq_len, model.hidden_size
    W = model.linear.weight.detach()
    W_hh = model.rnn.weight_hh_l0.detach()

    def run(seed):
        x = randn_inputs(n, T, seed=seed, dtype=th.float64)
        with th.no_grad():
            c = run_with_cache(model, x)
        o = x.argsort(-1, descending=True)
        return x, c, o[:, 0], o[:, 1]

    _, c_fit, q1, q2 = run(0)
    x, c, p1, p2 = run(1)
    norm = lambda h: h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # noqa: E731
    h = c.h[:, -1]
    hn = norm(h)
    pred = (h @ W.T).argmax(-1)
    R = {"accuracy": (pred == p2).double().mean().item()}

    # additive code and matched filter
    a, b, r2 = additive_codes(norm(c_fit.h[:, -1]), q1, q2, T)
    s = a[p1] + b[p2]
    R["additive_r2"] = r2
    R["surrogate_agreement"] = ((s @ W.T).argmax(-1) == pred).double().mean().item()
    A = ((a - a.mean(0)) @ W.T).T
    B = ((b + a.mean(0)) @ W.T).T
    off = B - th.diag(th.full((T,), float("inf"), dtype=th.float64))
    margins = B.diagonal() - off.amax(0)
    R["B_min_margin_0_to_8"] = margins[:-1].min().item()
    R["A_mean_diag_0_to_8"] = A.diagonal()[:-1].mean().item()
    R["A_diag_last"] = A[-1, -1].item()

    # value-dependent residual
    r = hn - s
    Ls, Lr = s @ W.T, r @ W.T
    marg = Ls.gather(1, p2[:, None]) - Ls
    shift = Lr.gather(1, p2[:, None]) - Lr
    others = th.ones_like(Ls, dtype=th.bool)
    others.scatter_(1, p2[:, None], False)
    ru = Ls.masked_fill(~others, -float("inf")).argmax(-1)
    R["residual_effect_over_margin_median"] = (
        shift.gather(1, ru[:, None])[:, 0].abs() / marg.gather(1, ru[:, None])[:, 0]).median().item()
    U, sv, _ = th.linalg.svd((W - W.mean(0)).T, full_matrices=False)
    U = U[:, sv > 1e-8 * sv[0]]
    rc = r - r.mean(0)
    R["residual_share_readout_visible"] = (((rc @ U) ** 2).sum() / (rc ** 2).sum()).item()
    R["random_share_readout_visible"] = U.shape[1] / H

    # membership flags and shift-register polarity rule
    flags = membership_flags(c)
    clear = {k: f for k, f in flags.items() if f["accuracy"] - f["majority_baseline"] > 0.05}
    R["membership_flags"] = flags
    R["n_lags_with_clear_flag"] = len(clear)
    links = []
    for k in range(1, T - 1):
        if k in clear and k + 1 in clear and clear[k]["neuron"] != clear[k + 1]["neuron"]:
            a_n, b_n = clear[k]["neuron"], clear[k + 1]["neuron"]
            same = clear[k]["polarity"] == clear[k + 1]["polarity"]
            wgt = W_hh[b_n, a_n].item()
            # also: is this the largest-magnitude incoming weight from the other flags/neurons?
            rank = int((W_hh[b_n].abs() > abs(wgt)).sum()) + 1
            links.append({"lags": [k, k + 1], "from": a_n, "to": b_n, "weight": wgt,
                          "follows_polarity_rule": (wgt > 0) == same, "rank_among_incoming": rank})
    R["shift_register_links"] = links

    # identity vs membership: swap max and 2nd
    idx = th.arange(len(x))
    xw = x.clone()
    xw[idx, p1], xw[idx, p2] = x[idx, p2], x[idx, p1]
    with th.no_grad():
        hw = run_with_cache(model, xw).h[:, -1]
    ok = (pred == p2) & ((hw @ W.T).argmax(-1) == p1)
    member_neurons = sorted({f["neuron"] for f in clear.values()})
    hp = h.clone()
    hp[:, member_neurons] = hw[:, member_neurons]
    R["swap_membership_neurons"] = member_neurons
    R["swap_moves_membership_only"] = (((hp @ W.T).argmax(-1) == p1)[ok]).double().mean().item()
    curve = greedy(h, hw, W, p1, ok, [], steps=min(max_greedy, H))
    reached = next((i + 1 for i, step in enumerate(curve) if step["moved"] >= greedy_target), None)
    R["swap_greedy_curve"] = curve
    R["swap_neurons_for_90pct"] = reached
    R["swap_greedy_overlap_with_membership"] = sorted(
        set(step["add"] for step in curve[: reached or max_greedy]) & set(member_neurons))
    return R


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--n", type=int, default=2**17)
    args = parser.parse_args()
    out = RESULTS_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.models:
        model, label = load(spec)
        print(f"Analysing {label}")
        results[label] = analyse(model, args.n)

    lines = [
        "# Readout story across models",
        "",
        f"Generated by `python scripts/readout_generality.py {' '.join(args.models) if args.models != DEFAULT_MODELS else ''}`"
        f" on {args.n:,} float64 randn inputs per model. No neuron numbers from seed 0 are used; flags and greedy "
        "sets are found per model. Margins and diagonals are in unit-normalised final-state units.",
        "",
        "| model | accuracy | additive R² | surrogate agrees | min W·b margin (pos 0-8) | mean W·a diag (pos 0-8) "
        "| residual effect / margin (median) | residual in readout view (random) | lags with clear flag | shift-register links following sign rule "
        "| swap: membership flags move | swap: neurons for 90% |",
        "|---" * 12 + "|",
    ]
    for label, r in results.items():
        links = r["shift_register_links"]
        lines.append(
            f"| {label} | {r['accuracy']:.3f} | {r['additive_r2']:.3f} | {r['surrogate_agreement']:.4f} "
            f"| {r['B_min_margin_0_to_8']:.2f} | {r['A_mean_diag_0_to_8']:+.2f} "
            f"| {r['residual_effect_over_margin_median']:.2f} "
            f"| {r['residual_share_readout_visible']:.2f} ({r['random_share_readout_visible']:.2f}) "
            f"| {r['n_lags_with_clear_flag']} of {len(r['membership_flags'])} "
            f"| {sum(l['follows_polarity_rule'] for l in links)} of {len(links)} "
            f"| {r['swap_moves_membership_only']:.3f} | {r['swap_neurons_for_90pct'] or 'not reached'} |"
        )
    (out / "readout_generality.md").write_text("\n".join(lines) + "\n")
    with open(out / "readout_generality.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
