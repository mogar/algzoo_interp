"""How the readout decodes the 2nd max's position (32x10, seed 0).

1. Final-step group ablations: mean-ablate one neuron group at the last step
   only (nothing upstream changes) and measure accuracy, split by whether the
   2nd max arrived after or before the max, or at the last position.
2. Additive position code: the mean (unit-normalised) final state per
   (max position i, 2nd position j) cell fits a(i) + b(j). Report R², whether
   the readout decodes j from the fit, and the logit tables A = W a, B = W b.
3. Value-dependent residual: per input, the normalised final state minus
   a(i) + b(j). How often it flips a readout comparison (vs the error rate),
   and how much of it lies in the readout-visible subspace (directions that
   change logit differences) vs a random direction.
4. Top-2 delay line: P(neuron on | the input k steps ago is the running max /
   running 2nd / neither), for the delay neurons; and, for every lag, the best
   single-neuron on/off flag for "the input k steps ago is in the top two".
5. Where the 2nd max's position lives at the readout: pairs of inputs with the
   same max position but a different 2nd position (2nd swapped with another
   input); patch one neuron group's final state from the other input and
   measure how often the prediction moves to the other input's answer.
   Broken down by position, plus a greedy search for neurons to add.
6. Max vs 2nd identity: swap the max's and 2nd max's values. The set of top-two
   positions is unchanged, only which one is the max flips. Patch groups (and
   single neurons, greedily) from the swapped input and measure how often the
   prediction moves to the old max's position.
7. Role sets: keep the membership, identity and last-position neurons at the
   final step and replace the rest (unit-normalised mean, or values from
   another input with the same max and 2nd positions). As a control,
   resample the kept neurons instead. Also tests whether the value neurons
   cancel the reference neurons' value-dependent part (within-cell
   correlation of their contributions to the margin).

Neuron groups are the 32x10 seed-0 roles from docs/HYPOTHESES.md §D.

Writes results/<model>/readout/{summary.json, codes.png}.

Example:
    python scripts/readout_analysis.py --hidden 32 --seq-len 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch as th

from algzoo_interp import randn_inputs, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.plots import heatmap, savefig

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GROUPS = {
    "new-max detector": [7],
    "top-2 detector": [16],
    "values": [1, 5, 17, 31],
    "below-2nd": [12],
    "reference": [0, 6, 13, 18, 29],
    "clock": [3, 10, 11, 27, 28],
    "membership flags": [20, 24, 25, 30],
    "delay line": [2, 15, 19, 22, 26],
    "last-max": [21, 23],
    "other": [4, 8, 9, 14],
}
DELAY = {22: 2, 19: 3, 2: 6, 26: 7, 15: 8}  # neuron: lag
# Role sets for sections 6-7 (neuron 14 is a second lag-1 membership flag)
MEMBERSHIP = [2, 15, 19, 22, 26, 20, 24, 25, 30, 14]
IDENTITY = [0, 6, 13, 18, 29, 3, 10, 11, 27, 28, 7]
LAST = [12, 16, 21, 23]
PATCH_COMBOS = [["delay line", "membership flags"],
                ["delay line", "membership flags", "reference"],
                ["delay line", "membership flags", "reference", "below-2nd"]]


def run(model, n, seed):
    x = randn_inputs(n, model.seq_len, seed=seed, dtype=th.float64)
    with th.no_grad():
        c = run_with_cache(model, x)
    order = x.argsort(-1, descending=True)
    return x, c, order[:, 0], order[:, 1]


def additive_codes(h, p1, p2, T):
    """Weighted least-squares fit of per-cell mean states: mu(i, j) ≈ a(i) + b(j)."""
    cell = p1 * T + p2
    cnt = th.bincount(cell, minlength=T * T).double()
    mu = th.zeros(T * T, h.shape[1], dtype=th.float64).index_add_(0, cell, h) / cnt.clamp(min=1)[:, None]
    I, J = th.arange(T * T) // T, th.arange(T * T) % T
    cells = th.where(I != J)[0]
    X = th.zeros(len(cells), 2 * T, dtype=th.float64)
    X[th.arange(len(cells)), I[cells]] = 1
    X[th.arange(len(cells)), T + J[cells]] = 1
    w = cnt[cells].sqrt()[:, None]
    coef = th.linalg.lstsq(X * w, mu[cells] * w).solution
    fit = X @ coef
    wmean = (cnt[cells][:, None] * mu[cells]).sum(0) / cnt[cells].sum()
    r2 = 1 - (cnt[cells][:, None] * (mu[cells] - fit) ** 2).sum() / (cnt[cells][:, None] * (mu[cells] - wmean) ** 2).sum()
    return coef[:T], coef[T:], r2.item()


def greedy(h, h_src, W, target, ok, start, steps):
    """Greedily add neurons to the patched set, maximising how often the prediction moves to `target`."""
    chosen, curve = list(start), []
    for _ in range(steps):
        best = None
        for n in range(h.shape[1]):
            if n in chosen:
                continue
            hp = h.clone()
            hp[:, chosen + [n]] = h_src[:, chosen + [n]]
            v = (((hp @ W.T).argmax(-1) == target)[ok]).double().mean().item()
            if best is None or v > best[0]:
                best = (v, n)
        chosen.append(best[1])
        curve.append({"add": best[1], "moved": best[0]})
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**19)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    T = model.seq_len
    W = model.linear.weight.detach()
    out = RESULTS_DIR / name / "readout"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Readout analysis for {name} -> {out}")
    S = {"model": name, "n_inputs": args.n, "groups": GROUPS}

    x, c, p1, p2 = run(model, args.n, seed=1)
    h = c.h[:, -1]
    pred = (h @ W.T).argmax(-1)
    cases = {"after": (p2 > p1) & (p2 < T - 1), "before": p2 < p1, "at last": p2 == T - 1}

    # 1. final-step group ablations
    def acc_by_case(hh):
        pr = (hh @ W.T).argmax(-1)
        return {k: (pr[s] == p2[s]).double().mean().item() for k, s in cases.items()}
    mu = h.mean(0)
    S["final_step_ablation"] = {"baseline": acc_by_case(h)}
    for g, ns in GROUPS.items():
        hh = h.clone()
        hh[:, ns] = mu[ns]
        S["final_step_ablation"][g] = acc_by_case(hh)

    # 2. additive code on unit-normalised states (fit on a separate sample)
    _, c_fit, q1, q2 = run(model, args.n, seed=0)
    hn_fit = c_fit.h[:, -1] / c_fit.h[:, -1].norm(dim=-1, keepdim=True).clamp(min=1e-12)
    a, b, r2 = additive_codes(hn_fit, q1, q2, T)
    hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    s = a[p1] + b[p2]
    surrogate_pred = (s @ W.T).argmax(-1)
    a_c, b_c = a - a.mean(0), b + a.mean(0)
    A, B = (a_c @ W.T).T, (b_c @ W.T).T
    off = B - th.diag(th.full((T,), float("inf"), dtype=th.float64))
    S["additive_code"] = {
        "r2_cell_means": r2,
        "surrogate_accuracy": (surrogate_pred == p2).double().mean().item(),
        "surrogate_agreement_with_model": (surrogate_pred == pred).double().mean().item(),
        "per_input_r2": (1 - ((hn - s) ** 2).sum() / ((hn - hn.mean(0)) ** 2).sum()).item(),
        "A_logit_by_max_position": A.tolist(),
        "B_logit_by_2nd_position": B.tolist(),
        "B_margin_per_position": (B.diagonal() - off.amax(0)).tolist(),
        "A_diagonal": A.diagonal().tolist(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = float(B[:-1].abs().max())
    heatmap(axes[0], B, title="B = W·b(j): logits from the 2nd-position code", xlabel="2nd-max position j",
            ylabel="logit p", diverging=True, lim=lim, annotate=True, fmt="{:.1f}", colorbar=False)
    heatmap(axes[1], A, title="A = W·a(i): logits from the max-position code", xlabel="max position i",
            diverging=True, lim=lim, annotate=True, fmt="{:.1f}")
    fig.suptitle("Unit-normalised mean final state ≈ a(max position) + b(2nd position)\n"
                 "(colour clipped at positions 0-8's range; logit 9 extends beyond)")
    savefig(fig, out / "codes.png")

    # 3. value-dependent residual
    r = hn - s
    Ls, Lr = s @ W.T, r @ W.T
    marg = Ls.gather(1, p2[:, None]) - Ls
    shift = Lr.gather(1, p2[:, None]) - Lr
    others = th.ones_like(Ls, dtype=th.bool)
    others.scatter_(1, p2[:, None], False)
    flips = ((-shift > marg) & others).any(1)
    other_ls = Ls.masked_fill(~others, -float("inf"))
    ru = other_ls.argmax(-1)
    rel = (shift.gather(1, ru[:, None])[:, 0].abs() / marg.gather(1, ru[:, None])[:, 0])
    D = W - W.mean(0)
    U, sv, _ = th.linalg.svd(D.T, full_matrices=False)
    U = U[:, sv > 1e-8 * sv[0]]
    share = lambda v: (((v @ U) ** 2).sum() / (v ** 2).sum()).item()  # noqa: E731
    S["residual"] = {
        "frac_inputs_where_residual_flips_a_comparison": flips.double().mean().item(),
        "model_error_rate": (pred != p2).double().mean().item(),
        "residual_effect_over_margin_median": rel.median().item(),
        "residual_effect_over_margin_99pct": rel.quantile(0.99).item(),
        "readout_visible_dims": U.shape[1],
        "random_direction_share": U.shape[1] / W.shape[1],
        "residual_share_in_readout_visible": share(r - r.mean(0)),
        "position_code_share_in_readout_visible": share(s - s.mean(0)),
    }

    # 4. top-2 delay line and membership flags (pooled over steps)
    delay = {}
    for n, k in DELAY.items():
        counts = {"max": [0, 0], "2nd": [0, 0], "neither": [0, 0]}
        for t in range(k + 1, T):
            o = x[:, : t + 1].argsort(-1, descending=True)
            pos = t - k
            is1, is2 = o[:, 0] == pos, o[:, 1] == pos
            on = c.mask[:, t, n]
            for key, sel in [("max", is1), ("2nd", is2), ("neither", ~is1 & ~is2)]:
                counts[key][0] += int(on[sel].sum())
                counts[key][1] += int(sel.sum())
        delay[n] = {"lag": k, **{f"p_on_if_{key}": v[0] / max(v[1], 1) for key, v in counts.items()}}
    S["delay_line"] = delay
    flags = {}
    mask = c.mask.double()
    for k in range(1, T):
        on_in, n_in = th.zeros(model.hidden_size, dtype=th.float64), 0
        on_out, n_out = th.zeros(model.hidden_size, dtype=th.float64), 0
        for t in range(k, T):
            o = x[:, : t + 1].argsort(-1, descending=True)
            inn = (o[:, 0] == t - k) | (o[:, 1] == t - k)
            on_in += mask[inn, t].sum(0)
            n_in += int(inn.sum())
            on_out += mask[~inn, t].sum(0)
            n_out += int((~inn).sum())
        p_in, p_out = on_in / n_in, on_out / max(n_out, 1)
        frac = n_in / (n_in + n_out)
        acc_on = frac * p_in + (1 - frac) * (1 - p_out)
        acc = th.maximum(acc_on, 1 - acc_on)
        best = int(acc.argmax())
        flags[k] = {"neuron": best, "polarity": "on-when-in" if acc_on[best] >= 0.5 else "off-when-in",
                    "p_on_if_in_top2": p_in[best].item(), "p_on_if_not": p_out[best].item(),
                    "accuracy": acc[best].item(), "majority_baseline": max(frac, 1 - frac)}
    S["membership_flags_by_lag"] = flags

    # 5. patch groups between inputs that differ only in the 2nd max's position
    gen = th.Generator().manual_seed(3)
    q = th.randint(0, T, (len(x),), generator=gen)
    bad = (q == p1) | (q == p2)
    while bad.any():
        q[bad] = th.randint(0, T, (int(bad.sum()),), generator=gen)
        bad = (q == p1) | (q == p2)
    idx = th.arange(len(x))
    xs = x.clone()
    xs[idx, p2], xs[idx, q] = x[idx, q], x[idx, p2]
    with th.no_grad():
        hs = run_with_cache(model, xs).h[:, -1]
    ok = (pred == p2) & ((hs @ W.T).argmax(-1) == q)
    before = p2 < p1
    def moved(ns):
        hp = h.clone()
        hp[:, ns] = hs[:, ns]
        mv = (hp @ W.T).argmax(-1) == q
        return {"all": mv[ok].double().mean().item(), "before": mv[ok & before].double().mean().item(),
                "after": mv[ok & ~before].double().mean().item()}
    S["patch_2nd_position"] = {"n_pairs": int(ok.sum()), "groups": {g: moved(ns) for g, ns in GROUPS.items()},
                               "combos": {" + ".join(cmb): moved(sum((GROUPS[g] for g in cmb), [])) for cmb in PATCH_COMBOS}}
    base = sum((GROUPS[g] for g in PATCH_COMBOS[-1]), [])
    hp = h.clone()
    hp[:, base] = hs[:, base]
    mv = (hp @ W.T).argmax(-1) == q
    S["patch_2nd_position"]["base_15_by_position"] = {
        key: [mv[ok & (pos == k)].double().mean().item() for k in range(T)]
        for key, pos in [("new_2nd_position", q), ("old_2nd_position", p2), ("max_position", p1)]}
    S["patch_2nd_position"]["greedy_from_base_15"] = greedy(h, hs, W, q, ok, base, steps=8)

    # 6. max vs 2nd identity: swap their values
    xw = x.clone()
    xw[idx, p1], xw[idx, p2] = x[idx, p2], x[idx, p1]
    with th.no_grad():
        hw = run_with_cache(model, xw).h[:, -1]
    okw = (pred == p2) & ((hw @ W.T).argmax(-1) == p1)
    def moved_w(ns):
        hp = h.clone()
        hp[:, ns] = hw[:, ns]
        mv = (hp @ W.T).argmax(-1) == p1
        return {"all": mv[okw].double().mean().item(), "before": mv[okw & before].double().mean().item(),
                "after": mv[okw & ~before].double().mean().item()}
    single = {n: moved_w([n])["all"] for n in range(model.hidden_size)}
    S["swap_max_and_2nd"] = {
        "n_pairs": int(okw.sum()),
        "groups": {g: moved_w(ns) for g, ns in GROUPS.items()},
        "top_single_neurons": sorted(single.items(), key=lambda kv: -kv[1])[:10],
        "greedy": greedy(h, hw, W, p1, okw, [], steps=12),
    }

    # 7. role sets
    hn_all = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    mu_n = hn_all.mean(0)
    cell = p1 * T + p2
    perm = th.empty(len(x), dtype=th.long)
    g_perm = th.Generator().manual_seed(0)
    for cc in cell.unique():
        ii = th.where(cell == cc)[0]
        perm[ii] = ii[th.randperm(len(ii), generator=g_perm)]
    L_n = hn_all @ W.T
    top = L_n.topk(2, -1).values
    margin_n = top[:, 0] - top[:, 1]
    role_sets = {"membership + identity + last": MEMBERSHIP + IDENTITY + LAST,
                 "membership + identity": MEMBERSHIP + IDENTITY,
                 "membership only": MEMBERSHIP, "identity only": IDENTITY}
    roles = {}
    for rname, keep in role_sets.items():
        drop = [n for n in range(model.hidden_size) if n not in keep]
        hh = hn_all.clone()
        hh[:, drop] = mu_n[drop]
        a_mean = ((hh @ W.T).argmax(-1) == pred).double().mean().item()
        hh = hn_all.clone()
        hh[:, drop] = hn_all[perm][:, drop]
        a_res = ((hh @ W.T).argmax(-1) == pred).double().mean().item()
        roles[rname] = {"n_kept": len(keep), "agree_mean_ablate_rest": a_mean, "agree_resample_rest_within_cell": a_res}
    keep = MEMBERSHIP + IDENTITY + LAST
    drop = [n for n in range(model.hidden_size) if n not in keep]
    control = {}
    for cname, ns in [("resample dropped", drop), ("resample kept (control)", keep)]:
        hh = hn_all.clone()
        hh[:, ns] = hn_all[perm][:, ns]
        changed = (hh @ W.T).argmax(-1) != pred
        control[cname] = {"neurons": ns, "frac_changed": changed.double().mean().item(),
                          "median_margin_of_changed": margin_n[changed].median().item()}
    control["median_margin_all"] = margin_n.median().item()
    def within_dev(v):
        cnt = th.bincount(cell, minlength=T * T).double()
        return v - (th.bincount(cell, weights=v, minlength=T * T) / cnt.clamp(min=1))[cell]
    contrib = lambda ns: ((W[p2][:, ns] - W[p1][:, ns]) * hn_all[:, ns]).sum(-1)  # noqa: E731
    ref, val = within_dev(contrib(GROUPS["reference"])), within_dev(contrib(GROUPS["values"]))
    S["role_sets"] = {"membership": MEMBERSHIP, "identity": IDENTITY, "last": LAST, "sufficiency": roles,
                      "resample_control": control,
                      "reference_vs_values_within_cell_corr": th.corrcoef(th.stack([ref, val]))[0, 1].item()}

    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps({k: S[k] for k in ["final_step_ablation", "residual", "delay_line", "membership_flags_by_lag"]}, indent=1))
    print("additive:", {k: v for k, v in S["additive_code"].items() if not k.startswith(("A_", "B_"))})
    print("B margins:", [round(v, 1) for v in S["additive_code"]["B_margin_per_position"]])
    print("A diagonal:", [round(v, 1) for v in S["additive_code"]["A_diagonal"]])
    print("patch:", json.dumps({k: v for k, v in S["patch_2nd_position"].items() if k != "groups"}, indent=1))
    print("swap:", json.dumps({k: v for k, v in S["swap_max_and_2nd"].items()}, indent=1))
    print("roles:", json.dumps(S["role_sets"], indent=1))


if __name__ == "__main__":
    main()
