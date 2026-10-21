"""Complete explanation checks for a seq_len-3 2nd argmax model (roadmap step 5).

1. Exact accuracy without sampling: enumerate every activation-pattern cone
   exactly (algzoo_interp/cones.py) and compare with a sampled estimate.
2. Sign symmetry: the model's weights are (nearly) invariant under swapping
   neurons 0<->3 and 1<->2 with x -> -x, so f(-x) = f(x). For 3 inputs the
   2nd largest is the median, which has the same symmetry.
3. Ratio readout: the logits depend only on u = h0 + h3 and v = h1 + h2 at the
   last step, so the prediction is a function of v / u with two thresholds.
4. Error decomposition by cone type: "clean" cones (one full neuron pair on
   at every step) vs "mixed" or all-off cones.
5. Rebuilds from the explanation, compared with the model:
   - ratio rule on the model's own u, v;
   - u = |a d - e|, v = |b d + e| with d = x2 - x1, e = x1 - x0: model-like
     (a, b) = (4, 4.7), and ideal (a, b) that put the ratio thresholds exactly
     on the true median boundaries;
   - clean-state model: the model's own linear maps, but with the neuron pairs
     forced to switch together (no mixed states).
6. Resample check: swap final hidden states between inputs with nearly the
   same v / u. If only the ratio matters, predictions are unchanged.

Assumes the neuron-pair layout of the 4x3 zoo model (pairs (0, 1) and (2, 3),
symmetry 0<->3, 1<->2); other models need their own pairing.

Writes results/<model>/explain/{summary.json, cones.md, ratio.png}.

Example:
    python scripts/explain_small.py --hidden 4 --seq-len 3
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import get_weights, randn_inputs, run_with_cache
from algzoo_interp.cones import enumerate_cones, exact_accuracy
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.plots import savefig

SYM = [3, 2, 1, 0]
PAIR_POS, PAIR_NEG = [1, 1, 0, 0], [0, 0, 1, 1]


def cone_kind(pattern):
    """'+' / '-' when exactly one full pair is on at a step, '0' when all off, else 'm' + bits."""
    out = []
    for row in pattern.int().tolist():
        out.append("+" if row == PAIR_POS else "-" if row == PAIR_NEG else "0" if not any(row) else "m" + "".join(map(str, row)))
    return " ".join(out)


def ratio_rule(u, v, r1, r2):
    """Predicted class from v / u: 1 above r1, 2 below r2, else 0 (u = v = 0 gives 0, u = 0 < v gives 1)."""
    r = v / u.clamp(min=1e-300)
    out = th.where(r > r1, 1, th.where(r < r2, 2, 0))
    out[(u == 0) & (v == 0)] = 0
    return out


def clean_state_logits(x, W_ih, W_hh, W_out):
    """Logits if each step's neuron pair switched together, chosen by x0 > 0, x1 > x0, x2 > x1."""
    pos = th.tensor(PAIR_POS, dtype=x.dtype)
    neg = th.tensor(PAIR_NEG, dtype=x.dtype)
    bits = th.stack([x[:, 0] > 0, x[:, 1] > x[:, 0], x[:, 2] > x[:, 1]], 1)
    logits = th.zeros(len(x), W_out.shape[0], dtype=x.dtype)
    for code in range(8):
        key = [bool(code & 4), bool(code & 2), bool(code & 1)]
        A = th.zeros(W_ih.shape[0], 3, dtype=x.dtype)
        for t, on in enumerate(key):
            e = th.zeros(3, dtype=x.dtype)
            e[t] = 1
            A = (pos if on else neg)[:, None] * (W_ih[:, None] * e + W_hh @ A)
        sel = (bits == th.tensor(key)).all(1)
        logits[sel] = x[sel] @ (W_out @ A).T
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**20)
    parser.add_argument("--ratio-bins", type=int, default=2000, help="bins of log(v/u) for the resample check")
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    assert model.seq_len == 3 and model.hidden_size == 4, "this script assumes the 4x3 neuron-pair layout"
    out = RESULTS_DIR / name / "explain"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Explaining {name} -> {out}")
    w = get_weights(model)
    W_ih, W_hh, W_out = (m.detach() for m in (w.W_ih, w.W_hh, w.W_out))
    x = randn_inputs(args.n, 3, dtype=th.float64)
    true = task_2nd_argmax(x)
    with th.no_grad():
        cache = run_with_cache(model, x)
        cache_neg = run_with_cache(model, -x)
    pred = cache.logits.argmax(-1)
    sampled = (pred == true).double().mean().item()
    S = {"model": name, "n_inputs": args.n}

    # 1. exact accuracy
    cones = enumerate_cones(model)
    S["exact"] = {"n_cones": len(cones), "accuracy": exact_accuracy(cones), "sampled_accuracy": sampled,
                  "sampled_stderr": (sampled * (1 - sampled) / args.n) ** 0.5}

    # 2. symmetry
    S["symmetry"] = {
        "max_abs_deviation": {"W_ih": (W_ih[SYM] + W_ih).abs().max().item(),
                              "W_hh": (W_hh[SYM][:, SYM] - W_hh).abs().max().item(),
                              "W_out": (W_out[:, SYM] - W_out).abs().max().item()},
        "max_abs_weight": {"W_hh": W_hh.abs().max().item(), "W_out": W_out.abs().max().item()},
        "prediction_same_for_minus_x": (cache_neg.logits.argmax(-1) == pred).double().mean().item(),
        "pattern_maps_under_swap": (cache.mask[:, :, SYM] == cache_neg.mask).all(-1).all(-1).double().mean().item(),
    }

    # Which comparison each step's sign bit (neuron 0 on) makes, and how often a full pair is on
    comparisons = [x[:, 0] > 0, x[:, 1] > x[:, 0], x[:, 2] > x[:, 1]]
    full_pair = [((cache.mask[:, t, 0] == cache.mask[:, t, 1]) & (cache.mask[:, t, 2] == cache.mask[:, t, 3])
                  & (cache.mask[:, t, 0] != cache.mask[:, t, 3])).double().mean().item() for t in range(3)]
    # Step-1 switching points for x0 > 0: h_0 = x0 * relu(W_ih), so pre_1[i] = W_ih[i] x1 + c_i x0,
    # and neuron i is on when x1 is above (W_ih > 0) or below (W_ih < 0) lambda_i * x0
    c = W_hh @ W_ih.clamp(min=0)
    step1 = [{"neuron": i, "on_when": f"x1 {'>' if W_ih[i] > 0 else '<'} {(-c[i] / W_ih[i]).item():.3f} * x0"}
             for i in range(4)]
    S["sign_bits"] = {
        "step1_switching_points_for_x0_positive": step1,
        "comparison": ["x0 > 0", "x1 > x0", "x2 > x1"],
        "agreement_with_neuron0_on": [(cache.mask[:, t, 0] == comparisons[t]).double().mean().item() for t in range(3)],
        "frac_exactly_one_full_pair_on": full_pair,
    }

    # 3. ratio readout
    a = (W_out[:, 0] + W_out[:, 3]) / 2
    b = (W_out[:, 1] + W_out[:, 2]) / 2
    r1 = (-(a[1] - a[0]) / (b[1] - b[0])).item()
    r2 = (-(a[2] - a[0]) / (b[2] - b[0])).item()
    h = cache.h[:, -1]
    u, v = h[:, 0] + h[:, 3], h[:, 1] + h[:, 2]
    rule = ratio_rule(u, v, r1, r2)
    log_ratio = (v / u.clamp(min=1e-300)).clamp(min=1e-6, max=1e6).log10()
    S["ratio_readout"] = {
        "a_on_u": a.tolist(), "b_on_v": b.tolist(), "predict_1_if_ratio_above": r1, "predict_2_if_ratio_below": r2,
        "agreement_with_model": (rule == pred).double().mean().item(),
        "log10_ratio_quantiles_by_true_class": {
            int(k): th.quantile(log_ratio[(true == k) & (u > 0)], th.tensor([0.01, 0.5, 0.99], dtype=th.float64)).tolist()
            for k in range(3)},
    }
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(-1, 1.5, 120)
    for k, color in zip(range(3), ["#0b3a82", "#e69f00", "#009e73"]):
        ax.hist(log_ratio[(true == k) & (u > 0)].numpy(), bins=bins, histtype="step", lw=1.8, color=color,
                label=f"true position {k}", density=True)
    top = ax.get_ylim()[1]
    for thr, lab, ha in [(r1, " predict 1 →", "left"), (r2, "← predict 2 ", "right")]:
        ax.axvline(np.log10(thr), color="#666666", ls="--", lw=1)
        ax.text(np.log10(thr), top * 0.6, lab, fontsize=8, va="center", ha=ha)
    ax.set_xlabel("log10(v / u) at the last step")
    ax.set_ylabel("density")
    ax.set_title("The prediction is a threshold on v/u (v = h1 + h2, u = h0 + h3)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    savefig(fig, out / "ratio.png")

    # u and v as exact linear maps of x in the most probable clean cones
    uv_maps = []
    for c in [c for c in cones if all(k in "+-" for k in cone_kind(c.pattern).split())][:6]:
        A = th.zeros(4, 3, dtype=th.float64)
        for t in range(3):
            e_t = th.zeros(3, dtype=th.float64)
            e_t[t] = 1
            A = c.pattern[t].double()[:, None] * (W_ih[:, None] * e_t + W_hh @ A)
        uv_maps.append({"kind": cone_kind(c.pattern), "prob": c.prob,
                        "u": [round(v, 3) for v in (A[0] + A[3]).tolist()],
                        "v": [round(v, 3) for v in (A[1] + A[2]).tolist()]})
    S["ratio_readout"]["uv_maps_main_clean_cones"] = uv_maps

    # 4. error decomposition by cone type
    total_err = sum(p for c in cones for (pp, tt), p in c.confusion.items() if pp != tt)
    clean = [c for c in cones if all(k in "+-" for k in cone_kind(c.pattern).split())]
    clean_prob = sum(c.prob for c in clean)
    clean_err = sum(p for c in clean for (pp, tt), p in c.confusion.items() if pp != tt)
    all_off_last = sum(c.prob for c in cones if not c.pattern[-1].any())
    S["errors"] = {"total": total_err, "clean_cone_prob": clean_prob, "clean_cone_error": clean_err,
                   "all_off_at_last_step_prob": all_off_last,
                   "other_cone_prob": 1 - clean_prob, "other_cone_error": total_err - clean_err}
    lines = [f"# Exact cones: {name}", "",
             f"Generated by `python scripts/explain_small.py {'--hidden 4 --seq-len 3' if not args.example else ''}`. "
             "Kind per step: `+` = neurons 0, 1 on; `-` = neurons 2, 3 on; `0` = all off; `m....` = mixed. "
             "Probabilities are exact (cone area / 4π).", "",
             "| pattern | kind | probability | accuracy | error | wrong (pred, true): prob |", "|---|---|---|---|---|---|"]
    for c in cones:
        err = sum(p for (pp, tt), p in c.confusion.items() if pp != tt)
        wrong = ", ".join(f"({pp},{tt}): {p:.5f}" for (pp, tt), p in sorted(c.confusion.items()) if pp != tt and p > 1e-6)
        lines.append(f"| {c.pattern.int().tolist()} | `{cone_kind(c.pattern)}` | {c.prob:.5f} | {c.accuracy:.4f} | {err:.5f} | {wrong} |")
    (out / "cones.md").write_text("\n".join(lines) + "\n")

    # 5. rebuilds
    x0, x1, x2 = x.unbind(-1)
    d, e = x2 - x1, x1 - x0
    ideal_a = (1 + r2) / (r1 - r2)
    ideal_b = r1 * ideal_a
    rebuilds = {
        "ratio rule on model u, v": rule,
        "|4d - e|, |4.7d + e|": ratio_rule((4 * d - e).abs(), (4.7 * d + e).abs(), r1, r2),
        f"ideal |{ideal_a:.2f}d - e|, |{ideal_b:.2f}d + e|": ratio_rule((ideal_a * d - e).abs(), (ideal_b * d + e).abs(), r1, r2),
        "clean-state (no mixed pairs)": clean_state_logits(x, W_ih, W_hh, W_out).argmax(-1),
    }
    S["rebuilds"] = {k: {"agreement_with_model": (r == pred).double().mean().item(),
                         "accuracy": (r == true).double().mean().item()} for k, r in rebuilds.items()}

    # 6. resample check: swap final states among inputs with nearly equal v / u
    valid = u > 0
    idx = th.where(valid)[0]
    order = log_ratio[idx].argsort()
    groups = th.tensor_split(idx[order], args.ratio_bins)
    gen = th.Generator().manual_seed(0)
    h_swapped = h.clone()
    for g in groups:
        h_swapped[g] = h[g[th.randperm(len(g), generator=gen)]]
    pred_swapped = (h_swapped @ W_out.T).argmax(-1)
    S["resample_check"] = {"bins": args.ratio_bins,
                           "agreement_with_model": (pred_swapped[valid] == pred[valid]).double().mean().item(),
                           "frac_inputs_with_u_zero": 1 - valid.double().mean().item()}

    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps({k: S[k] for k in ["exact", "symmetry", "sign_bits", "errors", "rebuilds", "resample_check"]}, indent=2))
    print("ratio readout agreement:", S["ratio_readout"]["agreement_with_model"], " thresholds", r1, r2)


if __name__ == "__main__":
    main()
