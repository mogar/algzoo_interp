"""Compare mechanisms across 2nd argmax models (roadmap step 4: generality).

For each model, finds the neuron that best matches each key role found in
32x10 (neuron numbers differ between models) and reports how well it
matches, plus structural summaries:

- accuracy;
- for each key feature, the best single-neuron R² and which neuron;
- event detectors: best on/off agreement with "new positive max" and
  "new positive top-2";
- resets: share of all |W_hh| leaving the new-max detector neuron;
- veto: largest readout weight, and whether it is the new-max detector
  voting against the last position;
- clock: neurons whose on/off state, from lag 2 on, is (almost) always on at
  some lag of the max and (almost) always off at another;
- state R²: median share of neuron values explained by the ideal online state
  plus current input (algzoo_interp/state.py).

Writes results/compare/{compare.md, compare.json}.

Model specs: HxT (zoo best seed), HxT:sN (seed N), example (blog 16x10),
hcT (handcrafted, T in 2, 3, 10).

Example:
    python scripts/compare_models.py
    python scripts/compare_models.py 32x10 32x10:s1 example
"""

import argparse
import json

import torch as th
from alg_zoo import handcrafted_2nd_argmax
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import get_weights, load_2nd_argmax, load_example_2nd_argmax, randn_inputs, run_with_cache
from algzoo_interp.dictionary import build_dictionary, event_agreement
from algzoo_interp.features import events as all_events
from algzoo_interp.state import ideal_state, p_on_by_lag, state_r2
from algzoo_interp.paths import RESULTS_DIR

DEFAULT_MODELS = [
    "32x10", "32x10:s1", "32x10:s2", "32x10:s3", "32x10:s4",
    "16x10", "example", "hc10",
    "8x10", "8x6", "8x5", "8x4", "4x3",
]
# Key value features from the 32x10 analysis (see docs/HYPOTHESES.md §D)
KEY_FEATURES = {
    "max0-max0_prev": "new-max size (32x10: 7)",
    "max0_prev": "running max (17)",
    "2nd0_prev": "running 2nd max (31)",
    "max-2nd": "gap (1, 5)",
    "relu(2nd0_prev-x_t)": "below 2nd max (12)",
    "max0-x_t": "max − current",
}


def load(spec):
    if spec == "example":
        return load_example_2nd_argmax(), "example 16x10"
    if spec.startswith("hc"):
        return handcrafted_2nd_argmax(int(spec[2:])).eval(), f"handcrafted {spec[2:]}"
    shape, _, seed = spec.partition(":s")
    hidden, seq_len = map(int, shape.split("x"))
    seed = int(seed) if seed else None
    model = load_2nd_argmax(hidden, seq_len, seed=seed)
    label = f"{hidden}x{seq_len}" + (f" seed {seed}" if seed is not None else " (best seed)")
    return model, label


def analyse(model, n, clock_on=0.95, clock_off=0.05):
    model = model.eval()
    T, H = model.seq_len, model.hidden_size
    x = randn_inputs(n, T)
    with th.no_grad():
        cache = run_with_cache(model, x)
    acc = (cache.logits.argmax(-1) == task_2nd_argmax(x)).float().mean().item()
    _, names, r2 = build_dictionary(cache)

    features = {}
    for f in KEY_FEATURES:
        i = names.index(f)
        best = int(r2[i].argmax())
        features[f] = {"neuron": best, "r2": round(r2[i, best].item(), 3)}

    events = {}
    agree_all = event_agreement(cache, all_events(cache.x.cpu()), list(range(1, T)))
    for ev in ["new_max_pos", "new_top2_pos"]:
        best = int(agree_all[ev].argmax())
        events[ev] = {"neuron": best, "agreement": round(agree_all[ev][best].item(), 3)}

    w = get_weights(model)
    col_mass = w.W_hh.abs().sum(0)
    detector = events["new_max_pos"]["neuron"]
    reset_share = (col_mass[detector] / col_mass.sum()).item()
    top_share_neuron = int(col_mass.argmax())

    flat = w.W_out.abs().flatten().argmax()
    veto_logit, veto_neuron = int(flat // H), int(flat % H)
    veto = {"logit": veto_logit, "neuron": veto_neuron, "weight": round(w.W_out[veto_logit, veto_neuron].item(), 2),
            "is_detector_on_last": veto_neuron == detector and veto_logit == T - 1}

    clock = []
    if T >= 4:
        pooled, _ = p_on_by_lag(cache, ideal_state(cache.x.cpu())["lag_max"], min_count=200)
        late = pooled[:, 2:]
        for n_ in range(H):
            row = late[n_][~late[n_].isnan()]
            if len(row) and (row >= clock_on).any() and (row <= clock_off).any():
                clock.append(n_)

    state = state_r2(cache, list(range(1, T)), "values+input")
    return {
        "accuracy": round(acc, 4),
        "features": features,
        "events": events,
        "reset_share_of_detector": round(reset_share, 3),
        "neuron_with_largest_W_hh_share": top_share_neuron,
        "largest_W_hh_share": round((col_mass.max() / col_mass.sum()).item(), 3),
        "largest_readout_weight": veto,
        "clock_neurons": clock,
        "state_r2_median": round(state.median().item(), 3),
        "state_r2_frac_ge_0.9": round((state >= 0.9).float().mean().item(), 3),
    }


def write_markdown(results, args, path):
    lines = [
        "# Mechanisms across models",
        "",
        f"Generated by `python scripts/compare_models.py {' '.join(args.models) if args.models != DEFAULT_MODELS else ''}`"
        f" on {args.n:,} randn inputs per model. Cells show `neuron: score`; neuron numbers differ between models.",
        "",
        "## Event detectors and stored values",
        "",
        "Event columns: on/off agreement with the event (steps 1..T-1). Value columns: best single-neuron R².",
        "",
        "| model | accuracy | new max (event) | new top-2 (event) | "
        + " | ".join(KEY_FEATURES.values()) + " |",
        "|---" * (4 + len(KEY_FEATURES)) + "|",
    ]
    for label, r in results.items():
        ev = r["events"]
        cells = [f"{r['features'][f]['neuron']}: {r['features'][f]['r2']:.2f}" for f in KEY_FEATURES]
        lines.append(
            f"| {label} | {r['accuracy']:.3f} | {ev['new_max_pos']['neuron']}: {ev['new_max_pos']['agreement']:.3f} "
            f"| {ev['new_top2_pos']['neuron']}: {ev['new_top2_pos']['agreement']:.3f} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "## Structure",
        "",
        "*Reset share*: share of all |W_hh| leaving the new-max detector (largest share, and which neuron, in brackets). "
        "*Largest readout weight*: `neuron → logit: weight`; ✓ when it is the new-max detector voting against the last "
        "position. *Clock*: neurons always on at some lag ≥ 2 of the max and always off at another. "
        "*State R²*: median (and share ≥ 0.9) of neuron values explained by the ideal online state plus current input.",
        "",
        "| model | reset share | largest readout weight | clock neurons | state R² |",
        "|---|---|---|---|---|",
    ]
    for label, r in results.items():
        v = r["largest_readout_weight"]
        lines.append(
            f"| {label} | {r['reset_share_of_detector']:.2f} (max {r['largest_W_hh_share']:.2f}, n{r['neuron_with_largest_W_hh_share']}) "
            f"| {v['neuron']} → {v['logit']}: {v['weight']}{' ✓' if v['is_detector_on_last'] else ''} "
            f"| {len(r['clock_neurons'])} {r['clock_neurons'] if r['clock_neurons'] else ''} "
            f"| {r['state_r2_median']:.2f} ({r['state_r2_frac_ge_0.9']:.0%}) |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--n", type=int, default=2**16)
    args = parser.parse_args()
    out = RESULTS_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.models:
        model, label = load(spec)
        print(f"Analysing {label}")
        results[label] = analyse(model, args.n)
    with open(out / "compare.json", "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, args, out / "compare.md")


if __name__ == "__main__":
    main()
