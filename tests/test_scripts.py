"""Smoke tests: every script runs end to end on a small input and writes its outputs.

Opt-in (they take about a minute): pytest -m scripts
Outputs go to a temporary directory via ALGZOO_RESULTS_DIR, so committed
results are never touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# (script and args, files expected under the results directory)
CASES = {
    "zoo_baseline": (["zoo_baseline.py", "--n", "256"],
                     ["zoo_baseline/accuracy.csv", "zoo_baseline/accuracy.png"]),
    "inspect_model": (["inspect_model.py", "--hidden", "4", "--seq-len", "3", "--n", "4096", "--n-ablation", "1024"],
                      ["2nd_argmax_4x3/inspect/summary.json", "2nd_argmax_4x3/inspect/weights.png"]),
    "plot_regions_2d": (["plot_regions_2d.py", "--hidden", "2", "--resolution", "60"],
                        ["geometry/regions_2x2.png"]),
    "neuron_hypotheses": (["neuron_hypotheses.py", "--hidden", "8", "--seq-len", "5", "--n", "4096"],
                          ["2nd_argmax_8x5/hypotheses/summary.json"]),
    "neuron_dictionary": (["neuron_dictionary.py", "--handcrafted", "10", "--n", "4096", "--n-ablation", "1024"],
                          ["handcrafted_10/dictionary/dictionary.md", "handcrafted_10/dictionary/r2.png"]),
    "state_analysis": (["state_analysis.py", "--hidden", "8", "--seq-len", "5", "--n", "8192"],
                       ["2nd_argmax_8x5/state/summary.json", "2nd_argmax_8x5/state/position_codes.png"]),
    "lag_codes": (["lag_codes.py", "--hidden", "8", "--seq-len", "5", "--n", "8192", "--min-count", "20"],
                  ["2nd_argmax_8x5/lag_codes/summary.json", "2nd_argmax_8x5/lag_codes/coefs_lag_max.png"]),
    "compare_models": (["compare_models.py", "4x3", "8x5", "--n", "4096"],
                       ["compare/compare.md", "compare/compare.json"]),
    "exact_accuracy": (["exact_accuracy.py", "--hidden", "4", "--n", "4096"],
                       ["compare/exact_accuracy.md"]),
    "explain_small": (["explain_small.py", "--hidden", "4", "--seq-len", "3", "--n", "8192", "--ratio-bins", "20"],
                      ["2nd_argmax_4x3/explain/summary.json", "2nd_argmax_4x3/explain/cones.md"]),
    "explain_32x10": (["explain_32x10.py", "--hidden", "32", "--seq-len", "10", "--n", "8192"],
                      ["2nd_argmax_32x10/explain/summary.json", "2nd_argmax_32x10/explain/errors.png"]),
    "clock_test": (["clock_test.py", "--hidden", "32", "--seq-len", "10", "--n", "8192"],
                   ["2nd_argmax_32x10/clock/summary.json"]),
    "wiring": (["wiring.py", "--hidden", "32", "--seq-len", "10", "--n", "4096"],
               ["2nd_argmax_32x10/wiring/summary.json"]),
    "shift_test": (["shift_test.py", "--hidden", "32", "--seq-len", "10", "--n", "8192", "--shifts", "-1", "0", "1"],
                   ["2nd_argmax_32x10/shift/summary.json", "2nd_argmax_32x10/shift/shift.png"]),
    "readout_generality": (["readout_generality.py", "32x10", "example", "--n", "4096"],
                           ["compare/readout_generality.md", "compare/readout_generality.json"]),
    "readout_analysis": (["readout_analysis.py", "--hidden", "32", "--seq-len", "10", "--n", "8192"],
                         ["2nd_argmax_32x10/readout/summary.json", "2nd_argmax_32x10/readout/codes.png"]),
}


def run_script(args, results_dir):
    env = {**os.environ, "ALGZOO_RESULTS_DIR": str(results_dir), "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / args[0]), *args[1:]], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{args[0]} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"


@pytest.mark.scripts
@pytest.mark.parametrize("name", sorted(CASES))
def test_script_runs(name, tmp_path):
    args, expected = CASES[name]
    run_script(args, tmp_path)
    for rel in expected:
        assert (tmp_path / rel).exists(), f"{name} did not write {rel}"


@pytest.mark.scripts
def test_causal_tests_then_summary(tmp_path):
    # summarize_causal reads causal_tests' output, so run them in sequence
    run_script(["causal_tests.py", "--hidden", "32", "--seq-len", "10", "--n", "4096"], tmp_path)
    assert (tmp_path / "2nd_argmax_32x10/causal/summary.json").exists()
    run_script(["summarize_causal.py", "2nd_argmax_32x10"], tmp_path)
    assert (tmp_path / "compare/causal.md").exists()


def test_every_script_has_a_smoke_test():
    covered = {args[0] for args, _ in CASES.values()} | {"causal_tests.py", "summarize_causal.py"}
    scripts = {p.name for p in (REPO / "scripts").glob("*.py")}
    assert scripts <= covered, f"scripts without a smoke test: {sorted(scripts - covered)}"
