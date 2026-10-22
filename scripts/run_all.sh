#!/usr/bin/env bash
# Regenerate every committed result with the commands the docs cite.
# Takes a few minutes on a CPU. Afterwards, `git status results/` shows any
# file whose content changed; `git diff results/` shows how.
#
#   bash scripts/run_all.sh
#
# Set ALGZOO_RESULTS_DIR to write somewhere other than results/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
run() { echo "+ $*"; "$PY" "$@" > /dev/null; }

run scripts/zoo_baseline.py
run scripts/plot_regions_2d.py --hidden 2
run scripts/plot_regions_2d.py --hidden 3

for model in "--hidden 32 --seq-len 10" "--hidden 4 --seq-len 3" "--example" "--handcrafted 10"; do
    run scripts/inspect_model.py $model
done
for model in "--hidden 32 --seq-len 10" "--example" "--handcrafted 10"; do
    run scripts/neuron_dictionary.py $model
    run scripts/state_analysis.py $model
done
run scripts/neuron_hypotheses.py --hidden 32 --seq-len 10
run scripts/lag_codes.py --hidden 32 --seq-len 10
run scripts/lag_codes.py --hidden 4 --seq-len 3

# Causal tests: seed 0 uses the defaults; other seeds use their own neuron
# numbers (new-max detector, top-2 detector, below-2nd neuron, running max),
# found by scripts/compare_models.py
run scripts/causal_tests.py --hidden 32 --seq-len 10
run scripts/causal_tests.py --hidden 32 --seq-len 10 --seed 1 --max-neuron 22 --top2-neuron 31 --veto 22:9 6:9 --memory 12 --value-neurons 12
run scripts/causal_tests.py --hidden 32 --seq-len 10 --seed 2 --max-neuron 8 --top2-neuron 13 --veto 8:9 17:9 --memory 19 --value-neurons 19
run scripts/causal_tests.py --hidden 32 --seq-len 10 --seed 3 --max-neuron 8 --top2-neuron 17 --veto 8:9 0:9 --memory 7 --value-neurons 7
run scripts/causal_tests.py --hidden 32 --seq-len 10 --seed 4 --max-neuron 9 --top2-neuron 0 --veto 9:9 29:9 --memory 3 --value-neurons 3
run scripts/summarize_causal.py
run scripts/compare_models.py

run scripts/exact_accuracy.py
run scripts/explain_small.py --hidden 4 --seq-len 3
run scripts/explain_32x10.py --hidden 32 --seq-len 10
run scripts/readout_analysis.py --hidden 32 --seq-len 10
run scripts/clock_test.py --hidden 32 --seq-len 10
run scripts/wiring.py --hidden 32 --seq-len 10
run scripts/shift_test.py --hidden 32 --seq-len 10
run scripts/readout_generality.py
run scripts/estimation_harness.py --hidden 4 --seq-len 3 --n-truth 4194304
run scripts/estimation_harness.py --hidden 32 --seq-len 10
run scripts/tie_offsets.py --hidden 32 --seq-len 10
echo "done"
