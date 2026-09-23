#!/usr/bin/env bash
# Uma pasta por combinação de busca, lags, folds e seed.
set -euo pipefail
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_directory}/.."
python_bin=${PYTHON_BIN:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/immuno-matplotlib}

# --help não prepara nem treina.
if [[ ${1:-} == --help ]]; then
    exec "$python_bin" -m runners.run_ml_experiment --help
fi
output_root=${OUTPUT_DIR:-studies/ml}
study_stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$output_root"
study_dir=$(mktemp -d "${output_root}/study_${study_stamp}_XXXXXX")
if [[ ${PREPARE_DATA:-1} == 1 ]]; then
    "$python_bin" -u data_preparation.py 2>&1 | tee "$study_dir/data_preparation.log"
fi
read -r -a searches <<< "${SEARCHES:-random}"
read -r -a lags <<< "${LAGS:-3}"
read -r -a folds <<< "${FOLDS:-5}"
read -r -a seeds <<< "${SEEDS:-42}"
for search in "${searches[@]}"; do
    for lag in "${lags[@]}"; do
        for fold in "${folds[@]}"; do
            for seed in "${seeds[@]}"; do
                "$python_bin" -u -m runners.run_ml_experiment \
                    --search "$search" --lags "$lag" --folds "$fold" --seed "$seed" \
                    --n-iter "${N_ITER:-30}" --n-jobs "${N_JOBS:--1}" \
                    --output-dir "$study_dir" "$@"
            done
        done
    done
done
printf 'Estudo salvo em: %s\n' "$study_dir"
