#!/usr/bin/env bash
# Uma pasta por combinação de busca, folds e seed.
set -euo pipefail

# Configuração por variáveis de ambiente.
PYTHON_BIN=${PYTHON_BIN:-python}
VARMODE=${VARMODE:-all}
USE_LYM=${USE_LYM:-1}
OUTPUT_DIR=${OUTPUT_DIR:-studies/ml}
PREPARE_DATA=${PREPARE_DATA:-1}
OCC=${OCC:-1}
SEARCHES=${SEARCHES:-random}
FOLDS=${FOLDS:-5}
SEEDS=${SEEDS:-42}
N_ITER=${N_ITER:-30}
N_JOBS=${N_JOBS:--1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/immuno-matplotlib}

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_directory}/.."
if [[ "$VARMODE" != all && "$VARMODE" != time-series ]]; then
    echo "VARMODE deve ser 'all' ou 'time-series'; recebido: $VARMODE" >&2
    exit 2
fi
if [[ "$USE_LYM" != 0 && "$USE_LYM" != 1 ]]; then
    echo "USE_LYM deve ser 0 ou 1; recebido: $USE_LYM" >&2
    exit 2
fi

# --help não prepara nem treina.
if [[ ${1:-} == --help ]]; then
    exec "$PYTHON_BIN" -m runners.run_ml_experiment --help
fi
study_stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUTPUT_DIR"
study_dir=$(mktemp -d "${OUTPUT_DIR}/study_${study_stamp}_XXXXXX")
if [[ "$PREPARE_DATA" == 1 ]]; then
    read -r -a occasions <<< "$OCC"
    "$PYTHON_BIN" -u data_preparation.py --occ "${occasions[@]}" \
        --var-mode "$VARMODE" \
        2>&1 | tee "$study_dir/data_preparation.log"
fi
read -r -a searches <<< "$SEARCHES"
read -r -a folds <<< "$FOLDS"
read -r -a seeds <<< "$SEEDS"
for search in "${searches[@]}"; do
    for fold in "${folds[@]}"; do
        for seed in "${seeds[@]}"; do
            "$PYTHON_BIN" -u -m runners.run_ml_experiment \
                --search "$search" --folds "$fold" --seed "$seed" \
                --normalization all \
                --var-mode "$VARMODE" \
                --n-iter "$N_ITER" --n-jobs "$N_JOBS" \
                --output-dir "$study_dir" \
                --use-lym "$USE_LYM" "$@"
        done
    done
done
printf 'Estudo salvo em: %s\n' "$study_dir"
