#!/usr/bin/env bash
# Executa o estudo de interpolação com configuração por variáveis de ambiente.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
OCC=${OCC:-1}
OUTPUT_DIR=${OUTPUT_DIR:-studies/interpolation}
STUDY_NAME=${STUDY_NAME:-}
MIN_OBSERVATIONS=${MIN_OBSERVATIONS:-5}
NORMALIZATION=${NORMALIZATION:-patient-train-minmax}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/immuno-matplotlib}

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_directory}/.."

if [[ ${1:-} == --help ]]; then
    exec "$PYTHON_BIN" -m runners.run_interpolation_study --help
fi

read -r -a occasions <<< "$OCC"
if (( ${#occasions[@]} == 0 )); then
    echo "OCC deve informar ao menos uma ocasião" >&2
    exit 2
fi

command=(
    "$PYTHON_BIN" -m runners.run_interpolation_study
    --occ "${occasions[@]}"
    --output-dir "$OUTPUT_DIR"
    --min-observations "$MIN_OBSERVATIONS"
    --normalization "$NORMALIZATION"
)

if [[ -n "$STUDY_NAME" ]]; then
    command+=(--study-name "$STUDY_NAME")
fi

command+=("$@")

echo "Executando estudo de interpolação para OCC: ${occasions[*]}"
exec "${command[@]}"
