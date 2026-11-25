"""Executa estudo LOOCV de interpolacao intrapaciente."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from data_preparation import (
    EXCLUDED_VARIABLE_COLUMNS,
    NON_TEMPORAL_COLUMNS,
    merge_db,
    normalizar_colunas,
    select_occ5,
    translating_flags,
)
from methods.interpolation import (
    INTERPOLATION_EXCLUSIONS,
    METHODS,
    SERIES_VALIDITY_RULES,
    calculate_metrics,
    run_patient_loocv,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOOCV intrapaciente de metodos de interpolacao temporal"
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help=(
            "séries contínuas; por padrão são descobertas a partir da classificação "
            "atual de variáveis"
        ),
    )
    parser.add_argument(
        "--occ",
        type=int,
        nargs="+",
        default=[1],
        help="ocasiões mantidas antes do merge (padrão: 1)",
    )
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument(
        "--normalization",
        choices=["patient-train-minmax", "none"],
        default="patient-train-minmax",
        help="MinMax nos exames de treino do paciente (padrao) ou escala original",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "studies" / "interpolation"
    )
    parser.add_argument("--study-name", default=None)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pre_interpolation_data(occasions=(1,)) -> pd.DataFrame:
    data_dir = ROOT / "data"
    frames = []
    for name in ("lr_input", "mpa", "tacrolimus"):
        frame = normalizar_colunas(pd.read_parquet(data_dir / f"{name}.parquet"))
        frame = select_occ5(frame, occasions)
        frames.append(frame.sort_values(["time", "id"]).reset_index(drop=True))
    return translating_flags(merge_db(*frames, occ=occasions)).sort_values(
        ["id", "time"]
    )


def classify_series(data: pd.DataFrame) -> pd.DataFrame:
    """Classifica as colunas temporais para o protocolo de interpolação."""
    ignored = {"id", "time", *NON_TEMPORAL_COLUMNS, *EXCLUDED_VARIABLE_COLUMNS}
    rows = []
    for column in data.columns:
        if column in ignored:
            continue
        reason = INTERPOLATION_EXCLUSIONS.get(column)
        observed = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = observed.notna()
        if SERIES_VALIDITY_RULES.get(column) == "strictly_positive":
            valid &= observed.gt(0)
        counts = data.loc[valid].groupby("id").size().to_numpy(dtype=float)
        rows.append(
            {
                "series": column,
                "interpolation_eligible": reason is None,
                "reason": reason or "série temporal contínua",
                "observed_exams": int(valid.sum()),
                "observed_patients": int(data.loc[valid, "id"].nunique()),
                "observed_exams_per_patient_sd": float(
                    np.std(counts, ddof=1) if len(counts) > 1 else 0.0
                ),
                "unique_values": int(observed.loc[valid].nunique()),
            }
        )
    return pd.DataFrame(rows)


def make_study_dir(base: Path, requested_name: str | None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = requested_name or f"study_{timestamp}_patient_loocv"
    target = base / stem
    suffix = 1
    while target.exists():
        target = base / f"{stem}_{suffix}"
        suffix += 1
    target.mkdir()
    return target


def save_metric_plots(macro: pd.DataFrame, target_dir: Path) -> list[str]:
    """Salva um gráfico por série, sempre com média e DP entre pacientes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metrics = (
        ("rmse", "patient_rmse_sd", "RMSE"),
        ("mae", "patient_mae_sd", "MAE"),
        ("mape_percent", "patient_mape_percent_sd", "MAPE (%)"),
    )
    for series in macro["series"].drop_duplicates():
        subset = macro.loc[macro["series"].eq(series)].sort_values("rmse")
        fig, axes = plt.subplots(1, 3, figsize=(19, max(5, 0.55 * len(subset) + 2)))
        for axis, (metric, sd_column, label) in zip(axes, metrics):
            axis.barh(
                subset["method"],
                subset[metric],
                xerr=subset[sd_column].fillna(0),
                color="#4472C4",
                alpha=0.85,
                capsize=3,
            )
            axis.invert_yaxis()
            axis.set_title(label)
            axis.set_xlabel(f"{label} macro ± DP entre pacientes")
            axis.grid(axis="x", alpha=0.25)
        fig.suptitle(f"Série: {series}", fontsize=14)
        fig.tight_layout()
        filename = f"{series}_metrics.png"
        fig.savefig(target_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(f"plots/{filename}")
    return outputs


def write_report(
    target: Path, eligibility: pd.DataFrame, macro: pd.DataFrame, best: pd.DataFrame
) -> None:
    lines = [
        "# Estudo de interpolacao temporal por paciente",
        "",
        "LOOCV intrapaciente restrito a exames internos (com observacoes antes e depois).",
        "As series usam MinMax ajustado apenas nos exames de treino do paciente.",
        "Folds com treino constante seguem o MinMaxScaler (amplitude efetiva igual a 1).",
        "Erros usam `predito - observado`. MAE, RMSE, MAPE e vies dao o mesmo peso",
        "a cada paciente; as metricas `*_micro` sao fornecidas apenas para auditoria.",
        "MAPE e calculado na escala original; observacoes reais iguais a zero sao excluidas.",
        "Em `m155`, zero e marcador estrutural sem medida e nao entra como exame observado.",
        "",
        "## Melhores metodos por serie",
        "",
        "| Serie | Metodo | Pacientes | Exames | MAE ± DP | RMSE ± DP | MAPE (%) ± DP | Vies ± DP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.series} | {row.method} | {row.n_patients} | {row.n_exams} | "
            f"{row.mae:.6g} ± {row.patient_mae_sd:.6g} | "
            f"{row.rmse:.6g} ± {row.patient_rmse_sd:.6g} | "
            f"{row.mape_percent:.6g} ± {row.patient_mape_percent_sd:.6g} | "
            f"{row.bias:.6g} ± {row.patient_bias_sd:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Elegibilidade",
            "",
            "| Serie | Observados (DP/paciente) | Pacientes observados | Pacientes elegiveis | Exames LOOCV (DP/paciente) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in eligibility.itertuples(index=False):
        lines.append(
            f"| {row.series} | {row.observed_exams} "
            f"({row.observed_exams_per_patient_sd:.3g}) | {row.observed_patients} | "
            f"{row.eligible_patients} | {row.eligible_exams_per_method} "
            f"({row.eligible_exams_per_patient_sd:.3g}) |"
        )
    lines.extend(
        [
            "",
            "A escolha automatica usa o menor RMSE macro, com MAE como desempate.",
            "Indicadores binarios, doses prescritas e variaveis estaticas nao fazem parte",
            "deste estudo, pois estes interpoladores continuos nao preservam sua semantica.",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    study_dir = make_study_dir(args.output_dir, args.study_name)
    data = load_pre_interpolation_data(args.occ)
    series_catalog = classify_series(data)
    discovered_series = series_catalog.loc[
        series_catalog["interpolation_eligible"], "series"
    ].tolist()
    series = args.series if args.series is not None else discovered_series
    predictions, eligibility = run_patient_loocv(
        data,
        series_names=series,
        min_observations=args.min_observations,
        normalization=args.normalization,
    )
    patient, macro = calculate_metrics(predictions)
    best = (
        macro.sort_values(["series", "rmse", "mae", "method"])
        .groupby("series", as_index=False)
        .first()
    )

    predictions.to_parquet(study_dir / "exam_predictions.parquet", index=False)
    patient.to_csv(study_dir / "patient_metrics.csv", index=False)
    macro.to_csv(study_dir / "macro_metrics.csv", index=False)
    best.to_csv(study_dir / "best_methods.csv", index=False)
    eligibility.to_csv(study_dir / "eligibility.csv", index=False)
    series_catalog.to_csv(study_dir / "series_catalog.csv", index=False)
    plot_paths = save_metric_plots(macro, study_dir / "plots")
    write_report(study_dir / "README.md", eligibility, macro, best)

    source_dir = study_dir / "source"
    source_dir.mkdir()
    for relative in (
        "data_preparation.py",
        "methods/interpolation.py",
        "runners/run_interpolation_study.py",
    ):
        source = ROOT / relative
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    inputs = {}
    for name in ("lr_input", "mpa", "tacrolimus"):
        path = ROOT / "data" / f"{name}.parquet"
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    metadata = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "validation": "leave-one-observed-internal-exam-out within patient",
            "endpoint_policy": "excluded; interpolation only, no extrapolation",
            "same_folds_for_all_methods": True,
            "aggregation": {
                "patient_weight": "equal",
                "mae": "mean of patient MAE",
                "rmse": "sqrt(mean of patient MSE)",
                "bias": "mean of patient mean(predicted - observed)",
                "mape": "mean of patient MAPE on original scale; observed zero excluded",
                "standard_deviation": "sample SD across patient-level metrics",
            },
            "selection": "minimum macro RMSE, then macro MAE",
            "min_observations": args.min_observations,
            "normalization": {
                "mode": args.normalization,
                "scope": "one scaler per patient, series and LOOCV fold",
                "fit": "training exams from that patient only",
                "metrics_scale": (
                    "normalized" if args.normalization != "none" else "original"
                ),
                "original_values_preserved": True,
                "constant_training_range": "effective range 1, matching MinMaxScaler",
            },
        },
        "series": series,
        "series_selection": (
            "explicit --series" if args.series is not None else "automatic taxonomy"
        ),
        "occasions": args.occ,
        "series_validity_rules": SERIES_VALIDITY_RULES,
        "excluded_by_semantics": INTERPOLATION_EXCLUSIONS,
        "methods": list(METHODS),
        "inputs": inputs,
        "outputs": {
            "exam_predictions": "exam_predictions.parquet",
            "patient_metrics": "patient_metrics.csv",
            "macro_metrics": "macro_metrics.csv",
            "best_methods": "best_methods.csv",
            "eligibility": "eligibility.csv",
            "series_catalog": "series_catalog.csv",
            "plots": plot_paths,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (study_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(study_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
