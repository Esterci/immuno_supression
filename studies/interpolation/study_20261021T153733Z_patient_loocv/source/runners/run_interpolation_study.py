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

from data_preparation import merge_db, normalizar_colunas, translating_flags
from methods.interpolation_validation import (
    DEFAULT_SERIES,
    METHODS,
    SERIES_VALIDITY_RULES,
    calculate_metrics,
    run_patient_loocv,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOOCV intrapaciente de metodos de interpolacao temporal"
    )
    parser.add_argument(
        "--series", nargs="+", default=list(DEFAULT_SERIES), help="series continuas"
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
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pre_interpolation_data() -> pd.DataFrame:
    data_dir = ROOT / "data"
    frames = []
    for name in ("lr_input", "mpa", "tacrolimus"):
        frame = normalizar_colunas(pd.read_parquet(data_dir / f"{name}.parquet"))
        frames.append(frame.sort_values(["time", "id"]).reset_index(drop=True))
    return translating_flags(merge_db(*frames)).sort_values(["id", "time"])


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


def save_metric_plot(macro: pd.DataFrame, target: Path) -> None:
    series_names = macro["series"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(series_names), 1, figsize=(11, max(4, 3.2 * len(series_names))), squeeze=False
    )
    for axis, series in zip(axes[:, 0], series_names):
        subset = macro.loc[macro["series"].eq(series)].sort_values("rmse")
        axis.barh(subset["method"], subset["rmse"], color="#4472C4")
        axis.invert_yaxis()
        axis.set_title(series)
        axis.set_xlabel("RMSE macro (peso igual por paciente)")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    target: Path, eligibility: pd.DataFrame, macro: pd.DataFrame, best: pd.DataFrame
) -> None:
    lines = [
        "# Estudo de interpolacao temporal por paciente",
        "",
        "LOOCV intrapaciente restrito a exames internos (com observacoes antes e depois).",
        "As series usam MinMax ajustado apenas nos exames de treino do paciente.",
        "Folds com treino constante seguem o MinMaxScaler (amplitude efetiva igual a 1).",
        "Erros usam `predito - observado`. MAE, RMSE e vies principais dao o mesmo peso",
        "a cada paciente; as metricas `*_micro` sao fornecidas apenas para auditoria.",
        "Em `m155`, zero e marcador estrutural sem medida e nao entra como exame observado.",
        "",
        "## Melhores metodos por serie",
        "",
        "| Serie | Metodo | Pacientes | Exames | MAE | RMSE | Vies |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.series} | {row.method} | {row.n_patients} | {row.n_exams} | "
            f"{row.mae:.6g} | {row.rmse:.6g} | {row.bias:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Elegibilidade",
            "",
            "| Serie | Observados | Pacientes observados | Pacientes elegiveis | Exames LOOCV |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in eligibility.itertuples(index=False):
        lines.append(
            f"| {row.series} | {row.observed_exams} | {row.observed_patients} | "
            f"{row.eligible_patients} | {row.eligible_exams_per_method} |"
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
    data = load_pre_interpolation_data()
    predictions, eligibility = run_patient_loocv(
        data,
        series_names=args.series,
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
    save_metric_plot(macro, study_dir / "rmse_comparison.png")
    write_report(study_dir / "README.md", eligibility, macro, best)

    source_dir = study_dir / "source"
    source_dir.mkdir()
    for relative in (
        "data_preparation.py",
        "methods/interpolation_validation.py",
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
        "series": args.series,
        "series_validity_rules": SERIES_VALIDITY_RULES,
        "excluded_by_semantics": {
            "binary_indicators": ["ifc", "oifc", "cmv", "bkv", "dm"],
            "treatments_or_timing": ["amt_tac", "amt_map", "tad"],
            "static": ["age", "sex", "hgt", "dty", "dage", "cisq", "dyl"],
        },
        "methods": list(METHODS),
        "inputs": inputs,
        "outputs": {
            "exam_predictions": "exam_predictions.parquet",
            "patient_metrics": "patient_metrics.csv",
            "macro_metrics": "macro_metrics.csv",
            "best_methods": "best_methods.csv",
            "eligibility": "eligibility.csv",
            "plot": "rmse_comparison.png",
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

