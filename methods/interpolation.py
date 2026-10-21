"""LOOCV intrapaciente para metodos de interpolacao temporal."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator


DEFAULT_SERIES = ("m155", "wgt", "gf", "lym", "dv_tac", "dv_map")
SERIES_VALIDITY_RULES = {
    # No conjunto original, zero e uma linha estrutural sem medida de miR155.
    "m155": "strictly_positive",
}


def _linear(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return float(np.interp(target, x, y))


def _pchip(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return float(PchipInterpolator(x, y, extrapolate=False)(target))


def _natural_cubic_spline(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return float(CubicSpline(x, y, bc_type="natural", extrapolate=False)(target))


def _temporal_regression(
    x: np.ndarray, y: np.ndarray, target: float, degree: int
) -> float:
    """Regressao polinomial em tempo padronizado, ajustada por minimos quadrados."""
    center = float(np.mean(x))
    scale = float(np.std(x))
    if scale == 0:
        raise ValueError("tempos de treinamento sem variacao")
    z = (x - center) / scale
    z_target = (target - center) / scale
    design = np.vander(z, N=degree + 1, increasing=True)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < degree + 1:
        raise ValueError("regressao temporal com matriz singular")
    target_design = np.array([z_target**power for power in range(degree + 1)])
    return float(target_design @ coefficients)


def _temporal_linear(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return _temporal_regression(x, y, target, degree=1)


def _temporal_quadratic(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return _temporal_regression(x, y, target, degree=2)


def _temporal_cubic(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return _temporal_regression(x, y, target, degree=3)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "linear": _linear,
    "pchip": _pchip,
    "cubic_spline_natural": _natural_cubic_spline,
    "temporal_regression_linear": _temporal_linear,
    "temporal_regression_quadratic": _temporal_quadratic,
    "temporal_regression_cubic": _temporal_cubic,
}


def _observed_unique_times(
    data: pd.DataFrame, series: str
) -> tuple[pd.DataFrame, int]:
    selected = data.loc[:, ["id", "time", series]].copy()
    selected["time"] = pd.to_numeric(selected["time"], errors="coerce")
    selected[series] = pd.to_numeric(selected[series], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if SERIES_VALIDITY_RULES.get(series) == "strictly_positive":
        selected = selected.loc[selected[series] > 0]
    duplicate_count = int(selected.duplicated(["id", "time"], keep=False).sum())
    if duplicate_count:
        selected = selected.groupby(["id", "time"], as_index=False)[series].mean()
    return selected.sort_values(["id", "time"]), duplicate_count


def run_patient_loocv(
    data: pd.DataFrame,
    series_names: Iterable[str] = DEFAULT_SERIES,
    methods: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = METHODS,
    min_observations: int = 5,
    normalization: str = "patient-train-minmax",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove um exame interno por vez e o prediz com os demais do paciente.

    Pontos extremos nao entram na validacao: remove-los exigiria extrapolacao, nao
    interpolacao. O mesmo conjunto de folds e usado para todos os metodos.
    """
    required = {"id", "time", *series_names}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")
    if min_observations < 5:
        raise ValueError("min_observations deve ser >= 5 para todos os metodos")
    if normalization not in ("patient-train-minmax", "none"):
        raise ValueError("normalization deve ser patient-train-minmax ou none")

    predictions: list[dict[str, float | int | str]] = []
    eligibility: list[dict[str, float | int | str]] = []

    for series in series_names:
        observed, duplicate_count = _observed_unique_times(data, series)
        observed = observed.reset_index(drop=True)
        eligible_patients = 0
        eligible_exams = 0
        for patient_id, patient in observed.groupby("id", sort=True):
            patient = patient.sort_values("time")
            n_observed = len(patient)
            n_eligible = n_observed - 2 if n_observed >= min_observations else 0
            if n_eligible <= 0:
                continue
            eligible_patients += 1
            eligible_exams += n_eligible
            times = patient["time"].to_numpy(dtype=float)
            values = patient[series].to_numpy(dtype=float)

            for held_index in range(1, n_observed - 1):
                keep = np.arange(n_observed) != held_index
                x_train, y_train = times[keep], values[keep]
                target_time = float(times[held_index])
                observed_original = float(values[held_index])
                if normalization == "patient-train-minmax":
                    scale_min = float(np.min(y_train))
                    scale_max = float(np.max(y_train))
                    data_range = scale_max - scale_min
                    # Mesmo comportamento do MinMaxScaler para atributo constante.
                    scale_range = data_range if data_range > 0 else 1.0
                    y_train_model = (y_train - scale_min) / scale_range
                    observed_value = (observed_original - scale_min) / scale_range
                else:
                    scale_min = 0.0
                    scale_max = 1.0
                    scale_range = 1.0
                    y_train_model = y_train
                    observed_value = observed_original
                for method_name, estimator in methods.items():
                    predicted = estimator(x_train, y_train_model, target_time)
                    if not np.isfinite(predicted):
                        raise ValueError(
                            f"previsao nao finita: {series=}, {patient_id=}, "
                            f"{target_time=}, {method_name=}"
                        )
                    error = predicted - observed_value
                    predictions.append(
                        {
                            "series": series,
                            "id": patient_id,
                            "time": target_time,
                            "n_observed_patient": n_observed,
                            "held_index": held_index,
                            "method": method_name,
                            "observed": observed_value,
                            "predicted": predicted,
                            "observed_original": observed_original,
                            "predicted_original": predicted * scale_range + scale_min,
                            "normalization": normalization,
                            "scale_min": scale_min,
                            "scale_max": scale_max,
                            "constant_training_range": scale_max == scale_min,
                            "error": error,
                            "absolute_error": abs(error),
                            "squared_error": error**2,
                        }
                    )

        eligibility.append(
            {
                "series": series,
                "observed_exams": len(observed),
                "observed_patients": int(observed["id"].nunique()),
                "eligible_patients": eligible_patients,
                "eligible_exams_per_method": eligible_exams,
                "duplicate_rows_collapsed": duplicate_count,
                "min_observations": min_observations,
                "normalization": normalization,
            }
        )

    return pd.DataFrame(predictions), pd.DataFrame(eligibility)


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula metricas individuais e macroagregadas com peso igual por paciente."""
    if predictions.empty:
        raise ValueError("nenhuma predicao elegivel")

    patient = (
        predictions.groupby(["series", "method", "id"], as_index=False)
        .agg(
            n_exams=("error", "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
            bias=("error", "mean"),
        )
        .sort_values(["series", "method", "id"])
    )
    patient["rmse"] = np.sqrt(patient["mse"])

    rows: list[dict[str, float | int | str]] = []
    grouped_predictions = predictions.groupby(["series", "method"], sort=True)
    for (series, method), patient_group in patient.groupby(
        ["series", "method"], sort=True
    ):
        point_group = grouped_predictions.get_group((series, method))
        rows.append(
            {
                "series": series,
                "method": method,
                "n_patients": int(len(patient_group)),
                "n_exams": int(patient_group["n_exams"].sum()),
                "mae": float(patient_group["mae"].mean()),
                "rmse": float(np.sqrt(patient_group["mse"].mean())),
                "bias": float(patient_group["bias"].mean()),
                "patient_mae_sd": float(patient_group["mae"].std(ddof=1)),
                "patient_rmse_mean": float(patient_group["rmse"].mean()),
                "patient_rmse_sd": float(patient_group["rmse"].std(ddof=1)),
                "patient_bias_sd": float(patient_group["bias"].std(ddof=1)),
                "mae_micro": float(point_group["absolute_error"].mean()),
                "rmse_micro": float(np.sqrt(point_group["squared_error"].mean())),
                "bias_micro": float(point_group["error"].mean()),
            }
        )

    macro = pd.DataFrame(rows).sort_values(["series", "rmse", "mae", "method"])
    macro["rank_rmse"] = macro.groupby("series")["rmse"].rank(
        method="min", ascending=True
    ).astype(int)
    return patient, macro

