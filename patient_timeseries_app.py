"""Dash viewer for the original longitudinal patient measurements.

Run from the repository root with::

    python patient_timeseries_app.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import json
import runpy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from data_preparation import merge_db, normalizar_colunas, translating_flags


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def call_merge_db(merge_function, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Call current or archived merge_db signatures."""
    if len(inspect.signature(merge_function).parameters) >= 4:
        return merge_function(*frames, occ=1)
    return merge_function(*frames)


def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load original, merged and interpolated datasets used by the viewer."""
    paths = {
        "tacrolimus": data_dir / "tacrolimus.parquet",
        "mpa": data_dir / "mpa.parquet",
        "mir155": data_dir / "lr_input.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos de dados não encontrados: {', '.join(missing)}"
        )
    data = {name: pd.read_parquet(path) for name, path in paths.items()}

    normalized = {
        name: normalizar_colunas(frame)
        .sort_values(["time", "id"])
        .reset_index(drop=True)
        for name, frame in data.items()
    }
    merge_inputs = [normalized["mir155"], normalized["mpa"], normalized["tacrolimus"]]
    data["merged"] = translating_flags(
        call_merge_db(merge_db, merge_inputs)
    ).sort_values(["id", "time"])

    interpolated_path = data_dir / "full_processed.parquet"
    if not interpolated_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {interpolated_path}")
    data["interpolated"] = pd.read_parquet(interpolated_path).sort_values(
        ["id", "time"]
    )
    return data


DATA = load_data()
PATIENT_IDS = sorted(
    {
        int(patient_id)
        for frame in (DATA["tacrolimus"], DATA["mpa"], DATA["mir155"])
        for patient_id in frame["#ID"].dropna().unique()
    }
)

EXPERIMENTS = {
    str(path.parent.relative_to(ROOT)): path.parent
    for path in sorted((ROOT / "studies" / "ml").glob("**/metadata.json"))
    if any(path.parent.glob("*/predictions.csv"))
}
DEFAULT_EXPERIMENT = next(reversed(EXPERIMENTS), None)


SERIES = {
    "tac_concentration": "Tacrolimo — concentração (ng/mL)",
    "tac_dose": "Tacrolimo — dose (mg)",
    "mpa_concentration": "MPA — concentração (µg/mL)",
    "mpa_dose": "MPA — dose (mg equivalentes)",
    "mir155": "miR-155-5p — expressão relativa",
    "lymphocytes": "Linfócitos (LYM)",
    "weight": "Peso corporal (kg)",
    "renal_function": "Função renal — GF (mL/min)",
    "infection": "Infecção geral (IFC)",
    "other_infection": "Outras infecções (OIFC)",
    "cmv_infection": "Citomegalovírus (CMV)",
    "bkv_infection": "BK vírus (BKV)",
}

STAGES = {
    "original": "Original",
    "merged": "Após o merge",
    "interpolated": "Após a interpolação",
}

BINARY_SERIES = {"infection", "other_infection", "cmv_infection", "bkv_infection"}

OCCASIONS = {
    1: "Primeira semana",
    2: "Mês 1",
    3: "Mês 2",
    4: "Mês 3",
    5: "Mês 6",
}


def occasion_options() -> list[dict[str, int | str]]:
    return [
        {
            "label": f"Ocorrência {occasion} — {description}",
            "value": occasion,
        }
        for occasion, description in OCCASIONS.items()
    ]


def format_study_date(value: str | None) -> str:
    if not value:
        return "não informada"
    try:
        if value.endswith("Z") and "_" in value:
            parsed = datetime.strptime(value, "%Y%m%dT%H%M%S_%fZ").replace(
                tzinfo=ZoneInfo("UTC")
            )
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(ZoneInfo("America/Sao_Paulo")).strftime(
            "%d/%m/%Y às %H:%M"
        )
    except ValueError:
        return value


def experiment_summary(study_key: str | None) -> dict | None:
    """Load a compact, backwards-compatible summary of an ML experiment."""
    if not study_key or study_key not in EXPERIMENTS:
        return None
    metadata_path = EXPERIMENTS[study_key] / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata.get("config") or {}
    preparation = metadata.get("data_preparation") or {}

    occasions = preparation.get("occasions")
    interpolation_applied = preparation.get("interpolation_applied")
    archived_preparation = EXPERIMENTS[study_key] / "data_preparation_metadata.json"
    if archived_preparation.exists() and (
        occasions is None or interpolation_applied is None
    ):
        details = json.loads(archived_preparation.read_text(encoding="utf-8"))
        selection = details.get("selecao_occ") or {}
        interpolation = details.get("interpolacao") or {}
        if occasions is None:
            occasions = selection.get("ocasioes_mantidas")
        if interpolation_applied is None:
            interpolation_applied = interpolation.get("aplicada")

    use_lym = config.get("use_lym")
    if use_lym is None:
        features = metadata.get("features") or []
        use_lym = any(
            feature in {"initial_lym", "d_time_lym", "val_lym", "tend_lym"}
            for feature in features
        )

    models = metadata.get("models") or {}
    model_names = list(models) if isinstance(models, dict) else models
    return {
        "name": Path(study_key).name,
        "date": format_study_date(metadata.get("started_at_utc")),
        "folds": config.get("folds"),
        "search": config.get("search"),
        "normalization": config.get("normalization"),
        "variable_mode": config.get("var_mode"),
        "occasions": occasions or [],
        "use_lym": bool(use_lym),
        "interpolation_applied": (
            None if interpolation_applied is None else bool(interpolation_applied)
        ),
        "models": model_names,
    }


def study_description(study_key: str | None) -> html.Div:
    """Render experiment metadata as a visual panel instead of prose."""
    summary = experiment_summary(study_key)
    if summary is None:
        return html.Div("Nenhum experimento de ML compatível foi encontrado.")

    occasions = ", ".join(str(value) for value in summary["occasions"]) or "N/D"
    models = ", ".join(
        str(model).replace("_", " ").title() for model in summary["models"]
    ) or "N/D"
    items = [
        ("Data", summary["date"]),
        ("Ocorrências", occasions),
        ("Histórico de LYM", "Sim" if summary["use_lym"] else "Não"),
        ("Variáveis", summary["variable_mode"] or "N/D"),
        ("Validação", f"{summary['folds']} folds" if summary["folds"] else "N/D"),
        ("Busca", summary["search"] or "N/D"),
        ("Normalização", summary["normalization"] or "N/D"),
        (
            "Interpolação prévia",
            (
                "N/D"
                if summary["interpolation_applied"] is None
                else "Sim" if summary["interpolation_applied"] else "Não"
            ),
        ),
        ("Modelos", models),
    ]

    cards = [
        html.Div(
            [
                html.Div(label, style={"fontSize": "0.76rem", "color": "#64748b"}),
                html.Div(
                    value,
                    style={"fontWeight": 650, "marginTop": "3px", "color": "#0f172a"},
                ),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "9px",
                "padding": "9px 11px",
            },
        )
        for label, value in items
    ]
    return html.Div(
        [
            html.H3(summary["name"], style={"margin": "2px 0 10px"}),
            html.Div(
                cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                    "gap": "8px",
                },
            ),
        ]
    )


@lru_cache(maxsize=None)
def experiment_test_patients(study_key: str | None) -> pd.DataFrame:
    """Return test patient IDs and evaluated row counts for one experiment."""
    columns = ["id", "measurements"]
    if not study_key or study_key not in EXPERIMENTS:
        return pd.DataFrame(columns=columns)

    experiment = EXPERIMENTS[study_key]
    split_path = experiment / "splits.csv"
    if split_path.exists():
        splits = pd.read_csv(split_path)
        if {"id", "split"}.issubset(splits.columns):
            test = splits.loc[splits["split"].astype(str).str.lower().eq("test")]
            if not test.empty:
                result = (
                    test.groupby("id", as_index=False)
                    .size()
                    .rename(columns={"size": "measurements"})
                )
                result["id"] = pd.to_numeric(result["id"], errors="coerce")
                return result.dropna(subset=["id"]).sort_values("id").reset_index(
                    drop=True
                )

    # Compatibility with archived studies that predate splits.csv.
    prediction_frames = []
    for prediction_path in experiment.glob("*/predictions.csv"):
        predictions = pd.read_csv(prediction_path, usecols=lambda name: name == "id")
        if "id" in predictions:
            prediction_frames.append(predictions[["id"]])
    if not prediction_frames:
        return pd.DataFrame(columns=columns)
    predictions = pd.concat(prediction_frames, ignore_index=True).drop_duplicates()
    predictions["id"] = pd.to_numeric(predictions["id"], errors="coerce")
    result = predictions.dropna(subset=["id"]).sort_values("id").reset_index(drop=True)
    result["measurements"] = pd.NA
    return result[columns]


def test_patients_panel(study_key: str | None) -> html.Div:
    """Render the patients that compose the selected experiment test set."""
    patients = experiment_test_patients(study_key)
    if patients.empty:
        return html.Div(
            [
                html.Strong("Pacientes do conjunto de teste"),
                html.P(
                    "Não foi possível identificar pacientes de teste neste experimento.",
                    style={"margin": "8px 0 0", "color": "#64748b"},
                ),
            ]
        )

    badges = []
    for row in patients.itertuples(index=False):
        patient_id = int(row.id) if float(row.id).is_integer() else row.id
        count = (
            f" · {int(row.measurements)} medidas"
            if pd.notna(row.measurements)
            else ""
        )
        badges.append(
            html.Span(
                f"Paciente {patient_id}{count}",
                style={
                    "display": "inline-block",
                    "backgroundColor": "#e0f2fe",
                    "border": "1px solid #7dd3fc",
                    "borderRadius": "999px",
                    "color": "#075985",
                    "fontSize": "0.86rem",
                    "fontWeight": 600,
                    "margin": "5px 7px 0 0",
                    "padding": "5px 10px",
                },
            )
        )
    return html.Div(
        [
            html.Strong(
                f"Pacientes do conjunto de teste ({len(patients)})",
                style={"display": "block"},
            ),
            html.Div(badges, style={"marginTop": "3px"}),
        ]
    )


def binary_patient_status(patient_id: int, column: str) -> str:
    """Summarize whether a binary event occurred in any original dataset."""
    values = []
    for frame in (DATA["mir155"], DATA["tacrolimus"], DATA["mpa"]):
        if column not in frame.columns:
            continue
        patient_values = pd.to_numeric(
            frame.loc[frame["#ID"] == patient_id, column], errors="coerce"
        )
        values.extend(patient_values.dropna().loc[patient_values.ne(9999)].tolist())
    if not values:
        return "Sem informação"
    return "Sim" if any(value == 1 for value in values) else "Não"


def renal_function_summary(patient_id: int) -> tuple[str, str]:
    """Return the latest valid GF value and its quantitative interpretation."""
    frame = DATA["mir155"]
    patient = frame.loc[frame["#ID"] == patient_id, ["TIME", "GF"]].copy()
    patient["GF"] = pd.to_numeric(patient["GF"], errors="coerce")
    patient = patient.loc[patient["GF"].notna() & patient["GF"].ne(9999)]
    if patient.empty:
        return "Sem informação", "Nenhum valor válido de GF"

    latest = patient.sort_values("TIME").iloc[-1]
    value = float(latest["GF"])
    if value < 15:
        category = "falência renal/estágio terminal"
    elif value < 30:
        category = "redução grave"
    elif value < 60:
        category = "redução moderada"
    elif value < 90:
        category = "redução leve"
    else:
        category = "função ≥ 90 mL/min"
    return f"{value:g} mL/min", f"{category}; avaliação em {latest['TIME']:.2f} h"


def patient_clinical_summary(patient_id: int) -> html.Div:
    statuses = [
        ("Rejeição aguda", binary_patient_status(patient_id, "AR")),
        ("Infecção geral", binary_patient_status(patient_id, "IFC")),
        ("Outras infecções", binary_patient_status(patient_id, "OIFC")),
        ("Diabetes mellitus", binary_patient_status(patient_id, "DM")),
        ("BK vírus", binary_patient_status(patient_id, "BKV")),
        ("Citomegalovírus", binary_patient_status(patient_id, "CMV")),
    ]
    renal_value, renal_detail = renal_function_summary(patient_id)

    def status_card(label: str, value: str, detail: str | None = None) -> html.Div:
        color = "#b91c1c" if value == "Sim" else "#166534"
        if value == "Sem informação":
            color = "#64748b"
        return html.Div(
            [
                html.Div(label, style={"fontSize": "0.82rem", "color": "#64748b"}),
                html.Div(
                    value,
                    style={"fontSize": "1.1rem", "fontWeight": 700, "color": color},
                ),
                (
                    html.Div(detail, style={"fontSize": "0.78rem", "color": "#64748b"})
                    if detail
                    else None
                ),
            ],
            style={
                "border": "1px solid #e2e8f0",
                "borderRadius": "9px",
                "padding": "10px 12px",
                "backgroundColor": "#f8fafc",
            },
        )

    cards = [status_card(label, value) for label, value in statuses]
    
    return html.Div(
        [
            html.H3(
                f"Situação clínica — paciente {patient_id}",
                style={"margin": "4px 0 12px"},
            ),
            html.Div(
                cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "9px",
                },
            ),
            html.P(
                "“Sim” indica ao menos um registro positivo durante o acompanhamento; "
                "não representa necessariamente a situação clínica atual.",
                style={"fontSize": "0.78rem", "color": "#64748b", "marginBottom": 0},
            ),
        ]
    )


@lru_cache(maxsize=None)
def load_study_pipeline(study_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a study pipeline from its archived source and current raw inputs."""
    if study_key == "current":
        return DATA["merged"], DATA["interpolated"]
    source = STUDY_SOURCES[study_key]
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path(str(source))
        raw_frames = [DATA["mir155"], DATA["mpa"], DATA["tacrolimus"]]
        normalized = [
            namespace["normalizar_colunas"](frame)
            .sort_values(["time", "id"])
            .reset_index(drop=True)
            for frame in raw_frames
        ]
        merged = namespace["translating_flags"](
            call_merge_db(namespace["merge_db"], normalized)
        ).sort_values(["id", "time"])
        interpolated = namespace["ts_interpolation"](merged.copy()).sort_values(
            ["id", "time"]
        )
    return merged, interpolated


def empty_figure(
    title: str, message: str = "Sem observações para este paciente"
) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
    )
    return style_figure(figure, title, "")


def style_figure(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.02},
        template="plotly_white",
        height=430,
        margin={"l": 65, "r": 25, "t": 65, "b": 55},
        hovermode="x unified",
        xaxis_title="Tempo desde o transplante (horas)",
        yaxis_title=y_title,
        legend_title_text="Etapa",
    )
    figure.update_xaxes(
        showspikes=True,
        spikemode="across",
        rangeslider_visible=True,
        rangemode="tozero",
    )
    return figure


def original_patient_series(
    patient_id: int, occasion: int | None = None
) -> dict[str, pd.DataFrame]:
    """Prepare concentrations, recorded doses and biomarker values for a patient."""
    tac = DATA["tacrolimus"]
    mpa = DATA["mpa"]
    mir = DATA["mir155"]

    if occasion is not None:
        occasion = int(occasion)
        tac = tac.loc[pd.to_numeric(tac["OCC"], errors="coerce") == occasion]
        mpa = mpa.loc[pd.to_numeric(mpa["OCC"], errors="coerce") == occasion]
        mir = mir.loc[pd.to_numeric(mir["OCC"], errors="coerce") == occasion]

    tac_observations = tac.loc[
        (tac["#ID"] == patient_id) & (tac["MDV"] == 0) & (tac["EVID"] == 0)
    ].copy()
    tac_observations["VALUE"] = np.exp(tac_observations["DV"])

    mpa_observations = mpa.loc[
        (mpa["#ID"] == patient_id) & (mpa["MDV"] == 0) & (mpa["EVID"] == 0)
    ].copy()
    mpa_observations["VALUE"] = mpa_observations["DV"]

    tac_doses = tac.loc[
        (tac["#ID"] == patient_id) & (tac["EVID"] == 1) & (tac["AMT"] > 0)
    ].copy()
    tac_doses["VALUE"] = tac_doses["AMT"]

    mpa_doses = mpa.loc[
        (mpa["#ID"] == patient_id) & (mpa["EVID"] == 1) & (mpa["AMT"] > 0)
    ].copy()
    mpa_doses["VALUE"] = mpa_doses["AMT"]

    mir_observations = mir.loc[
        (mir["#ID"] == patient_id)
        & mir["M155"].notna()
        & (mir["M155"] > 0)
        & (mir["M155"] != 9999)
    ].copy()
    mir_observations["VALUE"] = mir_observations["M155"]

    lymphocyte_observations = mir.loc[
        (mir["#ID"] == patient_id) & mir["LYM"].notna() & (mir["LYM"] != 9999)
    ].copy()
    lymphocyte_observations["VALUE"] = lymphocyte_observations["LYM"]

    weight_observations = mir.loc[
        (mir["#ID"] == patient_id)
        & mir["WGT"].notna()
        & (mir["WGT"] != 9999)
    ].copy()
    weight_observations["VALUE"] = weight_observations["WGT"]

    renal_observations = mir.loc[
        (mir["#ID"] == patient_id)
        & mir["GF"].notna()
        & (mir["GF"] != 9999)
    ].copy()
    renal_observations["VALUE"] = renal_observations["GF"]

    def binary_observations(column: str) -> pd.DataFrame:
        observations = mir.loc[
            (mir["#ID"] == patient_id)
            & mir[column].notna()
            & (mir[column] != 9999)
        ].copy()
        observations["VALUE"] = observations[column]
        return observations.sort_values("TIME")

    return {
        "tac_concentration": tac_observations.sort_values("TIME"),
        "tac_dose": tac_doses.sort_values("TIME"),
        "mpa_concentration": mpa_observations.sort_values("TIME"),
        "mpa_dose": mpa_doses.sort_values("TIME"),
        "mir155": mir_observations.sort_values("TIME"),
        "lymphocytes": lymphocyte_observations.sort_values("TIME"),
        "weight": weight_observations.sort_values("TIME"),
        "renal_function": renal_observations.sort_values("TIME"),
        "infection": binary_observations("IFC"),
        "other_infection": binary_observations("OIFC"),
        "cmv_infection": binary_observations("CMV"),
        "bkv_infection": binary_observations("BKV"),
    }


@lru_cache(maxsize=None)
def population_measure_counts(series_key: str, occasion: int) -> pd.DataFrame:
    """Count original measurements for every patient in one occasion."""
    if series_key not in SERIES:
        raise ValueError(f"Série desconhecida: {series_key}")
    rows = []
    for patient_id in PATIENT_IDS:
        frame = original_patient_series(patient_id, occasion)[series_key]
        rows.append({"patient_id": patient_id, "count": len(frame)})
    return pd.DataFrame(rows)


def processed_patient_series(
    frame: pd.DataFrame, patient_id: int
) -> dict[str, pd.DataFrame]:
    """Convert a merged/pipeline frame to the same visual series as the raw data."""
    patient = frame.loc[frame["id"] == patient_id].copy()

    def make_series(column: str, *, positive: bool = False, exponentiate: bool = False):
        selected = patient.loc[patient[column].notna(), ["time", column]].copy()
        if positive:
            selected = selected.loc[selected[column] > 0]
        selected["VALUE"] = (
            np.exp(selected[column]) if exponentiate else selected[column]
        )
        return selected.rename(columns={"time": "TIME"})[["TIME", "VALUE"]]

    return {
        "tac_concentration": make_series("dv_tac", exponentiate=True),
        "tac_dose": make_series("amt_tac", positive=True),
        "mpa_concentration": make_series("dv_map"),
        "mpa_dose": make_series("amt_map", positive=True),
        "mir155": make_series("m155", positive=True),
        "lymphocytes": make_series("lym"),
        "weight": make_series("wgt"),
        "renal_function": make_series("gf"),
        "infection": make_series("ifc"),
        "other_infection": make_series("oifc"),
        "cmv_infection": make_series("cmv"),
        "bkv_infection": make_series("bkv"),
    }


def series_by_stage(
    patient_id: int, series_key: str, study_key: str
) -> dict[str, pd.DataFrame]:
    merged, interpolated = load_study_pipeline(study_key)
    original = original_patient_series(patient_id)[series_key].copy()
    after_merge = processed_patient_series(merged, patient_id)[series_key].copy()
    after_interpolation = processed_patient_series(interpolated, patient_id)[
        series_key
    ].copy()

    def mark_added(frame: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        previous_times = set(previous["TIME"].round(6))
        frame["ADDED"] = ~frame["TIME"].round(6).isin(previous_times)
        return frame

    original["ADDED"] = False
    return {
        "original": original,
        "merged": mark_added(after_merge, original),
        "interpolated": mark_added(after_interpolation, after_merge),
    }


def original_figure(
    patient_id: int,
    occasion: int | None,
    selected_series: list[str] | None,
    plot_type: str,
) -> go.Figure:
    frames = original_patient_series(patient_id, occasion)
    selected_series = selected_series or []
    title = "Séries temporais originais"
    if occasion is not None:
        title += f" — ocorrência {int(occasion)}"
    if not selected_series:
        return empty_figure(title, "Selecione uma série")

    colors = {
        "tac_concentration": "#2563eb",
        "tac_dose": "#60a5fa",
        "mpa_concentration": "#dc2626",
        "mpa_dose": "#fb7185",
        "mir155": "#7c3aed",
        "lymphocytes": "#059669",
        "weight": "#a16207",
        "renal_function": "#0891b2",
        "infection": "#be123c",
        "other_infection": "#c2410c",
        "cmv_infection": "#9333ea",
        "bkv_infection": "#4f46e5",
    }
    figure = go.Figure()
    for series_key in selected_series:
        if series_key not in frames or frames[series_key].empty:
            continue
        frame = frames[series_key]
        figure.add_trace(
            go.Scatter(
                x=frame["TIME"],
                y=frame["VALUE"],
                mode="markers" if plot_type == "scatter" else "lines+markers",
                name=SERIES[series_key],
                marker={"size": 8, "color": colors[series_key], "opacity": 0.82},
                hovertemplate=(
                    f"{SERIES[series_key]}<br>Tempo: %{{x:.2f}} h"
                    "<br>Valor: %{y:.4g}<extra></extra>"
                ),
            )
        )
    if not figure.data:
        return empty_figure(title)
    figure = style_figure(figure, title, "Valor na unidade original")
    maximum_time = max(max(trace.x) for trace in figure.data)
    figure.update_xaxes(range=[0, maximum_time * 1.02 if maximum_time > 0 else 1])
    return figure


def population_count_histogram(series_key: str, occasion: int) -> go.Figure:
    """Plot the distribution of patient measurement counts for one occasion."""
    if series_key not in SERIES or int(occasion) not in OCCASIONS:
        return empty_figure(
            "Distribuição do número de medidas", "Selecione variável e ocorrência"
        )
    occasion = int(occasion)
    counts = population_measure_counts(series_key, occasion)
    maximum = int(counts["count"].max()) if not counts.empty else 0
    figure = go.Figure(
        go.Histogram(
            x=counts["count"],
            xbins={"start": -0.5, "end": maximum + 0.5, "size": 1},
            marker={"color": "#2563eb", "line": {"color": "white", "width": 1}},
            hovertemplate=(
                "Número de medidas: %{x}<br>Pacientes: %{y}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                f"Distribuição por paciente — {SERIES[series_key]} — "
                f"ocorrência {occasion}"
            ),
            "x": 0.02,
        },
        template="plotly_white",
        height=430,
        margin={"l": 65, "r": 25, "t": 70, "b": 60},
        bargap=0.08,
        xaxis_title="Número de medidas originais por paciente",
        yaxis_title="Número de pacientes",
        showlegend=False,
    )
    figure.update_xaxes(dtick=1, rangemode="tozero")
    figure.update_yaxes(dtick=1, rangemode="tozero")
    return figure


def experiment_lym_figure(
    patient_id: int, experiment_key: str | None, plot_type: str
) -> go.Figure:
    """Plot the complete original LYM series with experiment predictions."""
    if not experiment_key or experiment_key not in EXPERIMENTS:
        return empty_figure(
            "LYM original e previsto", "Selecione um experimento compatível"
        )

    original = original_patient_series(patient_id)["lymphocytes"]
    figure = go.Figure()
    if not original.empty:
        figure.add_trace(
            go.Scatter(
                x=original["TIME"],
                y=original["VALUE"],
                mode="markers" if plot_type == "scatter" else "lines+markers",
                name="LYM original",
                marker={"size": 9, "color": "#2563eb"},
                line={"color": "#2563eb", "width": 2},
                hovertemplate=(
                    "LYM original<br>Tempo: %{x:.2f} h"
                    "<br>Valor: %{y:.4g}<extra></extra>"
                ),
            )
        )

    experiment = EXPERIMENTS[experiment_key]
    metadata = json.loads(
        (experiment / "metadata.json").read_text(encoding="utf-8")
    )
    colors = ["#dc2626", "#059669", "#7c3aed", "#d97706"]
    prediction_count = 0
    for color, model in zip(colors, metadata.get("models") or []):
        prediction_path = experiment / model / "predictions.csv"
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        patient = predictions.loc[predictions["id"] == patient_id].sort_values("time")
        if patient.empty:
            continue
        prediction_count += len(patient)
        figure.add_trace(
            go.Scatter(
                x=patient["time"],
                y=patient["predicted"],
                mode="markers" if plot_type == "scatter" else "lines+markers",
                name=f"LYM previsto — {model.replace('_', ' ').title()}",
                marker={"size": 10, "symbol": "diamond", "color": color},
                line={"color": color, "width": 2, "dash": "dash"},
                hovertemplate=(
                    f"{model.replace('_', ' ').title()}"
                    "<br>Tempo: %{x:.2f} h"
                    "<br>LYM previsto: %{y:.4g}<extra></extra>"
                ),
            )
        )

    if not figure.data:
        return empty_figure("LYM original e previsto")
    if prediction_count == 0:
        figure.add_annotation(
            text="Este paciente não pertence ao conjunto de teste do experimento",
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
        )
    figure = style_figure(
        figure,
        f"LYM original e previsto — paciente {patient_id}",
        "LYM",
    )
    figure.update_layout(legend_title_text="Série")
    return figure


def series_figure(
    patient_id: int,
    series_key: str,
    selected_stages: list[str] | None,
    plot_type: str,
    study_key: str,
) -> go.Figure:
    selected_stages = selected_stages or []
    if not selected_stages:
        return empty_figure(SERIES[series_key], "Selecione pelo menos uma etapa")

    colors = {"original": "#2563eb", "merged": "#f59e0b", "interpolated": "#059669"}
    symbols = {"original": "circle", "merged": "diamond", "interpolated": "cross"}
    stage_frames = series_by_stage(patient_id, series_key, study_key)
    figure = go.Figure()
    for stage in selected_stages:
        if stage not in STAGES:
            continue
        frame = stage_frames[stage]
        if frame.empty:
            continue
        point_status = np.where(
            frame["ADDED"], "Adicionado nesta etapa", "Já existente na etapa anterior"
        )
        figure.add_trace(
            go.Scatter(
                x=frame["TIME"],
                y=frame["VALUE"],
                mode="markers" if plot_type == "scatter" else "lines+markers",
                name=STAGES[stage],
                marker={
                    "size": 8,
                    "color": colors[stage],
                    "symbol": symbols[stage],
                    "opacity": 0.8,
                },
                customdata=point_status,
                hovertemplate=(
                    f"{STAGES[stage]}<br>Tempo: %{{x:.2f}} h"
                    "<br>Valor: %{y:.4g}<br>%{customdata}<extra></extra>"
                ),
            )
        )
        added = frame.loc[frame["ADDED"]]
        if not added.empty:
            figure.add_trace(
                go.Scatter(
                    x=added["TIME"],
                    y=added["VALUE"],
                    mode="markers",
                    name=f"{STAGES[stage]} — pontos adicionados",
                    marker={
                        "size": 8,
                        "color": colors[stage],
                        "symbol": "star",
                        "line": {"color": "#0f172a", "width": 1.5},
                    },
                    hovertemplate=(
                        f"{STAGES[stage]} — adicionado"
                        "<br>Tempo: %{x:.2f} h"
                        "<br>Valor: %{y:.4g}<extra></extra>"
                    ),
                )
            )
    if not figure.data:
        return empty_figure(SERIES[series_key])
    figure = style_figure(figure, SERIES[series_key], "Valor")
    maximum_time = max(max(trace.x) for trace in figure.data)
    figure.update_xaxes(range=[0, maximum_time * 1.02 if maximum_time > 0 else 1])
    if series_key in BINARY_SERIES:
        figure.update_yaxes(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Não", "Sim"],
            range=[-0.1, 1.1],
        )
    return figure


app = Dash(__name__, title="Séries temporais por paciente")
server = app.server

CARD_STYLE = {
    "backgroundColor": "white",
    "border": "1px solid #e2e8f0",
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.05)",
    "padding": "8px 14px",
}


def series_panel(series_key: str) -> html.Div:
    return html.Div(
        [
            html.Label(
                "Etapas exibidas",
                style={"fontWeight": 600, "display": "block", "padding": "8px 8px 0"},
            ),
            dcc.Dropdown(
                id={"type": "stage-select", "series": series_key},
                options=[
                    {"label": label, "value": key} for key, label in STAGES.items()
                ],
                value=list(STAGES),
                multi=True,
                clearable=True,
                placeholder="Selecione uma ou mais etapas",
                style={"margin": "6px 8px 0"},
            ),
            dcc.Graph(id={"type": "series-graph", "series": series_key}),
        ],
        style={**CARD_STYLE, "marginTop": "14px"},
    )


app.layout = html.Main(
    [
        html.H1("Séries temporais originais", style={"marginBottom": "4px"}),
        html.P(
            "Visualização conjunta das medidas originais, sem merge ou interpolação.",
            style={"color": "#475569", "marginTop": 0},
        ),
        html.H2("Visão geral de todos os pacientes", style={"marginBottom": "4px"}),
        html.P(
            "Distribuição do número de medidas originais por paciente na variável e "
            "ocorrência selecionadas. Pacientes sem medidas entram na contagem zero.",
            style={"color": "#475569", "marginTop": 0},
        ),
        html.Label(
            "Variável",
            htmlFor="population-series-select",
            style={"fontWeight": 600},
        ),
        dcc.Dropdown(
            id="population-series-select",
            options=[{"label": label, "value": key} for key, label in SERIES.items()],
            value="lymphocytes",
            clearable=False,
            searchable=True,
            style={"maxWidth": "520px", "marginTop": "6px"},
        ),
        html.Label(
            "Ocorrência",
            htmlFor="occasion-select",
            style={"fontWeight": 600, "display": "block", "marginTop": "18px"},
        ),
        dcc.Dropdown(
            id="occasion-select",
            options=occasion_options(),
            value=1,
            clearable=False,
            searchable=False,
            style={"maxWidth": "520px", "marginTop": "6px"},
        ),
        html.Div(
            dcc.Graph(id="population-count-histogram"),
            style={**CARD_STYLE, "marginTop": "14px", "marginBottom": "36px"},
        ),
        html.H2("Visualização por paciente", style={"marginBottom": "14px"}),
        html.Label("Paciente", htmlFor="patient-select", style={"fontWeight": 600}),
        dcc.Dropdown(
            id="patient-select",
            options=[
                {"label": f"Paciente {patient_id}", "value": patient_id}
                for patient_id in PATIENT_IDS
            ],
            value=PATIENT_IDS[0],
            clearable=False,
            searchable=True,
            style={"maxWidth": "420px", "marginTop": "6px"},
        ),
        html.Label(
            "Tipo de gráfico",
            style={"fontWeight": 600, "display": "block", "marginTop": "18px"},
        ),
        dcc.RadioItems(
            id="plot-type",
            options=[
                {"label": " Scatter", "value": "scatter"},
                {"label": " Linha", "value": "line"},
            ],
            value="scatter",
            inline=True,
            inputStyle={"marginLeft": "12px"},
            style={"marginTop": "8px"},
        ),
        html.Div(id="patient-summary", style={**CARD_STYLE, "marginTop": "18px"}),
        html.Label(
            "Séries exibidas",
            htmlFor="original-series-select",
            style={"fontWeight": 600, "display": "block", "marginTop": "18px"},
        ),
        dcc.Dropdown(
            id="original-series-select",
            options=[{"label": label, "value": key} for key, label in SERIES.items()],
            value=list(SERIES),
            multi=True,
            clearable=True,
            placeholder="Selecione uma ou mais séries",
            style={"marginTop": "6px"},
        ),
        html.Div(
            dcc.Graph(id="original-timeseries-graph"),
            style={**CARD_STYLE, "marginTop": "14px"},
        ),
        html.Hr(
            style={"border": 0, "borderTop": "1px solid #cbd5e1", "margin": "42px 0"}
        ),
        html.H1("Previsões de LYM", style={"marginBottom": "4px"}),
        html.P(
            "Série original completa de LYM com as previsões do experimento selecionado.",
            style={"color": "#475569", "marginTop": 0},
        ),
        html.Label(
            "Experimento", htmlFor="study-select", style={"fontWeight": 600}
        ),
        dcc.Dropdown(
            id="study-select",
            options=[
                {"label": experiment_key, "value": experiment_key}
                for experiment_key in EXPERIMENTS
            ],
            value=DEFAULT_EXPERIMENT,
            clearable=False,
            searchable=True,
            style={"marginTop": "6px"},
        ),
        html.Div(
            id="study-description",
            children=study_description(DEFAULT_EXPERIMENT),
            style={
                **CARD_STYLE,
                "marginTop": "12px",
                "lineHeight": "1.55",
                "color": "#334155",
            },
        ),
        html.Div(
            id="test-patients-panel",
            children=test_patients_panel(DEFAULT_EXPERIMENT),
            style={
                **CARD_STYLE,
                "marginTop": "12px",
                "lineHeight": "1.4",
                "color": "#334155",
            },
        ),
        html.Div(
            dcc.Graph(id="experiment-lym-graph"),
            style={**CARD_STYLE, "marginTop": "14px"},
        ),
    ],
    style={
        "fontFamily": "system-ui, -apple-system, sans-serif",
        "maxWidth": "1200px",
        "margin": "28px auto",
        "padding": "0 20px 30px",
        "color": "#0f172a",
        "backgroundColor": "#f8fafc",
    },
)


@app.callback(
    Output("original-timeseries-graph", "figure"),
    Input("patient-select", "value"),
    Input("occasion-select", "value"),
    Input("original-series-select", "value"),
    Input("plot-type", "value"),
)
def update_original_graph(
    patient_id: int,
    occasion: int,
    selected_series: list[str] | None,
    plot_type: str,
):
    return original_figure(int(patient_id), occasion, selected_series, plot_type)


@app.callback(
    Output("population-count-histogram", "figure"),
    Input("population-series-select", "value"),
    Input("occasion-select", "value"),
)
def update_population_count_histogram(series_key: str, occasion: int):
    return population_count_histogram(series_key, occasion)


@app.callback(Output("study-description", "children"), Input("study-select", "value"))
def update_study_description(study_key: str):
    return study_description(study_key)


@app.callback(
    Output("test-patients-panel", "children"), Input("study-select", "value")
)
def update_test_patients_panel(study_key: str | None):
    return test_patients_panel(study_key)


@app.callback(
    Output("experiment-lym-graph", "figure"),
    Input("patient-select", "value"),
    Input("study-select", "value"),
    Input("plot-type", "value"),
)
def update_experiment_lym_graph(
    patient_id: int, experiment_key: str | None, plot_type: str
):
    return experiment_lym_figure(int(patient_id), experiment_key, plot_type)


@app.callback(Output("patient-summary", "children"), Input("patient-select", "value"))
def update_patient_summary(patient_id: int):
    return patient_clinical_summary(int(patient_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Interface do servidor")
    parser.add_argument("--port", default=8050, type=int, help="Porta do servidor")
    parser.add_argument(
        "--debug", action="store_true", help="Ativa recarga e mensagens de depuração"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
