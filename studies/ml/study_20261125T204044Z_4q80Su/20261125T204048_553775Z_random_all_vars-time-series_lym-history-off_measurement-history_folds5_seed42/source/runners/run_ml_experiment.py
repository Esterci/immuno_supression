"""Executa um experimento reproduzível com busca por grupos e teste separado."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from data_preparation import (
    EXCLUDED_VARIABLE_COLUMNS,
    INTERPOLATION_FLAG_PREFIX,
    NON_TEMPORAL_COLUMNS,
)
from methods.ml_models import model_specs, make_search
from utils.ml_reports import (
    write_json,
    regression_metrics,
    prediction_plot,
    shap_report,
    summarize,
)
from utils.analyze_patient_timeseries import analyze as analyze_patient_timeseries

BASE = Path(__file__).resolve().parents[1]


def prepare_measurement_features(data, target="lym", var_mode="all", use_lym=True):
    """Formata uma linha por medida do alvo usando apenas o histórico disponível."""
    if var_mode not in {"all", "time-series"}:
        raise ValueError("var_mode deve ser all ou time-series")
    required = {"id", "time", target}

    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"Colunas ausentes: {missing}")
    if data[["id", "time"]].isna().any().any():
        raise ValueError("id e time não podem conter valores ausentes")

    data = data.sort_values(["id", "time"], kind="stable").copy()

    available_static_columns = [
        column
        for column in NON_TEMPORAL_COLUMNS
        if column in data.columns and column not in {"id", "time", target}
    ]

    static_columns = available_static_columns if var_mode == "all" else []

    # Classifica as séries a partir das colunas disponíveis, preservando a ordem
    # original. Flags de interpolação são metadados, nunca preditores.
    temporal_column_set = (
        set(data.columns)
        - set(NON_TEMPORAL_COLUMNS)
        - set(EXCLUDED_VARIABLE_COLUMNS)
        - {"id", "time"}
    )
    if not use_lym:
        temporal_column_set.discard(target)
    temporal_columns = [
        column
        for column in data.columns
        if column in temporal_column_set
        and not column.startswith(INTERPOLATION_FLAG_PREFIX)
    ]

    target_flag = f"{INTERPOLATION_FLAG_PREFIX}{target}"
    target_rows = data[target].notna()
    if target_flag in data.columns:
        # Valores interpolados participam do histórico dos preditores, mas
        # nunca são tratados como verdade de referência supervisionada.
        target_rows &= ~data[target_flag].fillna(0).astype(bool)
    supervised = data.loc[target_rows]
    if supervised.empty:
        raise ValueError(f"Nenhuma medida real observada de {target}")

    # ID identifica os grupos, mas não é oferecido ao modelo. O tempo é feature.
    X = supervised[["time", *static_columns]].copy()
    y = pd.to_numeric(supervised[target], errors="coerce").copy()
    rows = supervised[["id", "time"]].copy()
    if target_flag in supervised.columns:
        rows["interpolated"] = supervised[target_flag].fillna(0).astype(bool)
    generated_columns = []

    for predictor in temporal_columns:
        observed = data.loc[
            data[predictor].notna(), ["id", "time", predictor]
        ].copy()
        observed[predictor] = pd.to_numeric(observed[predictor], errors="coerce")
        observed = observed.dropna(subset=[predictor])
        histories = {
            patient_id: history.sort_values("time", kind="stable")
            for patient_id, history in observed.groupby("id", sort=False)
        }

        initial = pd.Series(np.nan, index=supervised.index, dtype=float)
        delta_time = pd.Series(np.nan, index=supervised.index, dtype=float)
        value = pd.Series(np.nan, index=supervised.index, dtype=float)
        tendency = pd.Series(np.nan, index=supervised.index, dtype=float)

        for index, measurement in supervised[["id", "time"]].iterrows():
            history = histories.get(measurement["id"])
            if history is None:
                continue
            # Somente medidas estritamente anteriores ao alvo são elegíveis.
            eligible = history.loc[history["time"] < measurement["time"]]
            if eligible.empty:
                continue

            initial.loc[index] = eligible.iloc[0][predictor]
            if len(eligible) < 2:
                continue

            previous = eligible.iloc[-1]
            penultimate = eligible.iloc[-2]
            delta_time.loc[index] = measurement["time"] - previous["time"]
            value.loc[index] = previous[predictor]
            elapsed = previous["time"] - penultimate["time"]
            tendency.loc[index] = (
                (previous[predictor] - penultimate[predictor]) / elapsed
                if elapsed != 0
                else 0.0
            )

        predictor_features = [
            f"initial_{predictor}",
            f"d_time_{predictor}",
            f"val_{predictor}",
            f"tend_{predictor}",
        ]
        X[predictor_features] = pd.concat(
            [initial, delta_time, value, tendency], axis=1
        ).to_numpy()
        generated_columns.extend(predictor_features)

    if not np.isfinite(y).all():
        raise ValueError(f"O alvo {target} contém valores não numéricos ou infinitos")
    X = X.replace([np.inf, -np.inf], np.nan)
    predictable = X[generated_columns].notna().all(axis=1)
    X, y, rows = X.loc[predictable], y.loc[predictable], rows.loc[predictable]
    if X.empty:
        raise ValueError(
            "Nenhuma medida de LYM possui histórico temporal completo para previsão"
        )
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError("Features contêm valores ausentes: " f"{missing}")

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        rows.reset_index(drop=True),
    )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=("Parquet de entrada. Por padrão usa clean_processed.parquet."),
    )
    parser.add_argument("--output-dir", type=Path, default=BASE / "studies/ml")
    parser.add_argument("--search", choices=["grid", "random"], default="random")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--normalization", choices=["all"], default="all")
    parser.add_argument(
        "--var-mode",
        choices=["all", "time-series"],
        default="all",
        help=(
            "all inclui variáveis não temporais; time-series usa somente time e "
            "atributos derivados das séries temporais"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(model_specs()), default=list(model_specs())
    )
    parser.add_argument("--shap-samples", type=int, default=100)
    parser.add_argument("--shap-background", type=int, default=50)
    parser.add_argument("--shap-permutations", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grade pequena para validar o fluxo; não usar como estudo final",
    )
    parser.add_argument(
        "--use-lym",
        type=int,
        choices=[0, 1],
        default=1,
        help="Usa (1) ou não usa (0) o histórico de linfócitos como entrada",
    )
    args = parser.parse_args(argv)
    if args.folds < 2 or args.n_iter < 1 or args.n_jobs == 0 or args.seed < 0:
        parser.error("Exige folds >= 2, n-iter >= 1, n-jobs != 0, seed >= 0")
    if (
        not 0 < args.test_size < 1
        or min(args.shap_samples, args.shap_background, args.shap_permutations) < 1
    ):
        parser.error("Exige 0 < test-size < 1 e parâmetros SHAP positivos")
    args.models = list(dict.fromkeys(args.models))
    args.use_lym = bool(args.use_lym)
    if args.data is None:
        args.data = BASE / "data" / "clean_processed.parquet"
    return args


def execute(args, output, metadata):
    import shap  # Falha antes de treinar se a dependência estiver ausente.

    data = pd.read_parquet(args.data)
    real_target_rows = data["lym"].notna()
    target_flag = f"{INTERPOLATION_FLAG_PREFIX}lym"
    if target_flag in data.columns:
        real_target_rows &= ~data[target_flag].fillna(0).astype(bool)
    X, y, rows = prepare_measurement_features(
        data, target="lym", var_mode=args.var_mode, use_lym=args.use_lym
    )
    patients_before_formatting = int(data["id"].nunique())
    patients_after_formatting = int(rows["id"].nunique())
    patients_removed_during_formatting = (
        patients_before_formatting - patients_after_formatting
    )
    print(
        "\nPacientes elegíveis para a análise após a formatação temporal:\n"
        f"  Antes de prepare_measurement_features: {patients_before_formatting}\n"
        f"  Depois de prepare_measurement_features: {patients_after_formatting}\n"
        f"  Removidos por histórico temporal insuficiente: "
        f"{patients_removed_during_formatting}"
    )
    metadata.update(
        patients_before=patients_before_formatting,
        patients_after_formatting=patients_after_formatting,
        patients_removed_during_formatting=patients_removed_during_formatting,
    )
    write_json(output / "metadata.json", metadata)
    train, test = next(
        GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        ).split(X, y, rows["id"])
    )
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    groups = rows.iloc[train]["id"]
    if groups.nunique() < args.folds:
        raise ValueError(
            f"Folds={args.folds}, mas só há {groups.nunique()} pacientes de treino após formatação/teste"
        )
    splits = rows.copy()
    splits["split"] = "test"
    splits.loc[train, "split"] = "train"
    splits["validation_fold"] = -1
    for fold, (_, validation) in enumerate(
        GroupKFold(args.folds).split(X_train, y_train, groups)
    ):
        splits.loc[X_train.iloc[validation].index, "validation_fold"] = fold
    splits.to_csv(output / "splits.csv", index_label="row")
    test_patient_ids = rows.iloc[test]["id"].unique()
    target_series_columns = ["id", "time", "lym"]
    target_flag = f"{INTERPOLATION_FLAG_PREFIX}lym"
    if target_flag in data.columns:
        target_series_columns.append(target_flag)
    test_target_series = data.loc[
        data["id"].isin(test_patient_ids) & data["lym"].notna(),
        target_series_columns,
    ].sort_values(["id", "time"], kind="stable")
    test_target_series["interpolated"] = (
        test_target_series.pop(target_flag).fillna(0).astype(bool)
        if target_flag in test_target_series
        else False
    )

    test_target_series.to_csv(output / "test_target_series.csv", index=False)

    pd.concat([X, y.rename("lym")], axis=1).describe().to_csv(
        output / "input_summary.csv"
    )
    metadata.update(
        input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        rows_before=len(data),
        target_measurements_before_formatting=int(real_target_rows.sum()),
        rows_after_formatting=len(X),
        target_measurements_removed_without_history=int(
            real_target_rows.sum() - len(X)
        ),
        missing_before=int(data.isna().sum().sum()),
        missing_features=int(X.isna().sum().sum()),
        train_rows=len(train),
        test_rows=len(test),
        train_patients=int(groups.nunique()),
        test_patients=int(rows.iloc[test].id.nunique()),
        features=X.columns.tolist(),
        target="lym",
        feature_format={
            "mode": "measurement_history",
            "variable_mode": args.var_mode,
            "description": (
                "Uma linha por medida real de LYM; estáticas e tempo atual, mais "
                "valor inicial, último valor, tempo desde o último valor e "
                "tendência das séries temporais. Valores interpolados podem compor "
                "o histórico dos atributos, mas nunca o alvo. O histórico de LYM "
                f"{'é' if args.use_lym else 'não é'} usado como entrada."
            ),
            "feature_count": len(X.columns),
            "uses_target_history": args.use_lym,
            "target_excluded_from_history": not args.use_lym,
            "target_history_scope": (
                "strictly_previous_measurements" if args.use_lym else "excluded"
            ),
            "interpolated_values_as_predictor_history": True,
            "interpolated_target_rows_excluded": True,
        },
        protocol="GroupShuffleSplit para teste; GroupKFold apenas no treino; refit no treino completo",
        normalization={
            "mode": "all",
            "description": "MinMax em todos os atributos e no alvo LYM. Os transformadores são ajustados somente no treino de cada fold.",
            "target_scaled": True,
            "predictions_inverse_transformed": True,
            "identifiers_scaled": False,
            "metadata": "ID e tempo usados nos relatórios permanecem na escala original.",
        },
        preprocessing_note="Atributos históricos usam somente observações no mesmo paciente com time < ao time do alvo. Medidas de LYM com qualquer feature temporal ausente são excluídas. ID é usado apenas para agrupamento. Todas as features restantes são normalizadas somente com dados de treino em cada fold.",
        search_scoring="r2",
        metrics_weighting="Por observação, não por paciente",
        metrics_dispersion=(
            "Desvio padrão amostral (ddof=1) de RMSE, MAE e MAPE calculados "
            "separadamente por paciente no conjunto de teste"
        ),
        models={},
    )

    prep_metadata = args.data.parent / "data_preparation_metadata.json"
    if prep_metadata.exists():
        preparation = json.loads(prep_metadata.read_text(encoding="utf-8"))
        selection = preparation.get("selecao_occ") or {}
        interpolation = preparation.get("interpolacao") or {}
        metadata["data_preparation"] = {
            "occasions": selection.get("ocasioes_mantidas", []),
            "interpolation_applied": bool(interpolation.get("aplicada", False)),
        }
        if interpolation.get("aplicada"):
            metadata["data_preparation"]["interpolation_methods"] = interpolation.get(
                "metodos_por_serie", {}
            )
        shutil.copy2(prep_metadata, output / "data_preparation_metadata.json")
    write_json(output / "metadata.json", metadata)
    print(
        f"Experimento: {output}\nTreino: {len(train)} linhas; teste: {len(test)} linhas"
    )
    results = []
    specs = model_specs(args.seed, args.quick)
    for name in args.models:
        model, params = specs[name]
        folder = output / name
        folder.mkdir()
        candidates = len(ParameterGrid(params))
        count = candidates if args.search == "grid" else min(candidates, args.n_iter)
        print(f"\n{name}: {count} candidatos x {args.folds} folds + refit")
        start = time.perf_counter()
        search = make_search(
            model,
            params,
            args.search,
            args.folds,
            min(args.n_iter, candidates),
            args.seed,
            args.n_jobs,
            normalization=args.normalization,
        )
        search.fit(X_train, y_train, groups=groups)
        fit_seconds = time.perf_counter() - start
        prediction = search.predict(X_test)
        metrics = regression_metrics(
            y_test, prediction, groups=rows.iloc[test]["id"].to_numpy()
        )
        metrics.update(
            model=name, cv_r2=float(search.best_score_), fit_seconds=fit_seconds
        )
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "best_params.json", search.best_params_)
        pd.DataFrame(search.cv_results_).to_csv(folder / "cv_results.csv", index=False)
        predictions = rows.iloc[test].copy()
        predictions["observed"] = y_test
        predictions["predicted"] = prediction
        predictions.to_csv(folder / "predictions.csv", index_label="row")
        joblib.dump(search.best_estimator_, folder / "model.joblib")
        prediction_plot(
            y_test,
            prediction,
            folder / "predictions.png",
            interpolated=(
                rows.iloc[test]["interpolated"]
                if "interpolated" in rows.columns
                else None
            ),
        )
        print(
            f'  R² CV: {search.best_score_:.4f}; RMSE teste: {metrics["rmse"]:.4f}; calculando SHAP...'
        )
        shap_metadata = shap_report(
            search.best_estimator_,
            name,
            X_train,
            X_test,
            rows.iloc[test],
            folder,
            args.shap_samples,
            args.shap_background,
            args.seed,
            args.shap_permutations,
        )
        metadata["models"][name] = dict(
            parameters=params,
            candidates=count,
            best_params=search.best_params_,
            metrics=metrics,
            shap=shap_metadata,
        )
        write_json(output / "metadata.json", metadata)
        results.append(metrics)
        pd.DataFrame(results).to_csv(output / "results.csv", index=False)

    summarize(results, output)

    print("\nGerando análise temporal por paciente...")

    patient_analysis = analyze_patient_timeseries(output, require_complete=False)

    metadata["patient_temporal_analysis"] = {
        "status": "complete",
        "criterion": "RMSE por paciente no conjunto de teste",
        "directory": str(patient_analysis.relative_to(output)),
    }

    write_json(output / "metadata.json", metadata)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (
        args.output_dir.resolve()
        / (
            f"{stamp}_{args.search}_{args.normalization}_vars-{args.var_mode}_"
            f"lym-history-{'on' if args.use_lym else 'off'}_"
            f"measurement-history_folds{args.folds}_seed{args.seed}"
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "started_at_utc": stamp,
        "config": {
            k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "python": sys.version,
        "python_executable": sys.executable,
        "command": sys.argv,
        "versions": {},
    }
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "pyarrow",
        "joblib",
    ]:
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][package] = None
    sources = [
        "methods/ml_models.py",
        "utils/ml_reports.py",
        "utils/analyze_patient_timeseries.py",
        "runners/run_ml_experiment.py",
        "shells/run_ml_experiments.sh",
        "data_preparation.py",
        "transplant_ml.yml",
    ]
    metadata["source_sha256"] = {}
    for name in sources:
        source = BASE / name
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        metadata["source_sha256"][name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    write_json(output / "metadata.json", metadata)

    start = time.perf_counter()

    with (output / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(
            Tee(sys.stderr, log)
        ):
            try:
                execute(args, output, metadata)
                metadata["status"] = "complete"
            except BaseException as error:
                metadata.update(
                    status="failed", error=f"{type(error).__name__}: {error}"
                )
                traceback.print_exc()
                raise
            finally:
                metadata["elapsed_seconds"] = time.perf_counter() - start
                metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(output / "metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
