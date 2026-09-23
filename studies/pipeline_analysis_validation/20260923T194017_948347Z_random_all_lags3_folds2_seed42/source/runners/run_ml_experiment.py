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
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from methods.ml_models import prepare_features, model_specs, make_search
from utils.ml_reports import (
    write_json,
    regression_metrics,
    prediction_plot,
    shap_report,
    summarize,
)
from utils.analyze_patient_timeseries import analyze as analyze_patient_timeseries

BASE = Path(__file__).resolve().parents[1]


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
        "--data", type=Path, default=BASE / "data/clean_processed.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=BASE / "studies/ml")
    parser.add_argument("--search", choices=["grid", "random"], default="random")
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--normalization", choices=["all", "time-series", "none"], default="all"
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
    args = parser.parse_args(argv)
    if (
        args.lags < 0
        or args.folds < 2
        or args.n_iter < 1
        or args.n_jobs == 0
        or args.seed < 0
    ):
        parser.error("Exige lags >= 0, folds >= 2, n-iter >= 1, n-jobs != 0, seed >= 0")
    if (
        not 0 < args.test_size < 1
        or min(args.shap_samples, args.shap_background, args.shap_permutations) < 1
    ):
        parser.error("Exige 0 < test-size < 1 e parâmetros SHAP positivos")
    args.models = list(dict.fromkeys(args.models))
    return args


def execute(args, output, metadata):
    import shap  # Falha antes de treinar se a dependência estiver ausente.

    data = pd.read_parquet(args.data)
    X, y, rows = prepare_features(data, args.lags, args.normalization)
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
            f"Folds={args.folds}, mas só há {groups.nunique()} pacientes de treino após lags/teste"
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
    data.describe().to_csv(output / "input_summary.csv")
    metadata.update(
        input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        rows_before=len(data),
        rows_after_lags=len(X),
        patients_before=int(data.id.nunique()),
        missing_before=int(data.isna().sum().sum()),
        missing_features=int(X.isna().sum().sum()),
        patients_after_lags=int(rows.id.nunique()),
        train_rows=len(train),
        test_rows=len(test),
        train_patients=int(groups.nunique()),
        test_patients=int(rows.iloc[test].id.nunique()),
        features=X.columns.tolist(),
        target="lym",
        protocol="GroupShuffleSplit para teste; GroupKFold apenas no treino; refit no treino completo",
        normalization={
            "mode": args.normalization,
            "description": {
                "all": "MinMax em todos os atributos, lags e no alvo LYM, ajustado apenas no treino de cada fold; os scalers finais são ajustados no treino completo e salvos no Pipeline.",
                "time-series": "MinMax das quatro variáveis temporais por paciente, usando sua série completa (antigo patient).",
                "none": "Sem normalização adicional.",
            }[args.normalization],
            "target_scaled": args.normalization == "all",
            "predictions_inverse_transformed": args.normalization == "all",
            "identifiers_scaled": False,
        },
        preprocessing_note="t0 é o primeiro valor; features correntes são mantidas. A preparação anterior pode usar dados futuros via interpolação/preenchimento. Estudo retrospectivo.",
        metrics_weighting="Por observação, não por paciente",
        models={},
    )
    prep_metadata = args.data.parent / "data_preparation_metadata.json"
    if prep_metadata.exists():
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
        metrics = regression_metrics(y_test, prediction)
        metrics.update(
            model=name, cv_rmse=float(-search.best_score_), fit_seconds=fit_seconds
        )
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "best_params.json", search.best_params_)
        pd.DataFrame(search.cv_results_).to_csv(folder / "cv_results.csv", index=False)
        predictions = rows.iloc[test].copy()
        predictions["observed"] = y_test
        predictions["predicted"] = prediction
        predictions.to_csv(folder / "predictions.csv", index_label="row")
        joblib.dump(search.best_estimator_, folder / "model.joblib")
        prediction_plot(y_test, prediction, folder / "predictions.png")
        print(
            f'  RMSE CV: {-search.best_score_:.4f}; RMSE teste: {metrics["rmse"]:.4f}; calculando SHAP...'
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
        / f"{stamp}_{args.search}_{args.normalization}_lags{args.lags}_folds{args.folds}_seed{args.seed}"
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
