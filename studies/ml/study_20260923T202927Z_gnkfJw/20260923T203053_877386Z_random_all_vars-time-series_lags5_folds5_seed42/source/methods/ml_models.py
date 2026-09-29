"""Features temporais, modelos e buscas extraídos de ml_models_study.ipynb."""

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector

TEMPORAL_COLUMNS = ["amt_tac", "dv_tac", "amt_map", "dv_map"]


def prepare_features(data, n_lags=3, normalization="all", var_mode="all"):
    """Cria histórico e tempo atual; tempo/LYM são escalados dentro da busca."""
    if n_lags < 0:
        raise ValueError("n_lags deve ser >= 0")
    if normalization not in ("all", "time-series", "none"):
        raise ValueError("normalization deve ser all, time-series ou none")
    if var_mode not in ("all", "time-series"):
        raise ValueError("var_mode deve ser all ou time-series")
    required = ["id", "time", "lym", *TEMPORAL_COLUMNS]
    if missing := set(required) - set(data.columns):
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if data[["id", "time"]].isna().any().any():
        raise ValueError("id e time não podem conter valores ausentes")
    data = data.sort_values(["id", "time"], kind="stable").copy()
    if normalization == "time-series":
        for col in TEMPORAL_COLUMNS:
            data[col] = data.groupby("id")[col].transform(
                lambda x: MinMaxScaler()
                .fit_transform(x.to_numpy().reshape(-1, 1))
                .ravel()
            )

    lag_columns = []
    for col in [*TEMPORAL_COLUMNS, "time", "lym"]:
        # Semântica original: t0 é o primeiro valor, não o valor corrente.
        data[f"{col}_t0"] = data.groupby("id")[col].transform(lambda values: values.iloc[0])
        lag_columns.append(f"{col}_t0")
        for lag in range(1, n_lags + 1):
            name = f"{col}_t{lag}"
            data[name] = data.groupby("id")[col].shift(lag)
            lag_columns.append(name)
    # O valor inicial só pode ser usado em observações posteriores.
    data = data.loc[data.groupby("id").cumcount() > 0]
    data = data.dropna(subset=lag_columns).reset_index(drop=True)
    if data.empty:
        raise ValueError("Nenhuma linha restante após criação dos lags")
    X = data.drop(columns=["id", "lym", *TEMPORAL_COLUMNS])
    if var_mode == "time-series":
        X = X[["time", *lag_columns]]
    y = data["lym"]
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y).all():
        raise ValueError(
            "Features/target contêm NaN ou infinito; verifique a preparação dos dados"
        )
    return X, y, data[["id", "time"]]


def model_specs(seed=42, quick=False):
    specs = {
        "random_forest": (
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {
                "n_estimators": [100, 200, 300, 500, 800],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5, 1.0],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=seed),
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.7, 0.8, 1.0],
            },
        ),
        "adaboost": (
            AdaBoostRegressor(random_state=seed),
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                "loss": ["linear", "square", "exponential"],
            },
        ),
    }
    if quick:
        specs = {
            name: (model, {"n_estimators": [5, 10]})
            for name, (model, _) in specs.items()
        }
    return specs


def make_search(model, params, search, folds, n_iter, seed, n_jobs, normalization="none"):
    if normalization not in ("all", "time-series", "none"):
        raise ValueError("normalization deve ser all, time-series ou none")
    if normalization == "all":
        model = TransformedTargetRegressor(
            regressor=model,
            transformer=MinMaxScaler(),
        )
        model = Pipeline([
            ("scaler", MinMaxScaler().set_output(transform="pandas")),
            ("model", model),
        ])
        params = {f"model__regressor__{key}": value for key, value in params.items()}
    else:
        # Tempo e histórico do alvo sempre são normalizados, somente no treino.
        scaler = ColumnTransformer(
            [("history", MinMaxScaler(), make_column_selector(
                pattern=r"^(?:time|time_t[0-9]+|lym_t[0-9]+)$"
            ))],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        model = Pipeline([("scaler", scaler), ("model", model)])
        params = {f"model__{key}": value for key, value in params.items()}
    common = dict(
        estimator=model,
        scoring="r2",
        cv=GroupKFold(n_splits=folds),
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=True,
        verbose=1,
    )
    if search == "grid":
        return GridSearchCV(param_grid=params, **common)
    if search == "random":
        return RandomizedSearchCV(
            param_distributions=params, n_iter=n_iter, random_state=seed, **common
        )
    raise ValueError("Busca deve ser grid ou random")
