"""Métricas, prints, gráficos e explicações SHAP dos experimentos."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def regression_metrics(y, prediction):
    y = np.asarray(y)
    nonzero = y != 0
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else None,
        "mape_percent_nonzero": (
            float(
                100 * np.mean(np.abs((y[nonzero] - prediction[nonzero]) / y[nonzero]))
            )
            if nonzero.any()
            else None
        ),
        "mape_excluded_zeros": int((~nonzero).sum()),
    }


def prediction_plot(y, prediction, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, prediction, alpha=0.5, s=16)
    low, high = min(min(y), min(prediction)), max(max(y), max(prediction))
    axes[0].plot([low, high], [low, high], "--", color="black")
    axes[0].set(
        xlabel="LYM observado", ylabel="LYM previsto", title="Pacientes de teste"
    )
    axes[1].scatter(prediction, np.asarray(y) - prediction, alpha=0.5, s=16)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="LYM previsto", ylabel="Resíduo")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def shap_report(
    model,
    name,
    X_train,
    X_test,
    test_meta,
    output,
    samples,
    background,
    seed,
    permutations,
):
    import shap

    explained = X_test.sample(n=min(samples, len(X_test)), random_state=seed)
    reference = X_train.sample(n=min(background, len(X_train)), random_state=seed)
    prediction = model.predict(explained)
    scaled = isinstance(model, Pipeline)
    if scaled:
        # SHAP usa os mesmos atributos transformados recebidos pelo estimador.
        explained = model.named_steps["scaler"].transform(explained)
        reference = model.named_steps["scaler"].transform(reference)
        model = model.named_steps["model"]
    if name == "adaboost":
        # A interface predict funciona também para ensembles sem suporte TreeSHAP.
        def predict(values):
            return model.predict(pd.DataFrame(values, columns=X_train.columns))

        explainer = shap.PermutationExplainer(predict, reference, seed=seed)
        explanation = explainer(
            explained, max_evals=permutations * (2 * X_train.shape[1] + 1)
        )
        algorithm = "permutation"
    else:
        explainer = shap.TreeExplainer(
            model, data=reference, feature_perturbation="interventional"
        )
        explanation = explainer(explained)
        algorithm = "tree_interventional"
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        }
    )
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(
        output / "shap_importance.csv", index=False
    )
    pd.DataFrame(
        explanation.values, columns=X_train.columns, index=explained.index
    ).to_parquet(output / "shap_values.parquet")
    explained.to_parquet(output / "shap_features.parquet")
    rows = test_meta.loc[explained.index].copy()
    rows["base_value"] = explanation.base_values
    rows["prediction"] = prediction
    rows.to_parquet(output / "shap_samples.parquet")
    np.testing.assert_allclose(
        np.asarray(explanation.base_values) + explanation.values.sum(axis=1),
        rows["prediction"],
        rtol=1e-4,
        atol=1e-5,
    )
    for kind in ("bar", "beeswarm"):
        getattr(shap.plots, kind)(explanation, max_display=20, show=False)
        plt.gcf().savefig(output / f"shap_{kind}.png", dpi=160, bbox_inches="tight")
        plt.close("all")
    return {
        "algorithm": algorithm,
        "feature_scale": "minmax_train" if scaled else "prepared_features",
        "samples": len(explained),
        "background_samples": len(reference),
        "background_source": "train",
        "explained_source": "test",
        "seed": seed,
        "permutations": permutations if name == "adaboost" else None,
    }


def summarize(results, output):
    table = pd.DataFrame(results)
    table.to_csv(output / "results.csv", index=False)
    print("\nResultados no conjunto de teste:")
    print(table.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["model"], table["rmse"])
    ax.set(ylabel="RMSE de teste", title="Comparação dos modelos")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
