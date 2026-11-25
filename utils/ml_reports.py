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


def regression_metrics(y, prediction, groups=None):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    nonzero = y != 0
    metrics = {
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
    # Repeated measurements from the same patient are not independent.  Report
    # dispersion across patient-level metrics, while retaining the pooled
    # (per-observation) estimates above for backwards compatibility.
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("groups must have the same length as y")

    by_group = []
    for group in pd.unique(groups):
        selected = groups == group
        group_y = y[selected]
        group_prediction = prediction[selected]
        group_nonzero = group_y != 0
        by_group.append(
            {
                "rmse": float(np.sqrt(mean_squared_error(group_y, group_prediction))),
                "mae": float(mean_absolute_error(group_y, group_prediction)),
                "mape_percent_nonzero": (
                    float(
                        100
                        * np.mean(
                            np.abs(
                                (group_y[group_nonzero]
                                - group_prediction[group_nonzero])
                                / group_y[group_nonzero]
                            )
                        )
                    )
                    if group_nonzero.any()
                    else np.nan
                ),
            }
        )
    patient_metrics = pd.DataFrame(by_group)
    for metric in ("rmse", "mae", "mape_percent_nonzero"):
        values = patient_metrics[metric].dropna()
        metrics[f"{metric}_patient_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else None
        )
    metrics["metric_patient_count"] = int(len(patient_metrics))
    return metrics


def prediction_plot(y, prediction, path, interpolated=None):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    if interpolated is None:
        interpolated = np.zeros(len(y), dtype=bool)
    interpolated = np.asarray(interpolated, dtype=bool)
    if len(interpolated) != len(y):
        raise ValueError("interpolated must have the same length as y")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = (
        (~interpolated, "Observado originalmente", "#2563eb", "o"),
        (interpolated, "Alvo interpolado", "#dc2626", "X"),
    )
    residual = y - prediction
    for selected, label, color, marker in styles:
        if not selected.any():
            continue
        axes[0].scatter(
            y[selected], prediction[selected], alpha=0.65, s=28,
            color=color, marker=marker, label=label,
        )
        axes[1].scatter(
            prediction[selected], residual[selected], alpha=0.65, s=28,
            color=color, marker=marker, label=label,
        )
    low, high = min(min(y), min(prediction)), max(max(y), max(prediction))
    axes[0].plot([low, high], [low, high], "--", color="black")
    axes[0].set(
        xlabel="LYM observado",
        ylabel="LYM previsto",
        title="Medidas previsíveis dos pacientes de teste",
    )
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="LYM previsto", ylabel="Resíduo")
    if interpolated.any():
        axes[0].legend()
        axes[1].legend()
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
        # Explica o Pipeline completo para manter as contribuições na escala
        # original de LYM após a transformação inversa do alvo.
        pipeline = model
    fallback_reason = None

    def permutation_explanation():
        # A interface predict funciona também para ensembles sem suporte TreeSHAP.
        def predict(values):
            return model.predict(pd.DataFrame(values, columns=X_train.columns))

        explainer = shap.PermutationExplainer(predict, reference, seed=seed)
        return explainer(explained, max_evals=permutations * (2 * X_train.shape[1] + 1))

    if scaled:
        model = pipeline
        explanation = permutation_explanation()
        algorithm = "permutation_pipeline"
    elif name == "adaboost":
        explanation = permutation_explanation()
        algorithm = "permutation"
    else:
        try:
            explainer = shap.TreeExplainer(
                model, data=reference, feature_perturbation="interventional"
            )
            explanation = explainer(explained)
            np.testing.assert_allclose(
                explanation.base_values + explanation.values.sum(axis=1),
                prediction,
                rtol=1e-4,
                atol=1e-5,
            )
            algorithm = "tree_interventional"
        except (AssertionError, shap.utils._exceptions.ExplainerError) as error:
            fallback_reason = f"{type(error).__name__}: {error}"
            print(
                "  TreeSHAP falhou na verificação de aditividade; usando PermutationExplainer."
            )
            explanation = permutation_explanation()
            algorithm = "permutation"
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
        "fallback_reason": fallback_reason,
        "feature_scale": "prepared_features",
        "target_scale": "original_lym",
        "samples": len(explained),
        "background_samples": len(reference),
        "background_source": "train",
        "explained_source": "test",
        "seed": seed,
        "permutations": permutations if algorithm.startswith("permutation") else None,
    }


def summarize(results, output):
    table = pd.DataFrame(results)
    table.to_csv(output / "results.csv", index=False)
    print("\nResultados no conjunto de teste:")
    print(table.to_string(index=False))
    comparisons = (
        ("rmse", "rmse_patient_sd", "RMSE", "%.3f", "RMSE de teste"),
        ("mae", "mae_patient_sd", "MAE", "%.3f", "MAE de teste"),
        (
            "mape_percent_nonzero",
            "mape_percent_nonzero_patient_sd",
            "MAPE",
            "%.2f%%",
            "Erro percentual absoluto médio (MAPE, %)",
        ),
    )
    for metric, sd_metric, label, value_format, y_label in comparisons:
        comparison = table.dropna(subset=[metric])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        errors = comparison[sd_metric].fillna(0) if sd_metric in comparison else None
        bars = ax.bar(
            comparison["model"],
            comparison[metric],
            yerr=errors,
            capsize=5,
        )
        ax.bar_label(bars, fmt=value_format, padding=3)
        ax.set(
            ylabel=y_label,
            title=(
                f"Comparação dos modelos por {label} — medidas previsíveis\n"
                "Barras de erro: ±1 DP entre pacientes"
            ),
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        filename = "mape" if metric == "mape_percent_nonzero" else metric
        fig.savefig(output / f"model_comparison_{filename}.png", dpi=160)
        plt.close(fig)
