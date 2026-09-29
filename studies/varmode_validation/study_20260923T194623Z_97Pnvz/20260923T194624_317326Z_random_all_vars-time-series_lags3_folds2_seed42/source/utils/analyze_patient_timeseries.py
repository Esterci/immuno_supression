"""Gera análise temporal dos melhores e piores pacientes de um experimento."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def patient_metrics(predictions):
    """Calcula métricas por paciente, ponderando igualmente suas observações."""
    records = []
    for patient_id, patient in predictions.groupby("id", sort=True):
        error = patient["predicted"] - patient["observed"]
        records.append(
            {
                "id": patient_id,
                "observations": len(patient),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "time_min": float(patient["time"].min()),
                "time_max": float(patient["time"].max()),
            }
        )
    return pd.DataFrame(records).sort_values(["rmse", "id"]).reset_index(drop=True)


def temporal_plot(predictions, metrics, model, output):
    best = metrics.iloc[0]
    worst = metrics.iloc[-1]
    selected = [("Melhor", best), ("Pior", worst)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, (label, result) in zip(axes, selected):
        patient = predictions[predictions["id"] == result["id"]].sort_values("time")
        axis.plot(patient["time"], patient["observed"], "o-", label="LYM real", linewidth=2)
        axis.plot(patient["time"], patient["predicted"], "s--", label="LYM previsto", linewidth=2)
        axis.set_title(
            f"{label} paciente — ID {result['id']:g} | "
            f"RMSE={result['rmse']:.3f} | MAE={result['mae']:.3f} | n={int(result['observations'])}"
        )
        axis.set_xlabel("Tempo")
        axis.set_ylabel("LYM")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"Evolução temporal — {model.replace('_', ' ').title()}", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return best, worst


def analyze(experiment, require_complete=True):
    experiment = experiment.resolve()
    metadata = json.loads((experiment / "metadata.json").read_text(encoding="utf-8"))
    if require_complete and metadata.get("status") != "complete":
        raise ValueError(f"Experimento não está completo: {metadata.get('status')}")
    output = experiment / "patient_temporal_analysis"
    output.mkdir(exist_ok=True)
    summary = []
    for model in metadata["models"]:
        predictions = pd.read_csv(experiment / model / "predictions.csv")
        required = {"id", "time", "observed", "predicted"}
        if missing := required - set(predictions.columns):
            raise ValueError(f"{model}: colunas ausentes em predictions.csv: {sorted(missing)}")
        metrics = patient_metrics(predictions)
        metrics.to_csv(output / f"{model}_metrics_by_patient.csv", index=False)
        best, worst = temporal_plot(
            predictions,
            metrics,
            model,
            output / f"{model}_best_worst_patients.png",
        )
        for classification, result in (("best", best), ("worst", worst)):
            summary.append(
                {
                    "model": model,
                    "classification": classification,
                    **result.to_dict(),
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "best_worst_patients.csv", index=False)
    analysis_metadata = {
        "experiment": str(experiment),
        "criterion": "RMSE por paciente no conjunto de teste",
        "best_definition": "menor RMSE",
        "worst_definition": "maior RMSE",
        "models": list(metadata["models"]),
        "output": str(output),
    }
    (output / "metadata.json").write_text(
        json.dumps(analysis_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nAnálise salva em: {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args(argv)
    analyze(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
