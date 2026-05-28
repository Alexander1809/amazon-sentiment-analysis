from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_and_save(model, X_test, y_test, report_dir: str | Path = "reports") -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["negative", "positive"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    metrics_text = (
        "Metricas del modelo\n"
        "===================\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-score:  {metrics['f1_score']:.4f}\n\n"
        "Reporte de clasificacion\n"
        "========================\n"
        f"{metrics['classification_report']}\n"
        "Matriz de confusion [[TN, FP], [FN, TP]]\n"
        f"{metrics['confusion_matrix']}\n"
    )

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["negative", "positive"],
        values_format="d",
    )
    disp.ax_.set_title("Matriz de confusion - Amazon Reviews")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    return metrics
