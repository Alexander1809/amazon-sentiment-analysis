from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, classification_report


def evaluate_and_save(model, X_test, y_test, report_dir="reports") -> dict:
    report_dir = Path(report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(y_test, y_pred, target_names=["negative", "positive"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["negative", "positive"], values_format="d")
    disp.ax_.set_title("Matriz de confusión - Amazon Reviews")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    return metrics
