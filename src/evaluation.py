"""Model evaluation: metrics and reports (reusable from training, tests, and experiments)."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Run predictions on a held-out set and return a metrics dict and text report.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    report = classification_report(y_test, y_pred)
    return metrics, report
