"""Shared classification metrics and JSON report helpers."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from support_ticket_classifier import LABELS


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], latency_ms: float
) -> dict:
    """Build a stable, JSON-serializable classification report."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_actual_columns_predicted": confusion_matrix(
                y_true, y_pred, labels=LABELS
            ).tolist(),
        },
        "mean_inference_latency_ms": float(latency_ms),
        "estimated_cost_per_1000_tickets_usd": 0.0,
        "cost_note": "Local inference estimate; excludes hardware, electricity, and setup.",
    }


def save_report(report: dict, path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
