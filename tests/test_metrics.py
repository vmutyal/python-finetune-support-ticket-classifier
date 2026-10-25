import json

from support_ticket_classifier.metrics import classification_report, save_report


def test_report_contains_requested_metrics_and_is_json_serializable(tmp_path):
    report = classification_report(
        ["Billing", "Technical", "Account"],
        ["Billing", "Account", "Account"],
        latency_ms=12.5,
    )
    report_path = tmp_path / "metrics.json"

    save_report(report, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved["accuracy"] == 2 / 3
    assert saved["per_class"]["Technical"]["recall"] == 0.0
    assert saved["confusion_matrix"]["rows_actual_columns_predicted"][1][2] == 1
    assert saved["mean_inference_latency_ms"] == 12.5
    assert saved["estimated_cost_per_1000_tickets_usd"] == 0.0
