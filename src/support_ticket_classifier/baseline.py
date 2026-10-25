"""Evaluate a pretrained zero-shot classifier as the prompt-based baseline."""

import argparse
import time
from pathlib import Path

from transformers import pipeline

from support_ticket_classifier import LABELS
from support_ticket_classifier.data import load_tickets, split_tickets
from support_ticket_classifier.metrics import classification_report, save_report


def evaluate_baseline(
    data_path: str,
    model_name: str,
    output_path: str,
    seed: int,
    estimated_cost_per_1k: float,
    group_column: str | None = None,
) -> dict:
    tickets = load_tickets(data_path)
    _, _, test = split_tickets(tickets, seed=seed, group_column=group_column)
    classifier = pipeline("zero-shot-classification", model=model_name)
    predictions = []
    started = time.perf_counter()
    for text in test["text"]:
        result = classifier(
            text,
            candidate_labels=list(LABELS),
            hypothesis_template="This customer support ticket is about {}.",
        )
        predictions.append(result["labels"][0])
    elapsed_ms = (time.perf_counter() - started) * 1000
    report = classification_report(
        test["label"].tolist(), predictions, elapsed_ms / len(test)
    )
    report["model"] = model_name
    report["split_seed"] = seed
    report["group_column"] = group_column
    report["estimated_cost_per_1000_tickets_usd"] = estimated_cost_per_1k
    report["cost_note"] = (
        "User-supplied estimate; local inference by default costs $0 in API fees. "
        "Hardware, electricity, and setup are excluded."
    )
    save_report(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_tickets.jsonl")
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--output", default="outputs/baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimated-cost-per-1k", type=float, default=0.0)
    parser.add_argument(
        "--group-column",
        help="Optional JSONL field to keep customer/template groups in one split.",
    )
    args = parser.parse_args()
    if args.estimated_cost_per_1k < 0:
        parser.error("--estimated-cost-per-1k must be non-negative.")
    report = evaluate_baseline(
        args.data,
        args.model,
        args.output,
        args.seed,
        args.estimated_cost_per_1k,
        args.group_column,
    )
    print(f"Baseline metrics saved to {Path(args.output)}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Mean inference latency: {report['mean_inference_latency_ms']:.2f} ms/ticket")


if __name__ == "__main__":
    main()
