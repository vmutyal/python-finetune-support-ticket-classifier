import json

import pandas as pd
import pytest

from support_ticket_classifier import LABELS
from support_ticket_classifier.data import load_tickets, split_tickets


def _write_dataset(path, per_label=10):
    rows = [
        {"text": f"Example {label} ticket number {index}", "label": label}
        for label in LABELS
        for index in range(per_label)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return rows


def test_load_and_stratified_split(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_dataset(path)

    frame = load_tickets(path)
    train, validation, test = split_tickets(frame, seed=19)

    assert len(train) + len(validation) + len(test) == len(frame)
    assert len(train) == 21
    assert len(validation) == 5
    assert len(test) == 4
    for partition in (train, validation, test):
        assert set(partition["label"]) == set(LABELS)

    train, validation, test = split_tickets(frame, seed=19)
    assert train.equals(split_tickets(frame, seed=19)[0])


def test_rejects_unknown_label(tmp_path):
    path = tmp_path / "tickets.jsonl"
    path.write_text('{"text":"Hello","label":"Other"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported label"):
        load_tickets(path)


def test_split_requires_minimum_examples_per_label(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_dataset(path, per_label=2)

    with pytest.raises(ValueError, match="At least three examples"):
        split_tickets(load_tickets(path))


def test_group_split_keeps_groups_disjoint():
    frame = pd.DataFrame(
        [
            {
                "text": f"Ticket {group_id} {label}",
                "label": label,
                "customer_id": f"customer-{group_id}",
            }
            for group_id in range(12)
            for label in LABELS
        ]
    )

    train, validation, test = split_tickets(
        frame, seed=23, group_column="customer_id"
    )

    train_groups = set(train["customer_id"])
    validation_groups = set(validation["customer_id"])
    test_groups = set(test["customer_id"])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    for partition in (train, validation, test):
        assert set(partition["label"]) == set(LABELS)
