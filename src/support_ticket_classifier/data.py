"""Dataset loading, validation, and reproducible train/validation/test splitting."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from support_ticket_classifier import LABELS


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Load and validate a JSONL dataset with ``text`` and ``label`` fields."""
    rows = []
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            text = record.get("text")
            label = record.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number} must have non-empty string 'text'.")
            if label not in LABELS:
                raise ValueError(
                    f"Line {line_number} has unsupported label {label!r}; "
                    f"expected one of {', '.join(LABELS)}."
                )
            rows.append({**record, "text": text.strip(), "label": label})

    if not rows:
        raise ValueError(f"No ticket records found in {path}.")
    return pd.DataFrame(rows)


def split_tickets(
    tickets: pd.DataFrame, seed: int = 42, group_column: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic 70/15/15 partitions, optionally keeping groups together."""
    missing = {"text", "label"} - set(tickets.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")
    counts = tickets["label"].value_counts()
    insufficient = [label for label in LABELS if counts.get(label, 0) < 3]
    if insufficient:
        raise ValueError(
            "At least three examples per label are required for a stratified "
            f"70/15/15 split; insufficient labels: {', '.join(insufficient)}."
        )

    if group_column is not None:
        if group_column not in tickets.columns:
            raise ValueError(f"Group column {group_column!r} was not found in the dataset.")
        if tickets[group_column].isna().any():
            raise ValueError(f"Group column {group_column!r} must not contain empty values.")
        first_split = GroupShuffleSplit(
            n_splits=1, test_size=0.30, random_state=seed
        )
        train_indices, remainder_indices = next(
            first_split.split(tickets, groups=tickets[group_column])
        )
        remainder = tickets.iloc[remainder_indices]
        second_split = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed
        )
        validation_relative, test_relative = next(
            second_split.split(remainder, groups=remainder[group_column])
        )
        partitions = (
            tickets.iloc[train_indices],
            remainder.iloc[validation_relative],
            remainder.iloc[test_relative],
        )
        for partition_name, partition in zip(
            ("training", "validation", "test"), partitions
        ):
            missing_labels = set(LABELS) - set(partition["label"])
            if missing_labels:
                raise ValueError(
                    f"Group-based split leaves {partition_name} without labels: "
                    f"{', '.join(sorted(missing_labels))}. Add more groups or adjust "
                    "the dataset before training."
                )
        return tuple(partition.reset_index(drop=True) for partition in partitions)

    rng = np.random.default_rng(seed)
    partitions: list[list[pd.DataFrame]] = [[], [], []]
    for label_index, label in enumerate(LABELS):
        class_rows = tickets[tickets["label"] == label]
        count = len(class_rows)
        quotas = [count * 0.70, count * 0.15, count * 0.15]
        sizes = [math.floor(quota) for quota in quotas]
        sizes[0] = max(sizes[0], 1)
        sizes[1] = max(sizes[1], 1)
        sizes[2] = max(sizes[2], 1)
        while sum(sizes) < count:
            candidates = [index for index in range(3)]
            if abs(quotas[1] - sizes[1]) == abs(quotas[2] - sizes[2]):
                candidates = [0, (1 if label_index % 2 == 0 else 2)]
            target = max(candidates, key=lambda index: quotas[index] - sizes[index])
            sizes[target] += 1
        while sum(sizes) > count:
            if sizes[0] > 1:
                target = 0
            else:
                target = max(range(3), key=lambda index: sizes[index] - quotas[index])
            if sizes[target] <= 1:
                raise ValueError(f"Could not create non-empty splits for label {label}.")
            sizes[target] -= 1

        shuffled = class_rows.iloc[rng.permutation(count)]
        train_end = sizes[0]
        validation_end = train_end + sizes[1]
        partitions[0].append(shuffled.iloc[:train_end])
        partitions[1].append(shuffled.iloc[train_end:validation_end])
        partitions[2].append(shuffled.iloc[validation_end:])

    return tuple(
        pd.concat(partition, ignore_index=True)
        for partition in partitions
    )
