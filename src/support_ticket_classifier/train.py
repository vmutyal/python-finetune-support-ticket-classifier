"""Fine-tune a sequence classifier with LoRA and evaluate the held-out test split."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from support_ticket_classifier import LABELS
from support_ticket_classifier.data import load_tickets, split_tickets
from support_ticket_classifier.metrics import classification_report, save_report


def _dataset(frame, tokenizer, label_to_id: dict[str, int], max_length: int) -> Dataset:
    data = Dataset.from_dict(
        {
            "text": frame["text"].tolist(),
            "labels": [label_to_id[label] for label in frame["label"]],
        }
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return data.map(tokenize, batched=True, remove_columns=["text"])


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "distilbert/distilbert-base-uncased",
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    group_column: str | None = None,
) -> dict:
    if epochs < 1 or batch_size < 1 or max_length < 1:
        raise ValueError("epochs, batch_size, and max_length must be positive.")

    train_frame, validation_frame, test_frame = split_tickets(
        load_tickets(data_path), seed=seed, group_column=group_column
    )
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
        ),
    )
    tokenized_train = _dataset(train_frame, tokenizer, label_to_id, max_length)
    tokenized_validation = _dataset(validation_frame, tokenizer, label_to_id, max_length)
    tokenized_test = _dataset(test_frame, tokenizer, label_to_id, max_length)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    prediction_started = time.perf_counter()
    prediction = trainer.predict(tokenized_test)
    prediction_latency_ms = (
        (time.perf_counter() - prediction_started) * 1000 / len(test_frame)
    )
    y_pred = [id_to_label[int(index)] for index in np.argmax(prediction.predictions, axis=-1)]
    report = classification_report(
        test_frame["label"].tolist(), y_pred, latency_ms=prediction_latency_ms
    )
    report["model"] = model_name
    report["training_method"] = "LoRA"
    report["split_seed"] = seed
    report["group_column"] = group_column
    report["test_loss"] = float(prediction.metrics["test_loss"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)
    save_report(
        {"label_to_id": label_to_id, "id_to_label": id_to_label},
        output_path / "label_mapping.json",
    )
    save_report(report, output_path / "test_metrics.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_tickets.jsonl")
    parser.add_argument("--output-dir", default="outputs/ticket-classifier")
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--group-column",
        help="Optional JSONL field to keep customer/template groups in one split.",
    )
    args = parser.parse_args()
    report = train(
        args.data,
        args.output_dir,
        args.model,
        args.seed,
        args.epochs,
        args.batch_size,
        args.max_length,
        args.group_column,
    )
    print(f"Test metrics saved to {Path(args.output_dir) / 'test_metrics.json'}")
    print(f"Accuracy: {report['accuracy']:.3f}")


if __name__ == "__main__":
    main()
