# Fine-Tune a Support-Ticket Classifier

A small, reproducible Python project for classifying customer support tickets as **Billing**, **Technical**, or **Account**. It compares a pretrained zero-shot model with a fine-tuned open-weight model, evaluates on a held-out test set, and offers a local prediction CLI.

> `data/sample_tickets.jsonl` is a tiny learning/demo dataset. It is not large or representative enough for a reliable model. For a meaningful experiment, use a few hundred or more high-quality labeled tickets, review label consistency, and avoid putting near-duplicates, customers, or templates on both sides of a split.

## Project layout

```text
data/sample_tickets.jsonl
src/support_ticket_classifier/
  data.py       Dataset validation and deterministic stratified splitting
  metrics.py    Accuracy, per-class precision/recall, confusion matrix, latency
  baseline.py   Pretrained zero-shot baseline
  train.py      LoRA fine-tuning and held-out test evaluation
  predict.py    Prediction with a saved fine-tuned model
tests/
```

## Setup

Python 3.10+ is recommended. Install PyTorch appropriate for your hardware from [the PyTorch install selector](https://pytorch.org/get-started/locally/), then install this project's dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The core libraries are `torch`, `transformers`, `datasets`, `peft`, `accelerate`, and `scikit-learn`. The first baseline/training run downloads model weights from Hugging Face.

## Dataset format

Use one JSON object per line, with non-empty `text` and `label` fields. An optional group identifier can be included when related tickets must stay together:

```json
{"text": "I was charged twice for my subscription.", "label": "Billing", "customer_id": "customer-01"}
{"text": "The application crashes when I open it.", "label": "Technical", "customer_id": "customer-02"}
{"text": "I cannot reset my password.", "label": "Account", "customer_id": "customer-03"}
```

Only the three configured labels are accepted. The default splitter uses a deterministic 70/15/15 stratified split. It requires at least three records per label; pass `--seed` to reproduce or vary the split. If records from the same customer, template, or other source must remain together, use `--group-column customer_id`; the group-based split keeps groups intact and reports an error if it leaves a partition without one of the labels. Check near-duplicates and time leakage before training.

## 1. Run the pretrained baseline

The baseline uses a pretrained NLI model with a zero-shot classification prompt. This is a prompt-based reference, not a ticket-trained model:

```powershell
python -m support_ticket_classifier.baseline --data data/sample_tickets.jsonl
```

Metrics are written to `outputs/baseline_metrics.json` and include accuracy, per-class precision/recall/F1, a confusion matrix, mean per-ticket inference latency (model already loaded), and estimated inference cost per 1,000 tickets. Local open-weight inference has no API charge, so the default monetary estimate is `$0`; hardware, electricity, and setup costs are excluded. The tiny demo split is only to validate the pipeline; do not interpret its scores as meaningful.

Use `--model` to select another Hugging Face zero-shot classification model, `--group-column` to preserve groups across splits, and `--estimated-cost-per-1k` if you want to record a cost estimate for a hosted service.

## 2. Fine-tune with LoRA

```powershell
python -m support_ticket_classifier.train --data data/sample_tickets.jsonl --output-dir outputs/ticket-classifier
```

By default this fine-tunes `distilbert/distilbert-base-uncased` as a sequence classifier with LoRA adapters. Training uses the train partition, model selection uses validation macro-F1, and final metrics are computed once on the held-out test partition. The trained adapter, tokenizer, label mappings, and `test_metrics.json` are saved under the output directory.

For a real experiment, point `--data` at your larger curated dataset. Useful options include `--model`, `--seed`, `--epochs`, `--batch-size`, `--max-length`, and `--group-column`. Select a model and batch size that fit your hardware.

## Library roles

| Library | Role |
| --- | --- |
| PyTorch | Tensor operations and model execution/training |
| Transformers | Pretrained model, tokenizer, training, and inference interfaces |
| Datasets | Efficient tokenized datasets for the training loop |
| PEFT | LoRA adapters for parameter-efficient fine-tuning |
| Accelerate | Hardware-aware execution support used by Transformers Trainer |
| scikit-learn | Data splitting and classification metrics |

## 3. Predict

```powershell
python -m support_ticket_classifier.predict --model-dir outputs/ticket-classifier --text "My invoice has the wrong amount."
```

## Run tests

```powershell
python -m pytest
```

The unit tests run without downloading model weights.
