"""Predict a support-ticket label using a saved LoRA fine-tuned model."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text: str, model_dir: str) -> tuple[str, float]:
    if not text.strip():
        raise ValueError("Ticket text must not be empty.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    adapter_config = PeftConfig.from_pretrained(model_dir)
    mapping_path = Path(model_dir) / "label_mapping.json"
    with mapping_path.open(encoding="utf-8") as mapping_file:
        label_mapping = json.load(mapping_file)
    id_to_label = {int(index): label for index, label in label_mapping["id_to_label"].items()}
    label_to_id = label_mapping["label_to_id"]
    model = AutoModelForSequenceClassification.from_pretrained(
        adapter_config.base_model_name_or_path,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
    label_id = int(torch.argmax(probabilities).item())
    return str(id_to_label[label_id]), float(probabilities[label_id].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    label, confidence = predict(args.text, args.model_dir)
    print(f"{label} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    main()
