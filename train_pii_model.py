"""
Train a fresh BERT model on AI4Privacy for PII detection.
This creates a dedicated PII detector, separate from Stanford's PHI model.
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "bert-base-uncased"  # Start fresh
OUTPUT_DIR = Path.home() / ".privplay" / "models" / "pii_bert"
MAX_SAMPLES = 10000
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# =============================================================================
# Load and Prepare Dataset
# =============================================================================

def load_ai4privacy_dataset(max_samples=10000):
    """Load AI4Privacy dataset and extract label set."""
    logger.info("Loading AI4Privacy dataset...")
    
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    
    logger.info(f"Loaded {len(ds)} samples")
    
    # Extract all unique labels
    all_labels = set()
    for item in ds:
        for label in item.get("privacy_mask", []):
            if isinstance(label, dict):
                all_labels.add(label.get("label", "O"))
            elif isinstance(label, str):
                all_labels.add(label)
    
    # Build label list (O first, then sorted)
    label_list = ["O"] + sorted([l for l in all_labels if l != "O"])
    
    logger.info(f"Found {len(label_list)} labels: {label_list[:10]}...")
    
    return ds, label_list


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=512):
    """Tokenize text and align labels to tokens."""
    
    tokenized = tokenizer(
        examples["source_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    
    all_labels = []
    
    for i, offset_mapping in enumerate(tokenized["offset_mapping"]):
        labels = [-100] * len(offset_mapping)  # -100 = ignore in loss
        
        privacy_mask = examples["privacy_mask"][i]
        if not privacy_mask:
            all_labels.append(labels)
            continue
        
        # Build character-level label map
        text = examples["source_text"][i]
        char_labels = ["O"] * len(text)
        
        for mask in privacy_mask:
            if isinstance(mask, dict):
                start = mask.get("start", 0)
                end = mask.get("end", 0)
                label = mask.get("label", "O")
                
                for j in range(start, min(end, len(text))):
                    if j == start:
                        char_labels[j] = f"B-{label}"
                    else:
                        char_labels[j] = f"I-{label}"
        
        # Map to tokens
        for j, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                labels[j] = -100
            elif start < len(char_labels):
                char_label = char_labels[start]
                if char_label in label2id:
                    labels[j] = label2id[char_label]
                elif char_label.startswith("B-") or char_label.startswith("I-"):
                    # Unknown label, map to O
                    labels[j] = label2id["O"]
                else:
                    labels[j] = label2id.get("O", 0)
            else:
                labels[j] = label2id["O"]
        
        all_labels.append(labels)
    
    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping")
    
    return tokenized


def compute_metrics(eval_pred, id2label):
    """Compute seqeval metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = []
    pred_labels = []
    
    for pred, label in zip(predictions, labels):
        true_seq = []
        pred_seq = []
        
        for p, l in zip(pred, label):
            if l != -100:
                true_seq.append(id2label[l])
                pred_seq.append(id2label[p])
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def main():
    logger.info("=" * 60)
    logger.info("Training Fresh PII Model on AI4Privacy")
    logger.info("=" * 60)
    
    # Load dataset
    ds, base_labels = load_ai4privacy_dataset(MAX_SAMPLES)
    
    # Build BIO label set
    label_list = ["O"]
    for label in base_labels:
        if label != "O":
            label_list.append(f"B-{label}")
            label_list.append(f"I-{label}")
    
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    logger.info(f"Label set: {len(label_list)} labels")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_ds = ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, MAX_LENGTH),
        batched=True,
        remove_columns=ds.column_names,
    )
    
    # Split train/eval
    split = tokenized_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # Load model
    logger.info(f"Loading model: {BASE_MODEL}")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",
        fp16=True,  # Use mixed precision on GPU
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = OUTPUT_DIR / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Save label map
    with open(final_path / "label_map.json", "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "base_labels": base_labels,
        }, f, indent=2)
    
    logger.info(f"Model saved to: {final_path}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {final_path}")
    print(f"F1: {metrics.get('eval_f1', 'N/A'):.1%}")
    print(f"Precision: {metrics.get('eval_precision', 'N/A'):.1%}")
    print(f"Recall: {metrics.get('eval_recall', 'N/A'):.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
