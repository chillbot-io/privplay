#!/usr/bin/env python3
"""
Train fresh PII-BERT from RoBERTa-base on curated AI4Privacy data.

Usage:
    # First run curation:
    python curate_ai4privacy.py --max-samples 50000

    # Then train:
    python train_pii_bert.py --epochs 3 --batch-size 16
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import argparse

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# Label normalization - map AI4Privacy types to our standard
LABEL_MAP = {
    # Email, Phone
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONENUMBER": "PHONE",
    "FAX": "FAX",
    
    # SSN, ID numbers
    "SSN": "SSN",
    "SOCIALNUM": "SSN",
    "US_SSN": "SSN",
    
    # Financial
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "IBAN": "IBAN",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "BIC": "ACCOUNT_NUMBER",
    "SWIFT": "ACCOUNT_NUMBER",
    
    # Names
    "NAME": "NAME_PERSON",
    "PERSON": "NAME_PERSON",
    "NAME_PERSON": "NAME_PERSON",
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON",
    "GIVENNAME": "NAME_PERSON",
    "MIDDLENAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "SUFFIX": "NAME_PERSON",
    
    # Location
    "ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "COUNTRY": "LOCATION",
    "ZIPCODE": "ZIP",
    "ZIP": "ZIP",
    "POSTCODE": "ZIP",
    
    # Online
    "USERNAME": "USERNAME",
    "USER_ID": "USERNAME",
    "PASSWORD": "PASSWORD",
    "PIN": "PASSWORD",
    "URL": "URL",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "MAC": "MAC_ADDRESS",
    
    # Dates
    "DATE": "DATE",
    "DOB": "DATE",
    "DATE_TIME": "DATE",
    "TIME": "DATE",
    
    # Other
    "COMPANY": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "COMPANYNAME": "ORGANIZATION",
    "IMEI": "DEVICE_ID",
    "VEHICLEVIN": "DEVICE_ID",
    "VIN": "DEVICE_ID",
}


def normalize_label(label: str) -> str:
    """Normalize label to standard form."""
    label_clean = label.upper().replace(" ", "").replace("_", "")
    
    # Try direct lookup
    if label_clean in LABEL_MAP:
        return LABEL_MAP[label_clean]
    if label.upper() in LABEL_MAP:
        return LABEL_MAP[label.upper()]
    
    return label.upper()


def load_curated_data(path: str = "ai4privacy_curated.json") -> List[Dict]:
    """Load curated dataset."""
    console.print(f"Loading curated data from {path}...")
    
    with open(path) as f:
        data = json.load(f)
    
    samples = data.get("samples", data)
    console.print(f"Loaded {len(samples)} samples")
    
    return samples


def prepare_ner_dataset(
    samples: List[Dict],
    tokenizer,
    max_length: int = 256,
    label2id: Dict[str, int] = None,
):
    """
    Convert samples to NER training format with BIO tagging.
    """
    from datasets import Dataset
    
    # Build label vocabulary
    if label2id is None:
        all_labels = {"O"}
        for sample in samples:
            for entity in sample.get("entities", []):
                label = normalize_label(entity.get("label", ""))
                all_labels.add(f"B-{label}")
                all_labels.add(f"I-{label}")
        
        # Sort for consistency
        sorted_labels = sorted(all_labels)
        label2id = {label: i for i, label in enumerate(sorted_labels)}
    
    id2label = {v: k for k, v in label2id.items()}
    
    console.print(f"Label vocabulary: {len(label2id)} labels")
    
    # Process samples
    processed = []
    skipped = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Tokenizing...", total=len(samples))
        
        for sample in samples:
            text = sample.get("text", "")
            entities = sample.get("entities", [])
            
            if not text or not entities:
                skipped += 1
                progress.advance(task)
                continue
            
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                padding="max_length",
            )
            
            # Build character-level labels
            char_labels = ["O"] * len(text)
            
            for entity in entities:
                start = entity.get("start")
                end = entity.get("end")
                label = normalize_label(entity.get("label", ""))
                
                if start is None or end is None:
                    continue
                if start < 0 or end > len(text):
                    continue
                
                # Apply BIO tags
                for i in range(start, min(end, len(text))):
                    if i == start:
                        char_labels[i] = f"B-{label}"
                    else:
                        char_labels[i] = f"I-{label}"
            
            # Align labels with tokens
            offset_mapping = encoding["offset_mapping"]
            token_labels = []
            
            for offset in offset_mapping:
                start, end = offset
                
                if start == end:  # Special token
                    token_labels.append(-100)  # Ignored in loss
                elif start < len(char_labels):
                    # Use label of first character
                    label_str = char_labels[start]
                    token_labels.append(label2id.get(label_str, label2id["O"]))
                else:
                    token_labels.append(label2id["O"])
            
            processed.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": token_labels,
            })
            
            progress.advance(task)
    
    console.print(f"Processed: {len(processed)} | Skipped: {skipped}")
    
    dataset = Dataset.from_list(processed)
    
    return dataset, label2id, id2label


def train_ner_model(
    base_model: str = "roberta-base",
    curated_path: str = "ai4privacy_curated.json",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    eval_split: float = 0.1,
):
    """
    Train NER model on curated data.
    """
    console.print(f"\n[bold cyan]═══ Training PII-BERT ═══[/bold cyan]\n")
    console.print(f"Base model: {base_model}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")
    
    if output_dir is None:
        output_dir = Path.home() / ".privplay" / "models" / "pii_bert_v2"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    console.print(f"\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load and prepare data
    samples = load_curated_data(curated_path)
    
    console.print(f"\nPreparing dataset...")
    dataset, label2id, id2label = prepare_ner_dataset(
        samples, tokenizer, max_length=max_length
    )
    
    # Split train/eval
    dataset = dataset.train_test_split(test_size=eval_split, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    console.print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    # Load model
    console.print(f"\nLoading base model...")
    from transformers import AutoModelForTokenClassification
    
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    
    console.print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training arguments
    from transformers import TrainingArguments, Trainer
    from transformers import DataCollatorForTokenClassification
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        warmup_ratio=0.1,
        fp16=False,  # Set True if GPU supports it
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Metrics
    import evaluate
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_preds = []
            true_labs = []
            
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_preds.append(id2label[pred])
                    true_labs.append(id2label[lab])
            
            true_predictions.append(true_preds)
            true_labels.append(true_labs)
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    console.print(f"\n[bold]Starting training...[/bold]\n")
    trainer.train()
    
    # Evaluate
    console.print(f"\n[bold]Final evaluation...[/bold]")
    results = trainer.evaluate()
    
    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Precision: {results['eval_precision']:.1%}")
    console.print(f"  Recall:    {results['eval_recall']:.1%}")
    console.print(f"  F1:        {results['eval_f1']:.1%}")
    
    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    # Save label mappings
    with open(final_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    
    console.print(f"\n[green]✓ Model saved to {final_dir}[/green]")
    
    return trainer, results


def test_model(model_path: str):
    """Quick test of trained model."""
    console.print(f"\n[bold cyan]═══ Testing Model ═══[/bold cyan]\n")
    
    from transformers import pipeline
    
    ner = pipeline("ner", model=model_path, aggregation_strategy="simple")
    
    test_texts = [
        "Contact John Smith at john.smith@email.com",
        "SSN: 123-45-6789, Credit Card: 4111111111111111",
        "Username: darrick_kunze, Password: Abc123!@#",
        "Located at 123 Main Street, New York, NY 10001",
    ]
    
    for text in test_texts:
        console.print(f"\n[bold]Text:[/bold] {text}")
        entities = ner(text)
        for ent in entities:
            console.print(f"  {ent['entity_group']:15} | {ent['word']:25} | {ent['score']:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PII-BERT NER model")
    parser.add_argument("--base-model", type=str, default="roberta-base",
                       help="Base model (roberta-base, bert-base-cased)")
    parser.add_argument("--curated", type=str, default="ai4privacy_curated.json",
                       help="Path to curated data")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Max sequence length")
    parser.add_argument("--test", action="store_true",
                       help="Test model after training")
    
    args = parser.parse_args()
    
    trainer, results = train_ner_model(
        base_model=args.base_model,
        curated_path=args.curated,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
    
    if args.test:
        model_path = args.output or str(Path.home() / ".privplay" / "models" / "pii_bert_v2" / "final")
        test_model(model_path)
