"""Fine-tuning script for PHI/PII transformer model."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# Stanford model's original entity types
STANFORD_ENTITY_TYPES = ["AGE", "DATE", "ID", "NAME"]

# Extended entity types for PII
EXTENDED_ENTITY_TYPES = [
    "AGE", "DATE", "ID", "NAME",  # Original Stanford
    "SSN", "PHONE", "EMAIL", "ADDRESS", "ZIP",  # Contact/Location
    "CREDIT_CARD", "BANK_ACCOUNT", "ACCOUNT_NUMBER",  # Financial
    "MRN", "DATE_DOB", "FAX",  # Medical
    "IP_ADDRESS", "MAC_ADDRESS", "URL", "USERNAME", "DEVICE_ID",  # Digital
    "DRIVER_LICENSE", "PASSPORT",  # IDs
    "OTHER",  # Catch-all
]

# BIO label scheme
LABEL_TO_ID = {}
ID_TO_LABEL = {}


def build_label_maps(entity_types: List[str] = None):
    """Build label-to-id and id-to-label maps for BIO tagging."""
    global LABEL_TO_ID, ID_TO_LABEL
    
    if entity_types is None:
        entity_types = EXTENDED_ENTITY_TYPES
    
    LABEL_TO_ID = {"O": 0}
    idx = 1
    for etype in entity_types:
        LABEL_TO_ID[f"B-{etype}"] = idx
        idx += 1
        LABEL_TO_ID[f"I-{etype}"] = idx
        idx += 1
    
    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
    return LABEL_TO_ID, ID_TO_LABEL


@dataclass
class TrainingSample:
    """A single training sample with aligned tokens and labels."""
    text: str
    tokens: List[str]
    token_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class CorrectionDataset(Dataset):
    """Dataset from human corrections."""
    
    def __init__(self, samples: List[TrainingSample], max_length: int = 512):
        self.samples = samples
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Truncate/pad to max_length
        token_ids = sample.token_ids[:self.max_length]
        labels = sample.labels[:self.max_length]
        attention_mask = sample.attention_mask[:self.max_length]
        
        # Pad if needed
        pad_len = self.max_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
            labels = labels + [-100] * pad_len  # -100 is ignored in loss
            attention_mask = attention_mask + [0] * pad_len
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_corrections_from_db() -> List[Dict[str, Any]]:
    """Load corrections from SQLite database.
    
    Handles both positive (CONFIRMED/CHANGED) and negative (REJECTED) corrections.
    - CONFIRMED/CHANGED: Include entities with their types
    - REJECTED: Include as sample with empty entities (negative example)
    """
    from ..db import get_db
    
    db = get_db()
    corrections = db.get_corrections()
    
    # Group by document to reconstruct full text
    doc_corrections = {}
    for c in corrections:
        doc_id = c.document_id
        if doc_id not in doc_corrections:
            doc_corrections[doc_id] = []
        doc_corrections[doc_id].append(c)
    
    # Load document texts
    samples = []
    for doc_id, corrs in doc_corrections.items():
        doc = db.get_document(doc_id)
        if doc:
            entities = []
            has_rejected = False
            
            for c in corrs:
                if c.decision.value in ("confirmed", "changed"):
                    # Positive example - include entity
                    etype = c.correct_type if c.correct_type else c.detected_type
                    entities.append({
                        "text": c.entity_text,
                        "start": c.entity_start,
                        "end": c.entity_end,
                        "type": etype.value,
                    })
                elif c.decision.value == "rejected":
                    # Negative example - flag document
                    has_rejected = True
            
            # Include document if it has entities OR rejected corrections
            # Rejected-only documents become negative examples (all O labels)
            if entities or has_rejected:
                samples.append({
                    "text": doc.content,
                    "entities": entities,
                    "source": "corrections",
                })
    
    return samples


def load_corrections_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load corrections from exported JSON file.
    
    Handles both positive and negative corrections.
    """
    with open(path) as f:
        data = json.load(f)
    
    samples = []
    for c in data.get("corrections", []):
        # Reconstruct approximate text from context
        text = f"{c['context_before']}{c['entity_text']}{c['context_after']}"
        
        # Calculate entity position in reconstructed text
        start = len(c["context_before"])
        end = start + len(c["entity_text"])
        
        if c["decision"] in ("confirmed", "changed"):
            # Positive example
            etype = c["correct_type"] if c["correct_type"] else c["detected_type"]
            samples.append({
                "text": text,
                "entities": [{
                    "text": c["entity_text"],
                    "start": start,
                    "end": end,
                    "type": etype,
                }],
                "source": "json_export",
            })
        elif c["decision"] == "rejected":
            # Negative example - include text but NO entities
            # This teaches the model "this pattern is NOT an entity"
            samples.append({
                "text": text,
                "entities": [],  # Empty = all tokens labeled O
                "source": "json_export_negative",
            })
    
    return samples


def align_labels_to_tokens(
    text: str,
    entities: List[Dict],
    tokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Align entity span labels to wordpiece tokens.
    
    This is the tricky part - we need to map character offsets
    to token indices, handling subword tokenization.
    
    Returns:
        token_ids: List of token IDs
        labels: List of BIO label IDs aligned to tokens
        attention_mask: List of 1s
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    attention_mask = encoding["attention_mask"]
    
    # Initialize all labels to O
    labels = [LABEL_TO_ID["O"]] * len(token_ids)
    
    # Special tokens (CLS, SEP, PAD) get -100 (ignored in loss)
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            labels[i] = -100
    
    # Assign labels to tokens that overlap with entities
    for entity in entities:
        ent_start = entity["start"]
        ent_end = entity["end"]
        ent_type = entity["type"]
        
        # Check label exists
        b_label = f"B-{ent_type}"
        i_label = f"I-{ent_type}"
        
        if b_label not in LABEL_TO_ID:
            logger.warning(f"Unknown entity type: {ent_type}, skipping")
            continue
        
        first_token = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            # Skip special tokens
            if tok_start == 0 and tok_end == 0:
                continue
            
            # Check if token overlaps with entity
            if tok_start < ent_end and tok_end > ent_start:
                if first_token:
                    labels[i] = LABEL_TO_ID[b_label]
                    first_token = False
                else:
                    labels[i] = LABEL_TO_ID[i_label]
    
    return token_ids, labels, attention_mask


def prepare_training_data(
    corrections: List[Dict],
    tokenizer,
    entity_types: List[str] = None,
) -> List[TrainingSample]:
    """Prepare training samples from corrections.
    
    Now handles samples with empty entities (negative examples).
    """
    
    # Build label maps with extended types
    build_label_maps(entity_types)
    
    samples = []
    positive_count = 0
    negative_count = 0
    
    for corr in corrections:
        text = corr["text"]
        entities = corr["entities"]
        
        # REMOVED: if not entities: continue
        # We now keep samples with empty entities as negative examples
        
        try:
            token_ids, labels, attention_mask = align_labels_to_tokens(
                text, entities, tokenizer
            )
            
            samples.append(TrainingSample(
                text=text,
                tokens=tokenizer.convert_ids_to_tokens(token_ids),
                token_ids=token_ids,
                labels=labels,
                attention_mask=attention_mask,
            ))
            
            if entities:
                positive_count += 1
            else:
                negative_count += 1
                
        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            continue
    
    logger.info(f"Prepared {positive_count} positive samples, {negative_count} negative samples")
    
    return samples


def finetune(
    model_name: str = "StanfordAIMI/stanford-deidentifier-base",
    output_dir: Path = None,
    corrections_path: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 512,
    eval_split: float = 0.1,
    seed: int = 42,
    device: str = None,
):
    """
    Fine-tune the PHI detection model on corrections.
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Where to save fine-tuned model
        corrections_path: Path to exported corrections JSON (None = load from DB)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (2e-5 is typical for BERT fine-tuning)
        warmup_ratio: Portion of training for warmup
        weight_decay: Weight decay for regularization
        max_length: Maximum sequence length
        eval_split: Portion of data for validation
        seed: Random seed
        device: Device to train on (None = auto)
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup output directory
    if output_dir is None:
        from ..config import get_config
        output_dir = get_config().data_dir / "models" / "finetuned"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load corrections
    logger.info("Loading corrections...")
    if corrections_path:
        corrections = load_corrections_from_json(corrections_path)
    else:
        corrections = load_corrections_from_db()
    
    if len(corrections) < 10:
        raise ValueError(
            f"Not enough corrections ({len(corrections)}). "
            "Need at least 10 samples to fine-tune."
        )
    
    logger.info(f"Loaded {len(corrections)} correction samples")
    
    # Count positive vs negative
    positive = sum(1 for c in corrections if c["entities"])
    negative = len(corrections) - positive
    logger.info(f"  Positive (with entities): {positive}")
    logger.info(f"  Negative (rejected): {negative}")
    
    # Use extended entity types (not derived from corrections)
    # This ensures the model keeps all its original capabilities
    entity_types = EXTENDED_ENTITY_TYPES
    logger.info(f"Using {len(entity_types)} entity types")
    
    # Load tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Build label maps with extended types
    build_label_maps(entity_types)
    num_labels = len(LABEL_TO_ID)
    
    logger.info(f"Label map: {num_labels} labels (O + {len(entity_types)} entity types × 2 for B/I)")
    
    # Load model with extended label count
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # Allow different label count
    )
    
    # Prepare training data
    logger.info("Preparing training data...")
    samples = prepare_training_data(corrections, tokenizer, entity_types)
    
    if len(samples) < 10:
        raise ValueError(
            f"Only {len(samples)} valid samples after processing. "
            "Need at least 10."
        )
    
    # Split into train/eval
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - eval_split))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    logger.info(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")
    
    # Create datasets
    train_dataset = CorrectionDataset(train_samples, max_length)
    eval_dataset = CorrectionDataset(eval_samples, max_length) if eval_samples else None
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if eval_dataset else False,
        save_total_limit=2,
        seed=seed,
        report_to="none",  # Disable wandb etc
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Save label maps
    with open(final_path / "label_map.json", "w") as f:
        json.dump({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
            "entity_types": entity_types,
        }, f, indent=2)
    
    logger.info(f"Model saved to: {final_path}")
    
    return final_path


def evaluate_finetuned(
    model_path: Path,
    test_corrections: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Evaluate a fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model
        test_corrections: Test data (None = use held-out from DB)
        
    Returns:
        Dict with precision, recall, f1
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import numpy as np
    
    model_path = Path(model_path)
    
    # Load label map
    with open(model_path / "label_map.json") as f:
        label_data = json.load(f)
    
    global LABEL_TO_ID, ID_TO_LABEL
    LABEL_TO_ID = label_data["label_to_id"]
    ID_TO_LABEL = {int(k): v for k, v in label_data["id_to_label"].items()}
    entity_types = label_data["entity_types"]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Load test data
    if test_corrections is None:
        test_corrections = load_corrections_from_db()
    
    # Prepare samples
    samples = prepare_training_data(test_corrections, tokenizer, entity_types)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sample in samples:
            inputs = {
                "input_ids": torch.tensor([sample.token_ids], device=device),
                "attention_mask": torch.tensor([sample.attention_mask], device=device),
            }
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            labels = np.array(sample.labels)
            
            # Filter out ignored labels (-100)
            mask = labels != -100
            all_preds.extend(preds[mask].tolist())
            all_labels.extend(labels[mask].tolist())
    
    # Calculate metrics (entity-level)
    # Convert BIO to spans and compare
    pred_entities = bio_to_spans(all_preds, ID_TO_LABEL)
    true_entities = bio_to_spans(all_labels, ID_TO_LABEL)
    
    tp = len(pred_entities & true_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def bio_to_spans(labels: List[int], id_to_label: Dict[int, str]) -> set:
    """Convert BIO labels to set of (start, end, type) spans."""
    spans = set()
    current_type = None
    current_start = None
    
    for i, label_id in enumerate(labels):
        label = id_to_label.get(label_id, "O")
        
        if label.startswith("B-"):
            # Save previous span
            if current_type:
                spans.add((current_start, i, current_type))
            # Start new span
            current_type = label[2:]
            current_start = i
            
        elif label.startswith("I-"):
            # Continue span only if same type
            if current_type != label[2:]:
                if current_type:
                    spans.add((current_start, i, current_type))
                current_type = label[2:]
                current_start = i
                
        else:  # O
            if current_type:
                spans.add((current_start, i, current_type))
            current_type = None
            current_start = None
    
    # Don't forget last span
    if current_type:
        spans.add((current_start, len(labels), current_type))
    
    return spans
