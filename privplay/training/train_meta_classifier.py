"""Meta-classifier training pipeline.

Generates training data, captures signals from detectors, and trains the meta-classifier.

Usage:
    python -m privplay.training.train_meta_classifier --samples 5000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def generate_training_data(
    n_synthetic: int = 3000,
    n_adversarial: int = 1000,
    n_ai4privacy: int = 1000,
) -> Tuple[List, List, List]:
    """Generate all training data."""
    from .synthetic_generator import generate_synthetic_dataset, validate_dataset
    from .adversarial_cases import generate_adversarial_dataset, get_adversarial_stats
    
    console.print("\n[bold]Step 1: Generate Training Data[/bold]")
    console.print("─" * 50)
    
    # Synthetic clinical notes
    console.print(f"\nGenerating {n_synthetic} synthetic clinical notes...")
    synthetic_docs = generate_synthetic_dataset(n_synthetic)
    valid, invalid = validate_dataset(synthetic_docs)
    console.print(f"  ✓ Generated {valid} valid documents ({invalid} invalid)")
    
    # Adversarial cases
    console.print(f"\nGenerating {n_adversarial} adversarial cases...")
    adversarial_docs = generate_adversarial_dataset(n_adversarial)
    stats = get_adversarial_stats(adversarial_docs)
    console.print(f"  ✓ Generated {len(adversarial_docs)} adversarial cases")
    console.print("  Distribution:")
    for adv_type, count in sorted(stats.items(), key=lambda x: -x[1])[:5]:
        console.print(f"    {adv_type}: {count}")
    
    # AI4Privacy sample
    ai4privacy_docs = []
    if n_ai4privacy > 0:
        console.print(f"\nLoading {n_ai4privacy} AI4Privacy samples...")
        try:
            ai4privacy_docs = load_ai4privacy_sample(n_ai4privacy)
            console.print(f"  ✓ Loaded {len(ai4privacy_docs)} samples")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not load AI4Privacy: {e}[/yellow]")
            console.print("  Continuing without AI4Privacy data...")
    
    return synthetic_docs, adversarial_docs, ai4privacy_docs


def load_ai4privacy_sample(n: int) -> List:
    """Load sample from AI4Privacy dataset."""
    from .synthetic_generator import LabeledDocument, LabeledEntity
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")
    
    # Load dataset
    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")
    
    # Sample
    indices = list(range(min(n * 2, len(dataset))))  # Get more than needed
    import random
    random.shuffle(indices)
    
    docs = []
    for idx in indices[:n]:
        row = dataset[idx]
        
        # Parse entities from the dataset format
        entities = []
        if "privacy_mask" in row and row["privacy_mask"]:
            # AI4Privacy format has privacy_mask with entity info
            text = row.get("source_text", "")
            masks = row["privacy_mask"]
            
            # Parse mask format (varies by dataset version)
            if isinstance(masks, list):
                for mask in masks:
                    if isinstance(mask, dict):
                        entities.append(LabeledEntity(
                            start=mask.get("start", 0),
                            end=mask.get("end", 0),
                            text=mask.get("value", ""),
                            entity_type=normalize_ai4privacy_type(mask.get("label", "UNKNOWN")),
                        ))
        
        if entities:  # Only include docs with entities
            docs.append(LabeledDocument(
                id=f"ai4privacy_{idx}",
                text=row.get("source_text", ""),
                entities=entities,
                doc_type="ai4privacy",
                metadata={"source": "ai4privacy"},
            ))
    
    return docs


def normalize_ai4privacy_type(label: str) -> str:
    """Normalize AI4Privacy entity types to our types."""
    mapping = {
        "FIRSTNAME": "NAME_PERSON",
        "LASTNAME": "NAME_PERSON",
        "FULLNAME": "NAME_PERSON",
        "NAME": "NAME_PERSON",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE",
        "PHONENUMBER": "PHONE",
        "SSN": "SSN",
        "SOCIALSECURITYNUMBER": "SSN",
        "DATE": "DATE",
        "DOB": "DATE_DOB",
        "DATEOFBIRTH": "DATE_DOB",
        "ADDRESS": "ADDRESS",
        "STREETADDRESS": "ADDRESS",
        "CITY": "LOCATION",
        "STATE": "LOCATION",
        "ZIP": "ZIP",
        "ZIPCODE": "ZIP",
        "CREDITCARD": "CREDIT_CARD",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "IP": "IP_ADDRESS",
        "IPADDRESS": "IP_ADDRESS",
        "URL": "URL",
        "USERNAME": "USERNAME",
        "PASSWORD": "PASSWORD",
    }
    return mapping.get(label.upper().replace("_", ""), label.upper())


def capture_signals_for_documents(
    docs: List,
    engine,
) -> List[Dict]:
    """Run detection and capture signals for each document."""
    from ..engine.classifier import SpanSignals
    
    console.print("\n[bold]Step 2: Capture Detection Signals[/bold]")
    console.print("─" * 50)
    
    all_signals = []
    
    engine.capture_signals = True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(docs))
        
        for doc in docs:
            # Clear previous signals
            engine.clear_captured_signals()
            
            # Run detection
            try:
                engine.detect(doc.text, verify=False)
            except Exception as e:
                logger.warning(f"Detection failed for {doc.id}: {e}")
                progress.update(task, advance=1)
                continue
            
            # Get captured signals
            signals = engine.get_captured_signals()
            
            # Label signals against ground truth
            labeled_signals = label_signals(signals, doc)
            all_signals.extend(labeled_signals)
            
            progress.update(task, advance=1)
    
    engine.capture_signals = False
    
    # Stats
    total = len(all_signals)
    positive = sum(1 for s in all_signals if s["ground_truth_type"] != "NONE")
    negative = total - positive
    
    console.print(f"\n  Total signals captured: {total}")
    console.print(f"  Positive (real entities): {positive} ({positive/total*100:.1f}%)")
    console.print(f"  Negative (false positives): {negative} ({negative/total*100:.1f}%)")
    
    return all_signals


def label_signals(signals: List, doc) -> List[Dict]:
    """Label captured signals against document ground truth."""
    labeled = []
    
    # Build ground truth lookup with tolerance
    gt_lookup = {}
    for entity in doc.entities:
        for offset in range(-2, 3):  # ±2 char tolerance
            gt_lookup[(entity.start + offset, entity.end + offset)] = entity.entity_type
    
    for signal in signals:
        signal_dict = signal_to_dict(signal)
        signal_dict["document_id"] = doc.id
        signal_dict["doc_type"] = doc.doc_type
        
        # Check exact match first
        span_key = (signal.span_start, signal.span_end)
        if span_key in gt_lookup:
            signal_dict["ground_truth_type"] = gt_lookup[span_key]
            signal_dict["ground_truth_source"] = "exact_match"
        else:
            # Check overlap match
            matched = False
            for entity in doc.entities:
                overlap = calculate_overlap(
                    signal.span_start, signal.span_end,
                    entity.start, entity.end
                )
                if overlap > 0.5:
                    signal_dict["ground_truth_type"] = entity.entity_type
                    signal_dict["ground_truth_source"] = "overlap_match"
                    matched = True
                    break
            
            if not matched:
                signal_dict["ground_truth_type"] = "NONE"
                signal_dict["ground_truth_source"] = "no_match"
        
        labeled.append(signal_dict)
    
    return labeled


def signal_to_dict(signal) -> Dict:
    """Convert SpanSignals to dictionary."""
    return {
        "id": signal.id,
        "span_start": signal.span_start,
        "span_end": signal.span_end,
        "span_text": signal.span_text,
        "phi_bert_detected": signal.phi_bert_detected,
        "phi_bert_conf": signal.phi_bert_conf,
        "phi_bert_type": signal.phi_bert_type,
        "pii_bert_detected": signal.pii_bert_detected,
        "pii_bert_conf": signal.pii_bert_conf,
        "pii_bert_type": signal.pii_bert_type,
        "presidio_detected": signal.presidio_detected,
        "presidio_conf": signal.presidio_conf,
        "presidio_type": signal.presidio_type,
        "rule_detected": signal.rule_detected,
        "rule_conf": signal.rule_conf,
        "rule_type": signal.rule_type,
        "rule_has_checksum": signal.rule_has_checksum,
        "sources_agree_count": signal.sources_agree_count,
        "span_length": signal.span_length,
        "has_digits": signal.has_digits,
        "has_letters": signal.has_letters,
        "all_caps": signal.all_caps,
        "all_digits": signal.all_digits,
        "mixed_case": signal.mixed_case,
        "merged_type": signal.merged_type,
        "merged_conf": signal.merged_conf,
        "merged_source": signal.merged_source,
    }


def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate overlap ratio between two spans."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_len = overlap_end - overlap_start
    min_len = min(end1 - start1, end2 - start2)
    
    if min_len == 0:
        return 0.0
    
    return overlap_len / min_len


def train_meta_classifier(
    signals: List[Dict],
    output_dir: Path,
    use_xgboost: bool = False,
    test_size: float = 0.2,
) -> Dict:
    """Train the meta-classifier on labeled signals."""
    console.print("\n[bold]Step 3: Train Meta-Classifier[/bold]")
    console.print("─" * 50)
    
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
    
    # Prepare features
    feature_names = [
        "phi_bert_detected", "phi_bert_conf",
        "pii_bert_detected", "pii_bert_conf",
        "presidio_detected", "presidio_conf",
        "rule_detected", "rule_conf", "rule_has_checksum",
        "sources_agree_count", "span_length",
        "has_digits", "has_letters", "all_caps", "all_digits", "mixed_case",
    ]
    
    X = []
    y_is_entity = []
    y_entity_type = []
    
    for s in signals:
        features = [
            int(s["phi_bert_detected"]),
            s["phi_bert_conf"],
            int(s["pii_bert_detected"]),
            s["pii_bert_conf"],
            int(s["presidio_detected"]),
            s["presidio_conf"],
            int(s["rule_detected"]),
            s["rule_conf"],
            int(s["rule_has_checksum"]),
            s["sources_agree_count"],
            s["span_length"],
            int(s["has_digits"]),
            int(s["has_letters"]),
            int(s["all_caps"]),
            int(s["all_digits"]),
            int(s["mixed_case"]),
        ]
        X.append(features)
        
        is_entity = 1 if s["ground_truth_type"] != "NONE" else 0
        y_is_entity.append(is_entity)
        y_entity_type.append(s["ground_truth_type"])
    
    X = np.array(X)
    y_is_entity = np.array(y_is_entity)
    
    console.print(f"\n  Features: {len(feature_names)}")
    console.print(f"  Samples: {len(X)}")
    console.print(f"  Positive: {sum(y_is_entity)} ({sum(y_is_entity)/len(y_is_entity)*100:.1f}%)")
    console.print(f"  Negative: {len(y_is_entity) - sum(y_is_entity)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_is_entity, test_size=test_size, random_state=42, stratify=y_is_entity
    )
    
    console.print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train is_entity classifier
    console.print("\n  Training is_entity classifier...")
    
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        except ImportError:
            console.print("  [yellow]XGBoost not available, using RandomForest[/yellow]")
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    console.print(f"\n  [bold]Results (is_entity):[/bold]")
    console.print(f"    F1:        {f1:.1%}")
    console.print(f"    Precision: {precision:.1%}")
    console.print(f"    Recall:    {recall:.1%}")
    
    # Feature importance
    importances = list(zip(feature_names, clf.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    
    console.print(f"\n  [bold]Top Features:[/bold]")
    for name, imp in importances[:5]:
        console.print(f"    {name}: {imp:.3f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    model_path = output_dir / "is_entity_model.pkl"
    joblib.dump(clf, model_path)
    console.print(f"\n  Model saved: {model_path}")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "importances": importances,
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "use_xgboost": use_xgboost,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train meta-classifier")
    parser.add_argument("--synthetic", type=int, default=3000, help="Number of synthetic docs")
    parser.add_argument("--adversarial", type=int, default=1000, help="Number of adversarial cases")
    parser.add_argument("--ai4privacy", type=int, default=1000, help="Number of AI4Privacy samples")
    parser.add_argument("--xgboost", action="store_true", help="Use XGBoost instead of RandomForest")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock model (for testing)")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]═══ Meta-Classifier Training Pipeline ═══[/bold cyan]\n")
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        from ..config import get_config
        output_dir = get_config().data_dir / "meta_classifier"
    
    # Generate data
    synthetic_docs, adversarial_docs, ai4privacy_docs = generate_training_data(
        n_synthetic=args.synthetic,
        n_adversarial=args.adversarial,
        n_ai4privacy=args.ai4privacy,
    )
    
    all_docs = synthetic_docs + adversarial_docs + ai4privacy_docs
    console.print(f"\n  Total documents: {len(all_docs)}")
    
    # Initialize engine
    console.print("\n  Initializing detection engine...")
    from ..config import get_config
    from ..engine.classifier import ClassificationEngine
    
    config = get_config()
    engine = ClassificationEngine(use_mock_model=args.mock, config=config)
    
    # Capture signals
    signals = capture_signals_for_documents(all_docs, engine)
    
    # Save signals for analysis
    signals_path = output_dir / "training_signals.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(signals_path, "w") as f:
        json.dump(signals, f)
    console.print(f"\n  Signals saved: {signals_path}")
    
    # Train
    metrics = train_meta_classifier(
        signals,
        output_dir,
        use_xgboost=args.xgboost,
    )
    
    console.print("\n[bold green]═══ Training Complete ═══[/bold green]\n")
    
    # Summary table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Documents", str(len(all_docs)))
    table.add_row("Signals", str(len(signals)))
    table.add_row("F1 Score", f"{metrics['f1']:.1%}")
    table.add_row("Precision", f"{metrics['precision']:.1%}")
    table.add_row("Recall", f"{metrics['recall']:.1%}")
    table.add_row("Output", str(output_dir))
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
