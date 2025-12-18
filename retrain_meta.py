#!/usr/bin/env python3
"""
Retrain meta-classifier for simplified pipeline.

Collects signals from PHI-BERT + PII-BERT + Checksum rules,
then trains XGBoost to make accept/reject decisions.

Usage:
    python retrain_meta.py --samples 1000
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


def collect_training_data(num_samples: int = 1000):
    """
    Collect training data by running simplified pipeline on AI4Privacy.
    
    Returns list of (features, label) tuples.
    """
    console.print(f"\n[bold cyan]═══ Collecting Training Data ═══[/bold cyan]\n")
    
    # Import here to avoid slow startup
    from privplay.benchmark.datasets import load_ai4privacy_dataset
    from privplay.engine.classifier import ClassificationEngine
    
    console.print(f"Loading {num_samples} samples...")
    dataset = load_ai4privacy_dataset(max_samples=num_samples)
    
    console.print("Initializing engine with signal capture...")
    engine = ClassificationEngine(capture_signals=True, use_meta_classifier=False)
    
    training_data = []
    stats = {"tp": 0, "fp": 0, "fn": 0}
    
    console.print(f"\nProcessing samples...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting...", total=len(dataset.samples))
        
        for sample in dataset.samples:
            # Build ground truth spans
            gt_spans = {}
            for entity in sample.entities:
                # Allow fuzzy matching
                for offset in range(-3, 4):
                    key = (entity.start + offset, entity.end + offset)
                    gt_spans[key] = entity.normalized_type
            
            # Run detection
            engine.clear_captured_signals()
            try:
                engine.detect(sample.text, verify=False)
            except Exception as e:
                progress.advance(task)
                continue
            
            # Label each signal
            for signals in engine.get_captured_signals():
                # Check if this span matches ground truth
                span_key = (signals.span_start, signals.span_end)
                
                # Try exact match first
                if span_key in gt_spans:
                    label = 1  # True positive
                    signals.ground_truth_type = gt_spans[span_key]
                    stats["tp"] += 1
                else:
                    # Try overlap matching
                    found_match = False
                    for (gt_start, gt_end), gt_type in gt_spans.items():
                        # Check significant overlap
                        overlap_start = max(signals.span_start, gt_start)
                        overlap_end = min(signals.span_end, gt_end)
                        
                        if overlap_start < overlap_end:
                            overlap_len = overlap_end - overlap_start
                            span_len = signals.span_end - signals.span_start
                            
                            if overlap_len / span_len >= 0.5:
                                label = 1
                                signals.ground_truth_type = gt_type
                                stats["tp"] += 1
                                found_match = True
                                break
                    
                    if not found_match:
                        label = 0  # False positive
                        signals.ground_truth_type = "NONE"
                        stats["fp"] += 1
                
                # Extract features
                features = signals.to_feature_dict()
                training_data.append((features, label, signals))
            
            progress.advance(task)
    
    console.print(f"\n[green]✓ Collected {len(training_data)} training samples[/green]")
    console.print(f"  TP: {stats['tp']} | FP: {stats['fp']}")
    
    return training_data


def train_meta_classifier(
    training_data: List,
    output_dir: str = None,
    use_xgboost: bool = True,
):
    """
    Train XGBoost meta-classifier.
    """
    console.print(f"\n[bold cyan]═══ Training Meta-Classifier ═══[/bold cyan]\n")
    
    if output_dir is None:
        output_dir = Path.home() / ".privplay" / "meta_classifier"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X = []
    y = []
    feature_names = None
    
    for features, label, _ in training_data:
        if feature_names is None:
            feature_names = list(features.keys())
        X.append([features[k] for k in feature_names])
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    console.print(f"Training samples: {len(X)}")
    console.print(f"Features: {len(feature_names)}")
    console.print(f"Positive rate: {y.mean():.1%}")
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    console.print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train model
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
            )
            console.print("\nTraining XGBoost...")
        except ImportError:
            console.print("[yellow]XGBoost not available, using RandomForest[/yellow]")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        console.print("\nTraining RandomForest...")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    
    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    console.print(f"  Precision: {metrics['precision']:.1%}")
    console.print(f"  Recall:    {metrics['recall']:.1%}")
    console.print(f"  F1:        {metrics['f1']:.1%}")
    
    # Feature importance
    console.print(f"\n[bold]Feature Importance:[/bold]")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    table = Table()
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    
    for idx in sorted_idx[:10]:
        table.add_row(feature_names[idx], f"{importances[idx]:.3f}")
    
    console.print(table)
    
    # Save model
    import joblib
    
    model_file = output_dir / "is_entity_model.pkl"
    joblib.dump(model, model_file)
    
    # Save feature importance
    importance_data = {
        "feature_names": feature_names,
        "importances": [(feature_names[i], float(importances[i])) for i in sorted_idx],
        "metrics": metrics,
        "saved_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance_data, f, indent=2)
    
    console.print(f"\n[green]✓ Saved model to {model_file}[/green]")
    
    # Find optimal threshold
    console.print(f"\n[bold]Optimal Threshold Analysis:[/bold]")
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.2, 0.8, 0.05):
        y_pred_t = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        prec = precision_score(y_test, y_pred_t)
        rec = recall_score(y_test, y_pred_t)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        console.print(f"  t={threshold:.2f}: F1={f1:.1%} P={prec:.1%} R={rec:.1%}")
    
    console.print(f"\n[green]Best threshold: {best_threshold:.2f} (F1={best_f1:.1%})[/green]")
    
    return model, metrics


def run_benchmark_test():
    """Run benchmark with retrained model."""
    console.print(f"\n[bold cyan]═══ Benchmark Test ═══[/bold cyan]\n")
    
    from privplay.benchmark.datasets import load_ai4privacy_dataset
    from privplay.benchmark.runner import BenchmarkRunner
    from privplay.engine.classifier import ClassificationEngine
    
    dataset = load_ai4privacy_dataset(max_samples=100)
    engine = ClassificationEngine(use_meta_classifier=True)
    runner = BenchmarkRunner(engine)
    
    result = runner.run(dataset, verify=False, show_progress=True)
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  F1:        {result.f1:.1%}")
    console.print(f"  Precision: {result.precision:.1%}")
    console.print(f"  Recall:    {result.recall:.1%}")
    console.print(f"  TP: {result.true_positives} | FP: {result.false_positives} | FN: {result.false_negatives}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrain meta-classifier")
    parser.add_argument("--samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run benchmark after training")
    parser.add_argument("--no-xgboost", action="store_true", help="Use RandomForest instead")
    
    args = parser.parse_args()
    
    # Collect data
    training_data = collect_training_data(num_samples=args.samples)
    
    # Train
    model, metrics = train_meta_classifier(
        training_data,
        output_dir=args.output,
        use_xgboost=not args.no_xgboost,
    )
    
    # Test
    if args.test:
        run_benchmark_test()
