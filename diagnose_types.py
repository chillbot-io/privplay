#!/usr/bin/env python3
"""Diagnose per-type performance to find problem areas."""

import sys
from collections import defaultdict
from pathlib import Path

from privplay.benchmark.datasets import load_ai4privacy_dataset
from privplay.engine.classifier import ClassificationEngine
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_by_type(n_samples: int = 500):
    """Run benchmark and break down results by entity type."""
    
    console.print(f"\n[bold]Loading {n_samples} AI4Privacy samples...[/bold]")
    dataset = load_ai4privacy_dataset(max_samples=n_samples)
    samples = dataset.samples
    
    console.print("[bold]Initializing engine...[/bold]")
    engine = ClassificationEngine(
        use_coreference=False,
        capture_signals=False,
        use_meta_classifier=True,
    )
    
    # Track per-type stats
    type_stats = defaultdict(lambda: {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "type_errors": 0,  # Right span, wrong type
        "ground_truth_count": 0,
        "predicted_count": 0,
    })
    
    # Track specific spurious examples for debugging
    spurious_examples = defaultdict(list)
    missed_examples = defaultdict(list)
    type_confusion = defaultdict(lambda: defaultdict(int))  # pred_type -> true_type -> count
    
    console.print(f"\n[bold]Processing {len(samples)} samples...[/bold]")
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            console.print(f"  Processing {i}/{len(samples)}...")
        
        text = sample.text
        ground_truth = sample.entities  # List of AnnotatedEntity
        
        # Run detection
        try:
            detected = engine.detect(text, verify=False)
        except Exception as e:
            console.print(f"[red]Error on sample {i}: {e}[/red]")
            continue
        
        # Convert to comparable format
        pred_spans = {}  # (start, end) -> type
        for ent in detected:
            pred_spans[(ent.start, ent.end)] = ent.entity_type.name
            type_stats[ent.entity_type.name]["predicted_count"] += 1
        
        true_spans = {}  # (start, end) -> type
        for ent in ground_truth:
            # Use normalized_type for comparison
            true_spans[(ent.start, ent.end)] = ent.normalized_type
            type_stats[ent.normalized_type]["ground_truth_count"] += 1
        
        # Match predictions to ground truth
        pred_matched = set()
        true_matched = set()
        
        # Exact matches
        for span, pred_type in pred_spans.items():
            if span in true_spans:
                true_type = true_spans[span]
                pred_matched.add(span)
                true_matched.add(span)
                
                if pred_type == true_type:
                    type_stats[pred_type]["true_positives"] += 1
                else:
                    # Right span, wrong type - track the confusion
                    type_stats[pred_type]["type_errors"] += 1
                    type_confusion[pred_type][true_type] += 1
        
        # Spurious (predicted but not in ground truth)
        for span, pred_type in pred_spans.items():
            if span not in pred_matched:
                # Check for overlaps (partial matches)
                has_overlap = False
                for true_span in true_spans:
                    if span[0] < true_span[1] and true_span[0] < span[1]:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    type_stats[pred_type]["false_positives"] += 1
                    if len(spurious_examples[pred_type]) < 5:
                        start, end = span
                        context_start = max(0, start - 30)
                        context_end = min(len(text), end + 30)
                        spurious_examples[pred_type].append(
                            f"...{text[context_start:start]}[{text[start:end]}]{text[end:context_end]}..."
                        )
        
        # Missed (in ground truth but not predicted)
        for span, true_type in true_spans.items():
            if span not in true_matched:
                # Check for overlaps
                has_overlap = False
                for pred_span in pred_spans:
                    if span[0] < pred_span[1] and pred_span[0] < span[1]:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    type_stats[true_type]["false_negatives"] += 1
                    if len(missed_examples[true_type]) < 5:
                        start, end = span
                        context_start = max(0, start - 30)
                        context_end = min(len(text), end + 30)
                        missed_examples[true_type].append(
                            f"...{text[context_start:start]}[{text[start:end]}]{text[end:context_end]}..."
                        )
    
    # Display results
    console.print("\n" + "=" * 80)
    console.print("[bold]PER-TYPE BREAKDOWN - SORTED BY FALSE POSITIVES[/bold]")
    console.print("=" * 80)
    
    # Sort by false positives (biggest problem)
    sorted_types = sorted(
        type_stats.items(),
        key=lambda x: x[1]["false_positives"],
        reverse=True
    )
    
    # Table for types with false positives
    table = Table(title="Types by False Positives (Spurious Detections)")
    table.add_column("Type", style="cyan")
    table.add_column("FP", justify="right", style="red")
    table.add_column("TP", justify="right", style="green")
    table.add_column("FN", justify="right", style="yellow")
    table.add_column("Prec", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("GT", justify="right", style="dim")
    
    for entity_type, stats in sorted_types[:25]:
        tp = stats["true_positives"]
        fp = stats["false_positives"]
        fn = stats["false_negatives"]
        gt = stats["ground_truth_count"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if fp > 0 or tp > 0 or fn > 0:
            table.add_row(
                entity_type[:20],
                str(fp),
                str(tp),
                str(fn),
                f"{precision:.0%}",
                f"{recall:.0%}",
                f"{f1:.0%}",
                str(gt),
            )
    
    console.print(table)
    
    # Show worst offenders with examples
    console.print("\n" + "=" * 80)
    console.print("[bold]TOP SPURIOUS DETECTION EXAMPLES (FALSE POSITIVES)[/bold]")
    console.print("=" * 80)
    
    for entity_type, stats in sorted_types[:8]:
        if stats["false_positives"] > 0:
            console.print(f"\n[bold red]{entity_type}[/bold red] ({stats['false_positives']} false positives):")
            for ex in spurious_examples[entity_type][:3]:
                console.print(f"  • {ex[:120]}")
    
    # Show most missed
    console.print("\n" + "=" * 80)
    console.print("[bold]TOP MISSED ENTITY EXAMPLES (FALSE NEGATIVES)[/bold]")
    console.print("=" * 80)
    
    sorted_by_fn = sorted(
        type_stats.items(),
        key=lambda x: x[1]["false_negatives"],
        reverse=True
    )
    
    for entity_type, stats in sorted_by_fn[:8]:
        if stats["false_negatives"] > 0:
            console.print(f"\n[bold yellow]{entity_type}[/bold yellow] ({stats['false_negatives']} missed):")
            for ex in missed_examples[entity_type][:3]:
                console.print(f"  • {ex[:120]}")
    
    # Type confusion matrix (top confusions)
    console.print("\n" + "=" * 80)
    console.print("[bold]TYPE CONFUSION (predicted as X, was actually Y)[/bold]")
    console.print("=" * 80)
    
    confusions = []
    for pred_type, true_types in type_confusion.items():
        for true_type, count in true_types.items():
            confusions.append((pred_type, true_type, count))
    
    confusions.sort(key=lambda x: -x[2])
    
    for pred_type, true_type, count in confusions[:15]:
        console.print(f"  {pred_type} → {true_type}: {count}")
    
    # Summary stats
    total_fp = sum(s["false_positives"] for s in type_stats.values())
    total_tp = sum(s["true_positives"] for s in type_stats.values())
    total_fn = sum(s["false_negatives"] for s in type_stats.values())
    
    console.print("\n" + "=" * 80)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 80)
    console.print(f"Total True Positives:  {total_tp}")
    console.print(f"Total False Positives: {total_fp}")
    console.print(f"Total False Negatives: {total_fn}")
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    console.print(f"\nOverall Precision: {overall_precision:.1%}")
    console.print(f"Overall Recall:    {overall_recall:.1%}")
    console.print(f"Overall F1:        {overall_f1:.1%}")
    
    # Actionable recommendations
    console.print("\n" + "=" * 80)
    console.print("[bold]RECOMMENDATIONS[/bold]")
    console.print("=" * 80)
    
    # Top 3 FP sources
    top_fp_types = [t for t, s in sorted_types[:3] if s["false_positives"] > 10]
    if top_fp_types:
        console.print(f"\n[red]Precision killers:[/red] {', '.join(top_fp_types)}")
        console.print("  → Check rules/Presidio patterns for these types")
        console.print("  → Consider raising confidence threshold")
        console.print("  → Add to allowlist if common false patterns")
    
    # Top 3 FN sources
    top_fn_types = [t for t, s in sorted_by_fn[:3] if s["false_negatives"] > 10]
    if top_fn_types:
        console.print(f"\n[yellow]Recall killers:[/yellow] {', '.join(top_fn_types)}")
        console.print("  → Check if these types are in training data")
        console.print("  → May need more training examples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500, help="Number of samples")
    args = parser.parse_args()
    
    analyze_by_type(args.n)
