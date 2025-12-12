"""Evaluation and F1 scoring."""

from typing import List, Dict, Optional
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from ..types import Entity, EntityType, Correction, DecisionType, TestResult
from ..db import Database, get_db

console = Console()


def calculate_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1."""
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def evaluate_from_corrections(
    db: Optional[Database] = None,
) -> TestResult:
    """
    Evaluate model performance based on human corrections.
    
    Logic:
    - Confirmed = True Positive (model was right)
    - Rejected = False Positive (model was wrong - flagged non-PHI)
    - Changed = Partial (model detected PHI but wrong type)
    
    Note: This doesn't capture False Negatives (PHI that model missed)
    because those aren't in the corrections table. That would require
    a fully annotated test set.
    """
    if db is None:
        db = get_db()
    
    corrections = db.get_corrections()
    
    if not corrections:
        return TestResult(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            total_entities=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            by_type={},
        )
    
    # Count by type
    by_type: Dict[EntityType, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    
    total_tp = 0
    total_fp = 0
    total_fn = 0  # Can't measure FN from corrections alone
    
    for correction in corrections:
        entity_type = correction.detected_type
        
        if correction.decision == DecisionType.CONFIRMED:
            total_tp += 1
            by_type[entity_type]["tp"] += 1
        
        elif correction.decision == DecisionType.REJECTED:
            total_fp += 1
            by_type[entity_type]["fp"] += 1
        
        elif correction.decision == DecisionType.CHANGED:
            # Partial credit - detected PHI but wrong type
            # Count as FP for detected type, TP for correct type
            total_fp += 1
            by_type[entity_type]["fp"] += 1
            
            if correction.correct_type:
                total_tp += 1
                by_type[correction.correct_type]["tp"] += 1
    
    precision, recall, f1 = calculate_f1(total_tp, total_fp, total_fn)
    
    # Calculate per-type metrics
    type_results = {}
    for entity_type, counts in by_type.items():
        p, r, f = calculate_f1(counts["tp"], counts["fp"], counts["fn"])
        type_results[entity_type] = {"precision": p, "recall": r, "f1": f}
    
    return TestResult(
        precision=precision,
        recall=recall,
        f1=f1,
        total_entities=len(corrections),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        by_type=type_results,
    )


def display_test_results(result: TestResult) -> None:
    """Display test results."""
    console.print()
    console.print("[bold]Evaluation Results[/bold]")
    console.print("─" * 50)
    console.print()
    
    # Overall metrics
    console.print(f"  [bold]Overall:[/bold]")
    console.print(f"    Precision: {result.precision:.3f}")
    console.print(f"    Recall:    {result.recall:.3f} [dim](limited - can't measure FN)[/dim]")
    console.print(f"    F1:        {result.f1:.3f}")
    console.print()
    console.print(f"    True Positives:  {result.true_positives}")
    console.print(f"    False Positives: {result.false_positives}")
    console.print(f"    False Negatives: {result.false_negatives} [dim](not measurable)[/dim]")
    console.print()
    
    if not result.by_type:
        return
    
    # Per-type metrics
    console.print("[bold]By Entity Type:[/bold]")
    console.print()
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    
    for entity_type, metrics in sorted(result.by_type.items(), key=lambda x: x[1]["f1"], reverse=True):
        f1_style = "green" if metrics["f1"] >= 0.9 else "yellow" if metrics["f1"] >= 0.7 else "red"
        
        table.add_row(
            entity_type.value,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"[{f1_style}]{metrics['f1']:.3f}[/{f1_style}]",
        )
    
    console.print(table)
    console.print()
    
    # Note about limitations
    console.print("[dim]Note: Recall and F1 are limited because we can only measure")
    console.print("false negatives (missed PHI) with a fully annotated test set.[/dim]")
    console.print()
