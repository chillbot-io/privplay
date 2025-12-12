"""Benchmark runner for PHI/PII detection evaluation."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import logging

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .datasets import BenchmarkDataset, BenchmarkSample, AnnotatedEntity
from .storage import BenchmarkStorage, BenchmarkRun
from ..types import Entity, EntityType, SourceType
from ..engine.classifier import ClassificationEngine

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class MatchResult:
    """Result of matching detected vs ground-truth entities."""
    detected: Optional[Entity]
    ground_truth: Optional[AnnotatedEntity]
    match_type: str  # 'true_positive', 'false_positive', 'false_negative'
    component: str  # 'model', 'rule', 'presidio', 'merged'


@dataclass
class SampleResult:
    """Benchmark result for a single sample."""
    sample_id: str
    matches: List[MatchResult]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    run_id: str
    timestamp: datetime
    dataset_name: str
    num_samples: int
    
    # Overall metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Per-type breakdown
    by_entity_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-component breakdown
    by_component: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-sample results (for detailed analysis)
    sample_results: List[SampleResult] = field(default_factory=list)
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_benchmark_run(self) -> BenchmarkRun:
        """Convert to BenchmarkRun for storage."""
        return BenchmarkRun(
            run_id=self.run_id,
            timestamp=self.timestamp,
            dataset_name=self.dataset_name,
            num_samples=self.num_samples,
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            false_negatives=self.false_negatives,
            by_entity_type=self.by_entity_type,
            by_component=self.by_component,
            config=self.config,
        )


class BenchmarkRunner:
    """Runs benchmarks against detection engine."""
    
    def __init__(
        self,
        engine: ClassificationEngine,
        storage: Optional[BenchmarkStorage] = None,
        overlap_threshold: float = 0.5,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            engine: Classification engine to benchmark
            storage: Storage for persisting results
            overlap_threshold: Minimum overlap ratio to count as match
        """
        self.engine = engine
        self.storage = storage
        self.overlap_threshold = overlap_threshold
    
    def run(
        self,
        dataset: BenchmarkDataset,
        verify: bool = False,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: Dataset to benchmark against
            verify: Whether to run LLM verification
            show_progress: Whether to show progress bar
            
        Returns:
            BenchmarkResult with metrics
        """
        run_id = f"bench_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        sample_results = []
        
        # Per-type and per-component counters
        type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        component_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        samples = list(dataset)
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Benchmarking {dataset.name}...",
                    total=len(samples)
                )
                
                for sample in samples:
                    result = self._evaluate_sample(sample, verify)
                    sample_results.append(result)
                    
                    total_tp += result.true_positives
                    total_fp += result.false_positives
                    total_fn += result.false_negatives
                    
                    # Aggregate by type and component
                    for match in result.matches:
                        if match.match_type == "true_positive":
                            entity_type = match.ground_truth.normalized_type
                            type_counts[entity_type]["tp"] += 1
                            component_counts[match.component]["tp"] += 1
                            
                        elif match.match_type == "false_positive":
                            entity_type = match.detected.entity_type.value
                            type_counts[entity_type]["fp"] += 1
                            component_counts[match.component]["fp"] += 1
                            
                        elif match.match_type == "false_negative":
                            entity_type = match.ground_truth.normalized_type
                            type_counts[entity_type]["fn"] += 1
                    
                    progress.update(task, advance=1)
        else:
            for sample in samples:
                result = self._evaluate_sample(sample, verify)
                sample_results.append(result)
                
                total_tp += result.true_positives
                total_fp += result.false_positives
                total_fn += result.false_negatives
                
                for match in result.matches:
                    if match.match_type == "true_positive":
                        entity_type = match.ground_truth.normalized_type
                        type_counts[entity_type]["tp"] += 1
                        component_counts[match.component]["tp"] += 1
                    elif match.match_type == "false_positive":
                        entity_type = match.detected.entity_type.value
                        type_counts[entity_type]["fp"] += 1
                        component_counts[match.component]["fp"] += 1
                    elif match.match_type == "false_negative":
                        entity_type = match.ground_truth.normalized_type
                        type_counts[entity_type]["fn"] += 1
        
        # Calculate metrics
        precision, recall, f1 = self._calculate_metrics(total_tp, total_fp, total_fn)
        
        # Calculate per-type metrics
        by_entity_type = {}
        for entity_type, counts in type_counts.items():
            p, r, f = self._calculate_metrics(counts["tp"], counts["fp"], counts["fn"])
            by_entity_type[entity_type] = {
                "precision": p,
                "recall": r,
                "f1": f,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        
        # Calculate per-component metrics
        by_component = {}
        for component, counts in component_counts.items():
            p, r, f = self._calculate_metrics(counts["tp"], counts["fp"], counts["fn"])
            by_component[component] = {
                "precision": p,
                "recall": r,
                "f1": f,
                "tp": counts["tp"],
                "fp": counts["fp"],
            }
        
        result = BenchmarkResult(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            dataset_name=dataset.name,
            num_samples=len(samples),
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            by_entity_type=by_entity_type,
            by_component=by_component,
            sample_results=sample_results,
            config={
                "verify": verify,
                "overlap_threshold": self.overlap_threshold,
                "model": getattr(self.engine.model, "name", "unknown"),
            },
        )
        
        # Save to storage
        if self.storage:
            self.storage.save_run(result.to_benchmark_run())
        
        return result
    
    def _evaluate_sample(
        self,
        sample: BenchmarkSample,
        verify: bool,
    ) -> SampleResult:
        """Evaluate a single sample."""
        # Run detection
        detected = self.engine.detect(sample.text, verify=verify)
        
        # Match detected entities to ground truth
        matches = self._match_entities(detected, sample.entities)
        
        tp = sum(1 for m in matches if m.match_type == "true_positive")
        fp = sum(1 for m in matches if m.match_type == "false_positive")
        fn = sum(1 for m in matches if m.match_type == "false_negative")
        
        return SampleResult(
            sample_id=sample.id,
            matches=matches,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
    
    def _match_entities(
        self,
        detected: List[Entity],
        ground_truth: List[AnnotatedEntity],
    ) -> List[MatchResult]:
        """Match detected entities to ground truth."""
        matches = []
        matched_gt_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()
        
        # Try to match each detected entity to ground truth
        for det_idx, det_entity in enumerate(detected):
            best_match_idx = None
            best_overlap = 0.0
            
            for gt_idx, gt_entity in enumerate(ground_truth):
                if gt_idx in matched_gt_indices:
                    continue
                
                overlap = self._calculate_overlap(
                    det_entity.start, det_entity.end,
                    gt_entity.start, gt_entity.end,
                )
                
                if overlap >= self.overlap_threshold and overlap > best_overlap:
                    # Also check if types match (at least partially)
                    if self._types_compatible(det_entity.entity_type.value, gt_entity.normalized_type):
                        best_match_idx = gt_idx
                        best_overlap = overlap
            
            if best_match_idx is not None:
                # True positive
                matched_gt_indices.add(best_match_idx)
                matched_det_indices.add(det_idx)
                matches.append(MatchResult(
                    detected=det_entity,
                    ground_truth=ground_truth[best_match_idx],
                    match_type="true_positive",
                    component=det_entity.source.value,
                ))
            else:
                # False positive
                matches.append(MatchResult(
                    detected=det_entity,
                    ground_truth=None,
                    match_type="false_positive",
                    component=det_entity.source.value,
                ))
        
        # Any unmatched ground truth entities are false negatives
        for gt_idx, gt_entity in enumerate(ground_truth):
            if gt_idx not in matched_gt_indices:
                matches.append(MatchResult(
                    detected=None,
                    ground_truth=gt_entity,
                    match_type="false_negative",
                    component="none",
                ))
        
        return matches
    
    def _calculate_overlap(
        self,
        start1: int, end1: int,
        start2: int, end2: int,
    ) -> float:
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
    
    def _types_compatible(self, detected_type: str, gt_type: str) -> bool:
        """Check if detected and ground truth types are compatible."""
        # Exact match
        if detected_type == gt_type:
            return True
        
        # Compatible name types
        name_types = {"NAME_PERSON", "NAME_PATIENT", "NAME_PROVIDER", "NAME_RELATIVE"}
        if detected_type in name_types and gt_type in name_types:
            return True
        
        # Compatible date types
        date_types = {"DATE", "DATE_DOB", "DATE_ADMISSION", "DATE_DISCHARGE"}
        if detected_type in date_types and gt_type in date_types:
            return True
        
        # Compatible address types
        address_types = {"ADDRESS", "ZIP"}
        if detected_type in address_types and gt_type in address_types:
            return True
        
        # Compatible account types
        account_types = {"ACCOUNT_NUMBER", "BANK_ACCOUNT", "MRN"}
        if detected_type in account_types and gt_type in account_types:
            return True
        
        return False
    
    def _calculate_metrics(
        self,
        tp: int,
        fp: int,
        fn: int,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1


def display_benchmark_result(
    result: BenchmarkResult,
    history: Optional[List[BenchmarkRun]] = None,
) -> None:
    """Display benchmark results with rich formatting."""
    console.print()
    console.print(f"[bold]Benchmark Results: {result.dataset_name}[/bold]")
    console.print("─" * 60)
    console.print(f"  Run ID: {result.run_id}")
    console.print(f"  Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"  Samples: {result.num_samples}")
    console.print()
    
    # Overall metrics
    f1_style = "green" if result.f1 >= 0.9 else "yellow" if result.f1 >= 0.7 else "red"
    console.print("[bold]Overall Metrics:[/bold]")
    console.print(f"  Precision: {result.precision:.1%}")
    console.print(f"  Recall:    {result.recall:.1%}")
    console.print(f"  F1 Score:  [{f1_style}]{result.f1:.1%}[/{f1_style}]")
    console.print()
    console.print(f"  True Positives:  {result.true_positives}")
    console.print(f"  False Positives: {result.false_positives}")
    console.print(f"  False Negatives: {result.false_negatives}")
    console.print()
    
    # History trend
    if history and len(history) > 1:
        console.print("[bold]Recent History (last 5 runs):[/bold]")
        
        history_table = Table(show_header=True, header_style="bold")
        history_table.add_column("Run")
        history_table.add_column("Date")
        history_table.add_column("Precision", justify="right")
        history_table.add_column("Recall", justify="right")
        history_table.add_column("F1", justify="right")
        history_table.add_column("Δ", justify="right")
        
        prev_f1 = None
        for i, run in enumerate(history[:5]):
            delta = ""
            if prev_f1 is not None:
                diff = run.f1 - prev_f1
                if diff > 0:
                    delta = f"[green]+{diff:.1%}[/green]"
                elif diff < 0:
                    delta = f"[red]{diff:.1%}[/red]"
                else:
                    delta = "="
            
            f1_style = "green" if run.f1 >= 0.9 else "yellow" if run.f1 >= 0.7 else "red"
            is_current = run.run_id == result.run_id
            row_style = "bold" if is_current else None
            
            history_table.add_row(
                f"{'→ ' if is_current else ''}{i+1}",
                run.timestamp.strftime("%m/%d %H:%M"),
                f"{run.precision:.1%}",
                f"{run.recall:.1%}",
                f"[{f1_style}]{run.f1:.1%}[/{f1_style}]",
                delta,
                style=row_style,
            )
            prev_f1 = run.f1
        
        console.print(history_table)
        console.print()
    
    # Per-entity-type breakdown
    if result.by_entity_type:
        console.print("[bold]By Entity Type:[/bold]")
        
        type_table = Table(show_header=True, header_style="bold")
        type_table.add_column("Entity Type")
        type_table.add_column("Precision", justify="right")
        type_table.add_column("Recall", justify="right")
        type_table.add_column("F1", justify="right")
        type_table.add_column("TP", justify="right")
        type_table.add_column("FP", justify="right")
        type_table.add_column("FN", justify="right")
        
        sorted_types = sorted(
            result.by_entity_type.items(),
            key=lambda x: x[1]["f1"],
            reverse=True
        )
        
        for entity_type, metrics in sorted_types[:15]:  # Top 15
            f1_style = "green" if metrics["f1"] >= 0.9 else "yellow" if metrics["f1"] >= 0.7 else "red"
            type_table.add_row(
                entity_type,
                f"{metrics['precision']:.1%}",
                f"{metrics['recall']:.1%}",
                f"[{f1_style}]{metrics['f1']:.1%}[/{f1_style}]",
                str(metrics.get("tp", 0)),
                str(metrics.get("fp", 0)),
                str(metrics.get("fn", 0)),
            )
        
        if len(result.by_entity_type) > 15:
            type_table.add_row("...", "", "", "", "", "", "")
        
        console.print(type_table)
        console.print()
    
    # Per-component breakdown
    if result.by_component:
        console.print("[bold]By Stack Component:[/bold]")
        
        comp_table = Table(show_header=True, header_style="bold")
        comp_table.add_column("Component")
        comp_table.add_column("Precision", justify="right")
        comp_table.add_column("TP", justify="right")
        comp_table.add_column("FP", justify="right")
        
        for component, metrics in sorted(result.by_component.items()):
            comp_table.add_row(
                component,
                f"{metrics['precision']:.1%}",
                str(metrics.get("tp", 0)),
                str(metrics.get("fp", 0)),
            )
        
        console.print(comp_table)
        console.print()


def capture_benchmark_errors(
    result: BenchmarkResult,
    dataset: BenchmarkDataset,
    db = None,
) -> dict:
    """
    Capture FPs from benchmark as REJECTED corrections for training.
    
    Args:
        result: Benchmark result with sample_results
        dataset: Original dataset (for sample text)
        db: Database instance (uses default if None)
        
    Returns:
        Dict with counts: {'documents': n, 'fps_captured': n, 'fns_skipped': n}
    """
    from ..db import get_db
    from ..types import Document, Entity, Correction, EntityType, DecisionType, SourceType
    import uuid
    
    if db is None:
        db = get_db()
    
    # Build sample lookup
    sample_lookup = {s.id: s for s in dataset.samples}
    
    docs_created = 0
    fps_captured = 0
    fns_skipped = 0
    
    for sample_result in result.sample_results:
        sample = sample_lookup.get(sample_result.sample_id)
        if not sample:
            continue
            
        # Get FPs for this sample
        fps = [m for m in sample_result.matches if m.match_type == "false_positive"]
        fns = [m for m in sample_result.matches if m.match_type == "false_negative"]
        
        if not fps:
            fns_skipped += len(fns)
            continue
        
        # Create document for this sample
        doc = Document(
            id=f"bench_{sample.id}",
            content=sample.text,
            source=f"benchmark:{sample.source}",
        )
        
        try:
            db.add_document(doc)
            docs_created += 1
        except Exception:
            # Document may already exist from previous run
            pass
        
        # Create corrections for FPs
        for match in fps:
            detected = match.detected
            
            # Create entity record
            entity = Entity(
                id=str(uuid.uuid4()),
                text=detected.text,
                start=detected.start,
                end=detected.end,
                entity_type=detected.entity_type,
                confidence=detected.confidence,
                source=detected.source,
            )
            
            try:
                db.add_entity(entity, doc.id)
            except Exception:
                # May already exist
                pass
            
            # Get context
            ctx_start = max(0, detected.start - 50)
            ctx_end = min(len(sample.text), detected.end + 50)
            
            # Create REJECTED correction
            correction = Correction(
                id=str(uuid.uuid4()),
                entity_id=entity.id,
                document_id=doc.id,
                entity_text=detected.text,
                entity_start=detected.start,
                entity_end=detected.end,
                detected_type=detected.entity_type,
                decision=DecisionType.REJECTED,
                correct_type=None,
                context_before=sample.text[ctx_start:detected.start],
                context_after=sample.text[detected.end:ctx_end],
                ner_confidence=detected.confidence,
            )
            
            try:
                db.add_correction(correction)
                fps_captured += 1
            except Exception as e:
                logger.warning(f"Failed to add correction: {e}")
        
        fns_skipped += len(fns)
    
    return {
        "documents": docs_created,
        "fps_captured": fps_captured,
        "fns_skipped": fns_skipped,
    }

