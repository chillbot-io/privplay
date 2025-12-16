"""Benchmark runner using nervaluate methodology (SemEval 2013).

This implements proper NER evaluation with:
- Four evaluation modes: strict, exact, partial, type
- Entity-level (not token-level) evaluation
- Type mapping and exclusion for non-PII categories
- Standard metrics: precision, recall, F1

References:
- SemEval-2013 Task 9: https://www.cs.york.ac.uk/semeval-2013/task9/
- nervaluate: https://github.com/MantisAI/nervaluate
- David Batista: https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import logging
import json

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from .datasets import BenchmarkDataset, BenchmarkSample, AnnotatedEntity, load_ai4privacy_dataset
from .storage import BenchmarkStorage, BenchmarkRun
from ..types import Entity, EntityType, SourceType
from ..engine.classifier import ClassificationEngine

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# PII TYPE TAXONOMY
# =============================================================================
# Map AI4Privacy raw types to canonical PII categories
# Types mapped to None are EXCLUDED from evaluation (not PII)

TYPE_MAPPING: Dict[str, Optional[str]] = {
    # === NAMES (PII) ===
    "FIRSTNAME": "NAME",
    "LASTNAME": "NAME", 
    "MIDDLENAME": "NAME",
    "NAME_PERSON": "NAME",
    "NAME_PATIENT": "NAME",
    "NAME_PROVIDER": "NAME",
    "NAME_RELATIVE": "NAME",
    "PERSON": "NAME",
    
    # === CONTACT (PII) ===
    "EMAIL": "CONTACT",
    "PHONENUMBER": "CONTACT",
    "PHONE": "CONTACT",
    "FAX": "CONTACT",
    "URL": "CONTACT",
    
    # === DATES (PII under HIPAA) ===
    "DATE": "DATE",
    "DOB": "DATE",
    "DATE_DOB": "DATE",
    "DATE_ADMISSION": "DATE",
    "DATE_DISCHARGE": "DATE",
    "TIME": "DATE",  # Could argue either way
    
    # === LOCATION (PII) ===
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "ZIPCODE": "LOCATION",
    "ZIP": "LOCATION",
    "BUILDINGNUMBER": "LOCATION",
    "SECONDARYADDRESS": "LOCATION",
    "NEARBYGPSCOORDINATE": "LOCATION",
    "LOCATION": "LOCATION",
    
    # === IDENTIFIERS (PII) ===
    "SSN": "IDNUM",
    "US_SSN": "IDNUM",
    "ACCOUNTNUMBER": "IDNUM",
    "ACCOUNT_NUMBER": "IDNUM",
    "BANK_ACCOUNT": "IDNUM",
    "MRN": "IDNUM",
    "HEALTH_PLAN_ID": "IDNUM",
    "DRIVER_LICENSE": "IDNUM",
    "PASSPORT": "IDNUM",
    "IBAN": "IDNUM",
    
    # === FINANCIAL (PII) ===
    "CREDITCARDNUMBER": "FINANCIAL",
    "CREDIT_CARD": "FINANCIAL",
    "CREDITCARDCVV": "FINANCIAL",
    "PIN": "FINANCIAL",
    
    # === DIGITAL IDENTIFIERS (PII) ===
    "IPV4": "DIGITAL",
    "IPV6": "DIGITAL",
    "IP": "DIGITAL",
    "IP_ADDRESS": "DIGITAL",
    "MAC": "DIGITAL",
    "MAC_ADDRESS": "DIGITAL",
    "USERNAME": "DIGITAL",
    "PASSWORD": "DIGITAL",  # Debatable - credential not identity
    "USERAGENT": None,  # Not PII - exclude
    
    # === DEVICE IDS (PII under HIPAA) ===
    "PHONEIMEI": "DEVICE",
    "DEVICE_ID": "DEVICE",
    "VEHICLEVIN": "DEVICE",
    "VIN": "DEVICE",
    "VEHICLEVRM": "DEVICE",  # Vehicle registration
    
    # === CRYPTO ADDRESSES (PII - can identify) ===
    "BITCOINADDRESS": "CRYPTO",
    "LITECOINADDRESS": "CRYPTO",
    "ETHEREUMADDRESS": "CRYPTO",
    
    # === AGE (PII under HIPAA if >89) ===
    "AGE": "AGE",
    
    # === NOT PII - EXCLUDE FROM EVALUATION ===
    "PREFIX": None,  # Mr, Mrs, Dr - not PII
    "GENDER": None,
    "SEX": None,
    "EYECOLOR": None,
    "HEIGHT": None,
    "JOBTITLE": None,
    "JOBAREA": None,
    "JOBTYPE": None,
    "COMPANYNAME": None,  # Could be PII in context, but generally not
    "CURRENCYSYMBOL": None,
    "CURRENCYCODE": None,
    "CURRENCYNAME": None,
    "CURRENCY": None,
    "AMOUNT": None,
    "CREDITCARDISSUER": None,  # Visa, Mastercard - not PII
    "MASKEDNUMBER": None,  # Already masked
    "ACCOUNTNAME": None,  # Ambiguous - exclude
    "ORDINALDIRECTION": None,  # North, South - not PII
    "OTHER": None,  # Catch-all - exclude
}

# Canonical PII categories we evaluate
PII_CATEGORIES = {"NAME", "CONTACT", "DATE", "LOCATION", "IDNUM", "FINANCIAL", "DIGITAL", "DEVICE", "CRYPTO", "AGE"}


def map_to_canonical(raw_type: str) -> Optional[str]:
    """Map raw entity type to canonical PII category.
    
    Returns None if type should be excluded from evaluation.
    """
    # Try direct mapping
    canonical = TYPE_MAPPING.get(raw_type.upper())
    if canonical is not None:
        return canonical
    
    # Try with underscores removed
    canonical = TYPE_MAPPING.get(raw_type.upper().replace("_", ""))
    if canonical is not None:
        return canonical
    
    # Unknown type - exclude by default
    logger.debug(f"Unknown type '{raw_type}' - excluding from evaluation")
    return None


# =============================================================================
# NERVALUATE EVALUATION
# =============================================================================

@dataclass
class NervaluateEntity:
    """Entity in nervaluate format."""
    label: str
    start: int
    end: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "start": self.start, "end": self.end}


@dataclass 
class EvaluationMetrics:
    """Metrics for one evaluation mode."""
    correct: int = 0
    incorrect: int = 0
    partial: int = 0
    missed: int = 0
    spurious: int = 0
    
    @property
    def possible(self) -> int:
        """Number of gold-standard annotations."""
        return self.correct + self.incorrect + self.partial + self.missed
    
    @property
    def actual(self) -> int:
        """Number of predicted annotations."""
        return self.correct + self.incorrect + self.partial + self.spurious
    
    @property
    def precision(self) -> float:
        if self.actual == 0:
            return 0.0
        return self.correct / self.actual
    
    @property
    def recall(self) -> float:
        if self.possible == 0:
            return 0.0
        return self.correct / self.possible
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def precision_partial(self) -> float:
        """Precision with partial credit (0.5 for partial matches)."""
        if self.actual == 0:
            return 0.0
        return (self.correct + 0.5 * self.partial) / self.actual
    
    @property
    def recall_partial(self) -> float:
        """Recall with partial credit (0.5 for partial matches)."""
        if self.possible == 0:
            return 0.0
        return (self.correct + 0.5 * self.partial) / self.possible
    
    @property
    def f1_partial(self) -> float:
        """F1 with partial credit."""
        p, r = self.precision_partial, self.recall_partial
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class NervaluateResult:
    """Results in all four evaluation modes."""
    strict: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    exact: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    partial: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    type_: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    
    # Per-type breakdown
    by_type: Dict[str, "NervaluateResult"] = field(default_factory=dict)
    
    # Counts
    samples_evaluated: int = 0
    entities_in_ground_truth: int = 0
    entities_predicted: int = 0
    entities_excluded: int = 0  # Ground truth entities excluded (non-PII)


def spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two spans overlap."""
    return start1 < end2 and start2 < end1


def spans_match_exact(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two spans match exactly."""
    return start1 == start2 and end1 == end2


def evaluate_sample(
    true_entities: List[NervaluateEntity],
    pred_entities: List[NervaluateEntity],
) -> NervaluateResult:
    """
    Evaluate predictions against ground truth for one sample.
    
    Implements SemEval 2013 evaluation methodology:
    - strict: exact boundary AND exact type match
    - exact: exact boundary match, type ignored
    - partial: any overlap, type ignored  
    - type: any overlap AND exact type match
    """
    result = NervaluateResult()
    
    # Track which entities have been matched
    true_matched = [False] * len(true_entities)
    pred_matched = [False] * len(pred_entities)
    
    # First pass: find exact boundary matches
    for i, true_ent in enumerate(true_entities):
        for j, pred_ent in enumerate(pred_entities):
            if pred_matched[j]:
                continue
                
            if spans_match_exact(true_ent.start, true_ent.end, pred_ent.start, pred_ent.end):
                true_matched[i] = True
                pred_matched[j] = True
                
                type_match = (true_ent.label == pred_ent.label)
                
                # Strict: both boundary and type must match
                if type_match:
                    result.strict.correct += 1
                else:
                    result.strict.incorrect += 1
                
                # Exact: only boundary matters
                result.exact.correct += 1
                
                # Partial: overlap exists (exact is a special case of overlap)
                result.partial.correct += 1
                
                # Type: overlap + type match
                if type_match:
                    result.type_.correct += 1
                else:
                    result.type_.incorrect += 1
                
                break
    
    # Second pass: find partial matches for unmatched entities
    for i, true_ent in enumerate(true_entities):
        if true_matched[i]:
            continue
            
        for j, pred_ent in enumerate(pred_entities):
            if pred_matched[j]:
                continue
                
            if spans_overlap(true_ent.start, true_ent.end, pred_ent.start, pred_ent.end):
                true_matched[i] = True
                pred_matched[j] = True
                
                type_match = (true_ent.label == pred_ent.label)
                
                # Strict: partial boundary = incorrect
                result.strict.incorrect += 1
                
                # Exact: partial boundary = incorrect
                result.exact.incorrect += 1
                
                # Partial: overlap = partial credit
                result.partial.partial += 1
                
                # Type: overlap + type match
                if type_match:
                    result.type_.partial += 1
                else:
                    result.type_.incorrect += 1
                
                break
    
    # Count missed (ground truth not matched) and spurious (predictions not matched)
    for i, matched in enumerate(true_matched):
        if not matched:
            result.strict.missed += 1
            result.exact.missed += 1
            result.partial.missed += 1
            result.type_.missed += 1
    
    for j, matched in enumerate(pred_matched):
        if not matched:
            result.strict.spurious += 1
            result.exact.spurious += 1
            result.partial.spurious += 1
            result.type_.spurious += 1
    
    return result


def aggregate_results(results: List[NervaluateResult]) -> NervaluateResult:
    """Aggregate results from multiple samples."""
    agg = NervaluateResult()
    
    for r in results:
        agg.samples_evaluated += 1
        agg.entities_in_ground_truth += r.entities_in_ground_truth
        agg.entities_predicted += r.entities_predicted
        agg.entities_excluded += r.entities_excluded
        
        for mode in ["strict", "exact", "partial", "type_"]:
            agg_metrics = getattr(agg, mode)
            r_metrics = getattr(r, mode)
            agg_metrics.correct += r_metrics.correct
            agg_metrics.incorrect += r_metrics.incorrect
            agg_metrics.partial += r_metrics.partial
            agg_metrics.missed += r_metrics.missed
            agg_metrics.spurious += r_metrics.spurious
    
    return agg


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class NervaluateBenchmarkRunner:
    """Benchmark runner using nervaluate methodology."""
    
    def __init__(
        self,
        engine: Optional[ClassificationEngine] = None,
        storage: Optional[BenchmarkStorage] = None,
    ):
        self.engine = engine or ClassificationEngine()
        self.storage = storage
        
    def run(
        self,
        dataset_name: str = "ai4privacy",
        n_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> NervaluateResult:
        """
        Run benchmark evaluation.
        
        Args:
            dataset_name: Name of dataset to evaluate on
            n_samples: Number of samples (None for all)
            show_progress: Show progress bar
            
        Returns:
            NervaluateResult with metrics in all four modes
        """
        # Load dataset
        console.print(f"\n[bold]Loading {dataset_name} dataset...[/bold]")
        
        if dataset_name == "ai4privacy":
            dataset = load_ai4privacy_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        samples = list(dataset.samples)
        if n_samples:
            samples = samples[:n_samples]
        
        console.print(f"Evaluating on {len(samples)} samples\n")
        
        # Evaluate each sample
        sample_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(samples))
            
            for sample in samples:
                result = self._evaluate_sample(sample)
                sample_results.append(result)
                progress.advance(task)
        
        # Aggregate results
        final_result = aggregate_results(sample_results)
        
        # Print results
        self._print_results(final_result)
        
        return final_result
    
    def _evaluate_sample(self, sample: BenchmarkSample) -> NervaluateResult:
        """Evaluate a single sample."""
        # Convert ground truth to nervaluate format, filtering non-PII
        true_entities = []
        excluded_count = 0
        
        for ent in sample.entities:
            # Map to canonical type
            canonical = map_to_canonical(ent.entity_type)
            if canonical is None:
                excluded_count += 1
                continue
            
            true_entities.append(NervaluateEntity(
                label=canonical,
                start=ent.start,
                end=ent.end,
            ))
        
        # Run detection
        detected = self.engine.detect(sample.text)
        
        # Convert predictions to nervaluate format
        pred_entities = []
        for ent in detected:
            # Map our types to canonical
            canonical = map_to_canonical(ent.entity_type.name)
            if canonical is None:
                continue
            
            pred_entities.append(NervaluateEntity(
                label=canonical,
                start=ent.start,
                end=ent.end,
            ))
        
        # Evaluate
        result = evaluate_sample(true_entities, pred_entities)
        result.entities_in_ground_truth = len(true_entities)
        result.entities_predicted = len(pred_entities)
        result.entities_excluded = excluded_count
        
        return result
    
    def _print_results(self, result: NervaluateResult) -> None:
        """Print formatted results."""
        console.print("\n" + "=" * 70)
        console.print("[bold]NERVALUATE BENCHMARK RESULTS[/bold]")
        console.print("=" * 70)
        
        console.print(f"\nSamples evaluated: {result.samples_evaluated}")
        console.print(f"Ground truth entities (PII only): {result.entities_in_ground_truth}")
        console.print(f"Predicted entities (PII only): {result.entities_predicted}")
        console.print(f"Excluded entities (non-PII): {result.entities_excluded}")
        
        # Main results table
        table = Table(title="\nEvaluation Modes", box=box.ROUNDED)
        table.add_column("Mode", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right", style="bold")
        table.add_column("Correct", justify="right")
        table.add_column("Incorrect", justify="right")
        table.add_column("Partial", justify="right")
        table.add_column("Missed", justify="right")
        table.add_column("Spurious", justify="right")
        
        modes = [
            ("Strict", result.strict),
            ("Exact", result.exact),
            ("Partial", result.partial),
            ("Type", result.type_),
        ]
        
        for name, metrics in modes:
            # Use partial credit for partial mode
            if name == "Partial":
                p, r, f1 = metrics.precision_partial, metrics.recall_partial, metrics.f1_partial
            else:
                p, r, f1 = metrics.precision, metrics.recall, metrics.f1
            
            table.add_row(
                name,
                f"{p:.1%}",
                f"{r:.1%}",
                f"{f1:.1%}",
                str(metrics.correct),
                str(metrics.incorrect),
                str(metrics.partial),
                str(metrics.missed),
                str(metrics.spurious),
            )
        
        console.print(table)
        
        # Explanation
        console.print("\n[dim]Mode definitions:[/dim]")
        console.print("[dim]  Strict  = Exact boundary AND exact type match[/dim]")
        console.print("[dim]  Exact   = Exact boundary match (type ignored)[/dim]")
        console.print("[dim]  Partial = Any overlap (type ignored, 0.5 credit for partial)[/dim]")
        console.print("[dim]  Type    = Any overlap AND exact type match[/dim]")
        
        # Recommendation
        console.print("\n[bold yellow]Recommended metrics:[/bold yellow]")
        console.print(f"  • For marketing/comparison: [bold]Partial F1 = {result.partial.f1_partial:.1%}[/bold]")
        console.print(f"  • For internal improvement: [bold]Strict F1 = {result.strict.f1:.1%}[/bold]")


def run_nervaluate_benchmark(
    dataset: str = "ai4privacy",
    n_samples: Optional[int] = None,
) -> NervaluateResult:
    """
    Convenience function to run benchmark.
    
    Args:
        dataset: Dataset name ("ai4privacy")
        n_samples: Number of samples (None for all)
        
    Returns:
        NervaluateResult with all metrics
    """
    runner = NervaluateBenchmarkRunner()
    return runner.run(dataset_name=dataset, n_samples=n_samples)


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def add_nervaluate_command(app):
    """Add nervaluate command to CLI app."""
    import typer
    
    @app.command()
    def nervaluate(
        dataset: str = typer.Argument("ai4privacy", help="Dataset to evaluate on"),
        n: int = typer.Option(None, "-n", "--samples", help="Number of samples"),
    ):
        """Run benchmark with nervaluate methodology (SemEval 2013)."""
        run_nervaluate_benchmark(dataset=dataset, n_samples=n)
