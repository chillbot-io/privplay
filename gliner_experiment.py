#!/usr/bin/env python3
"""
GLiNER Experiment: Zero-shot NER vs fine-tuned PII-BERT

Compares GLiNER (zero-shot) against current PII-BERT on AI4Privacy benchmark.
No training needed - GLiNER uses natural language entity descriptions.

Usage:
    pip install gliner
    python gliner_experiment.py --samples 100
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import time

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()


# GLiNER entity labels (natural language descriptions work best)
GLINER_LABELS = [
    "person name",
    "email address",
    "phone number",
    "social security number",
    "credit card number",
    "username",
    "password",
    "street address",
    "date",
    "IP address",
    "bank account",
    "url",
]

# Map GLiNER labels to our EntityType values
GLINER_TO_ENTITY_TYPE = {
    "person name": "NAME_PERSON",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "social security number": "SSN",
    "credit card number": "CREDIT_CARD",
    "username": "USERNAME",
    "password": "PASSWORD",
    "street address": "ADDRESS",
    "date": "DATE",
    "IP address": "IP_ADDRESS",
    "bank account": "ACCOUNT_NUMBER",
    "url": "URL",
}

# Map AI4Privacy labels to our normalized types
AI4PRIVACY_TO_ENTITY_TYPE = {
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONENUMBER": "PHONE",
    "SSN": "SSN",
    "SOCIALNUM": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "USERNAME": "USERNAME",
    "USERAGENT": "USERNAME",
    "PASSWORD": "PASSWORD",
    "PIN": "PASSWORD",
    "ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "DATE": "DATE",
    "DOB": "DATE",
    "TIME": "DATE",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IBAN": "ACCOUNT_NUMBER",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "ACCOUNTNAME": "ACCOUNT_NUMBER",
    "BIC": "ACCOUNT_NUMBER",
    "URL": "URL",
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON",
    "MIDDLENAME": "NAME_PERSON",
    "NAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "SUFFIX": "NAME_PERSON",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "COUNTRY": "LOCATION",
    "ZIPCODE": "ZIP",
    "POSTCODE": "ZIP",
    "COMPANYNAME": "ORGANIZATION",
    "JOBAREA": "OTHER",
    "JOBTITLE": "OTHER",
    "JOBTYPE": "OTHER",
    "GENDER": "OTHER",
    "SEX": "OTHER",
    "VEHICLEVIN": "DEVICE_ID",
    "VEHICLEVRM": "DEVICE_ID",
    "MAC": "MAC_ADDRESS",
    "IMEI": "DEVICE_ID",
    "CURRENCY": "OTHER",
    "AMOUNT": "OTHER",
    "LITECOINADDRESS": "CRYPTO_ADDRESS",
    "BITCOINADDRESS": "CRYPTO_ADDRESS",
    "ETHEREUMADDRESS": "CRYPTO_ADDRESS",
    "MASKEDNUMBER": "OTHER",
    "ORDINALDIRECTION": "OTHER",
    "NEARBYGPSCOORDINATE": "GPS_COORDINATE",
}


def normalize_ai4privacy_type(label: str) -> str:
    """Normalize AI4Privacy label to our entity type."""
    label_upper = label.upper().replace(" ", "").replace("_", "")
    
    # Try direct match first
    if label_upper in AI4PRIVACY_TO_ENTITY_TYPE:
        return AI4PRIVACY_TO_ENTITY_TYPE[label_upper]
    
    # Try with original
    if label.upper() in AI4PRIVACY_TO_ENTITY_TYPE:
        return AI4PRIVACY_TO_ENTITY_TYPE[label.upper()]
    
    return "OTHER"


@dataclass
class Detection:
    """A detected entity."""
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    source: str  # "gliner" or "pii_bert"


@dataclass 
class GroundTruth:
    """Ground truth entity from benchmark."""
    text: str
    start: int
    end: int
    entity_type: str


def spans_overlap(start1: int, end1: int, start2: int, end2: int, threshold: float = 0.5) -> bool:
    """Check if two spans overlap by at least threshold."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return False
    
    overlap_len = overlap_end - overlap_start
    min_len = min(end1 - start1, end2 - start2)
    
    return overlap_len / min_len >= threshold


def run_gliner_experiment(
    num_samples: int = 100,
    model_name: str = "urchade/gliner_base",
    compare_pii_bert: bool = True,
):
    """Run GLiNER vs PII-BERT comparison."""
    
    console.print(f"\n[bold cyan]═══ GLiNER Experiment ═══[/bold cyan]\n")
    
    # Load GLiNER
    console.print(f"Loading GLiNER model: {model_name}...")
    try:
        from gliner import GLiNER
        gliner = GLiNER.from_pretrained(model_name)
        console.print("[green]✓ GLiNER loaded[/green]")
    except ImportError:
        console.print("[red]GLiNER not installed. Run: pip install gliner[/red]")
        return
    except Exception as e:
        console.print(f"[red]Failed to load GLiNER: {e}[/red]")
        return
    
    # Load PII-BERT for comparison
    pii_bert = None
    if compare_pii_bert:
        console.print("Loading PII-BERT for comparison...")
        try:
            from privplay.engine.models.pii_transformer import get_pii_model
            pii_bert = get_pii_model()
            if pii_bert and pii_bert.is_available():
                console.print("[green]✓ PII-BERT loaded[/green]")
            else:
                console.print("[yellow]PII-BERT not available, skipping comparison[/yellow]")
                pii_bert = None
        except Exception as e:
            console.print(f"[yellow]Failed to load PII-BERT: {e}[/yellow]")
    
    # Load AI4Privacy benchmark
    console.print(f"\nLoading AI4Privacy dataset ({num_samples} samples)...")
    try:
        from privplay.benchmark.datasets import load_ai4privacy_dataset
        dataset = load_ai4privacy_dataset(max_samples=num_samples)
        console.print(f"[green]✓ Loaded {len(dataset.samples)} samples[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    # Run experiment
    gliner_results = {"tp": 0, "fp": 0, "fn": 0, "by_type": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    pii_bert_results = {"tp": 0, "fp": 0, "fn": 0, "by_type": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    
    gliner_time = 0
    pii_bert_time = 0
    
    console.print(f"\nRunning detection on {len(dataset.samples)} samples...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(dataset.samples))
        
        for sample in dataset.samples:
            text = sample.text
            
            # Build ground truth
            ground_truths = []
            for entity in sample.entities:
                gt_type = normalize_ai4privacy_type(entity.entity_type)
                ground_truths.append(GroundTruth(
                    text=entity.text,
                    start=entity.start,
                    end=entity.end,
                    entity_type=gt_type,
                ))
            
            # Run GLiNER
            start_time = time.time()
            try:
                gliner_entities = gliner.predict_entities(text, GLINER_LABELS, threshold=0.3)
            except Exception as e:
                gliner_entities = []
            gliner_time += time.time() - start_time
            
            gliner_detections = []
            for ent in gliner_entities:
                mapped_type = GLINER_TO_ENTITY_TYPE.get(ent["label"], "OTHER")
                gliner_detections.append(Detection(
                    text=ent["text"],
                    start=ent["start"],
                    end=ent["end"],
                    entity_type=mapped_type,
                    confidence=ent["score"],
                    source="gliner",
                ))
            
            # Run PII-BERT
            pii_bert_detections = []
            if pii_bert:
                start_time = time.time()
                try:
                    pii_entities = pii_bert.detect(text)
                    for ent in pii_entities:
                        ent_type = ent.entity_type.value if hasattr(ent.entity_type, 'value') else str(ent.entity_type)
                        pii_bert_detections.append(Detection(
                            text=ent.text,
                            start=ent.start,
                            end=ent.end,
                            entity_type=ent_type,
                            confidence=ent.confidence,
                            source="pii_bert",
                        ))
                except Exception as e:
                    pass
                pii_bert_time += time.time() - start_time
            
            # Score GLiNER
            score_detections(gliner_detections, ground_truths, gliner_results)
            
            # Score PII-BERT
            if pii_bert:
                score_detections(pii_bert_detections, ground_truths, pii_bert_results)
            
            progress.advance(task)
    
    # Print results
    print_results("GLiNER", gliner_results, gliner_time)
    
    if pii_bert:
        print_results("PII-BERT", pii_bert_results, pii_bert_time)
        print_comparison(gliner_results, pii_bert_results)


def score_detections(
    detections: List[Detection],
    ground_truths: List[GroundTruth],
    results: Dict,
):
    """Score detections against ground truth."""
    matched_gt = set()
    matched_det = set()
    
    # Find true positives (detection matches ground truth)
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            
            # Check overlap
            if spans_overlap(det.start, det.end, gt.start, gt.end, threshold=0.5):
                # Type match (flexible - both map to same category)
                det_type = det.entity_type
                gt_type = gt.entity_type
                
                # Consider it a match if types are compatible
                type_match = (
                    det_type == gt_type or
                    (det_type in ("NAME_PERSON", "NAME") and gt_type in ("NAME_PERSON", "NAME")) or
                    (det_type in ("PHONE", "FAX") and gt_type in ("PHONE", "FAX")) or
                    (det_type in ("DATE", "DATE_DOB") and gt_type in ("DATE", "DATE_DOB"))
                )
                
                if type_match:
                    results["tp"] += 1
                    results["by_type"][gt_type]["tp"] += 1
                    matched_gt.add(j)
                    matched_det.add(i)
                    break
    
    # False positives (detections that didn't match any ground truth)
    for i, det in enumerate(detections):
        if i not in matched_det:
            results["fp"] += 1
            results["by_type"][det.entity_type]["fp"] += 1
    
    # False negatives (ground truths that weren't detected)
    for j, gt in enumerate(ground_truths):
        if j not in matched_gt:
            results["fn"] += 1
            results["by_type"][gt.entity_type]["fn"] += 1


def print_results(name: str, results: Dict, elapsed_time: float):
    """Print results for a model."""
    tp, fp, fn = results["tp"], results["fp"], results["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    console.print(f"\n[bold cyan]═══ {name} Results ═══[/bold cyan]")
    console.print(f"F1: {f1:.1%} | Precision: {precision:.1%} | Recall: {recall:.1%}")
    console.print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    console.print(f"Time: {elapsed_time:.1f}s ({elapsed_time/max(1, tp+fp+fn)*1000:.1f}ms per entity)")
    
    # Per-type breakdown
    console.print(f"\n[bold]Per-Type Performance:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("Type")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    
    by_type = results["by_type"]
    for etype in sorted(by_type.keys()):
        counts = by_type[etype]
        t_tp, t_fp, t_fn = counts["tp"], counts["fp"], counts["fn"]
        
        if t_tp + t_fp + t_fn == 0:
            continue
        
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        
        table.add_row(
            etype,
            str(t_tp),
            str(t_fp),
            str(t_fn),
            f"{t_prec:.0%}",
            f"{t_rec:.0%}",
            f"{t_f1:.0%}",
        )
    
    console.print(table)


def print_comparison(gliner_results: Dict, pii_bert_results: Dict):
    """Print head-to-head comparison."""
    console.print(f"\n[bold cyan]═══ Head-to-Head Comparison ═══[/bold cyan]")
    
    table = Table(box=box.SIMPLE)
    table.add_column("Metric")
    table.add_column("GLiNER", justify="right")
    table.add_column("PII-BERT", justify="right")
    table.add_column("Winner", justify="center")
    
    def calc_metrics(r):
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1
    
    g_prec, g_rec, g_f1 = calc_metrics(gliner_results)
    p_prec, p_rec, p_f1 = calc_metrics(pii_bert_results)
    
    def winner(g, p):
        if abs(g - p) < 0.01:
            return "TIE"
        return "[green]GLiNER[/green]" if g > p else "[blue]PII-BERT[/blue]"
    
    table.add_row("F1 Score", f"{g_f1:.1%}", f"{p_f1:.1%}", winner(g_f1, p_f1))
    table.add_row("Precision", f"{g_prec:.1%}", f"{p_prec:.1%}", winner(g_prec, p_prec))
    table.add_row("Recall", f"{g_rec:.1%}", f"{p_rec:.1%}", winner(g_rec, p_rec))
    
    console.print(table)
    
    # Per-type winners
    console.print(f"\n[bold]Per-Type Winners (by F1):[/bold]")
    
    all_types = set(gliner_results["by_type"].keys()) | set(pii_bert_results["by_type"].keys())
    
    gliner_wins = []
    pii_bert_wins = []
    ties = []
    
    for etype in sorted(all_types):
        g_counts = gliner_results["by_type"].get(etype, {"tp": 0, "fp": 0, "fn": 0})
        p_counts = pii_bert_results["by_type"].get(etype, {"tp": 0, "fp": 0, "fn": 0})
        
        g_tp, g_fp, g_fn = g_counts["tp"], g_counts["fp"], g_counts["fn"]
        p_tp, p_fp, p_fn = p_counts["tp"], p_counts["fp"], p_counts["fn"]
        
        if g_tp + g_fp + g_fn + p_tp + p_fp + p_fn == 0:
            continue
        
        g_f1 = 2 * g_tp / (2 * g_tp + g_fp + g_fn) if (2 * g_tp + g_fp + g_fn) > 0 else 0
        p_f1 = 2 * p_tp / (2 * p_tp + p_fp + p_fn) if (2 * p_tp + p_fp + p_fn) > 0 else 0
        
        if abs(g_f1 - p_f1) < 0.05:
            ties.append(f"{etype} ({g_f1:.0%})")
        elif g_f1 > p_f1:
            gliner_wins.append(f"{etype} ({g_f1:.0%} vs {p_f1:.0%})")
        else:
            pii_bert_wins.append(f"{etype} ({p_f1:.0%} vs {g_f1:.0%})")
    
    console.print(f"  [green]GLiNER wins:[/green] {', '.join(gliner_wins) if gliner_wins else 'None'}")
    console.print(f"  [blue]PII-BERT wins:[/blue] {', '.join(pii_bert_wins) if pii_bert_wins else 'None'}")
    console.print(f"  [yellow]Ties:[/yellow] {', '.join(ties) if ties else 'None'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GLiNER vs PII-BERT experiment")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--model", type=str, default="urchade/gliner_base", 
                       help="GLiNER model (gliner_small, gliner_base, gliner_large)")
    parser.add_argument("--no-compare", action="store_true", help="Skip PII-BERT comparison")
    
    args = parser.parse_args()
    
    run_gliner_experiment(
        num_samples=args.samples,
        model_name=args.model,
        compare_pii_bert=not args.no_compare,
    )
