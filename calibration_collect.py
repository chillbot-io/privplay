#!/usr/bin/env python3
"""
Step 1: Collect calibration data for Platt scaling.

Runs benchmark samples through PHI-BERT and PII-BERT, recording:
- Raw confidence scores from each model
- Ground truth (was this actually PII? what type?)

Output: calibration_data.json with structure:
{
    "phi_bert": [{"raw_score": 0.92, "actual": 1, "type": "NAME_PATIENT"}, ...],
    "pii_bert": [{"raw_score": 0.75, "actual": 0, "type": "USERNAME"}, ...],
    "metadata": {"samples": 1000, "collected_at": "..."}
}
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def collect_calibration_data(num_samples: int = 1000, output_path: str = "calibration_data.json"):
    """Collect BERT scores matched against ground truth."""
    
    console.print(f"\n[bold cyan]═══ Calibration Data Collection ═══[/bold cyan]\n")
    
    # Import here to avoid slow startup
    from privplay.benchmark.datasets import load_ai4privacy_dataset
    from privplay.engine.classifier import ClassificationEngine
    
    console.print(f"Loading {num_samples} samples...")
    dataset = load_ai4privacy_dataset(max_samples=num_samples)
    
    console.print("Initializing engine with signal capture...")
    engine = ClassificationEngine(capture_signals=True, use_coreference=False)
    
    # Storage for calibration data
    phi_bert_data = []
    pii_bert_data = []
    
    # Also track by type for analysis
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    console.print(f"\nProcessing {len(dataset.samples)} samples...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting signals...", total=len(dataset.samples))
        
        for sample in dataset.samples:
            # Build ground truth lookup for this sample
            # Key: (start, end) with fuzzy matching
            gt_spans = {}
            for entity in sample.entities:
                # Allow +/- 2 char fuzzy match
                for offset in range(-2, 3):
                    gt_spans[(entity.start + offset, entity.end + offset)] = {
                        "type": entity.normalized_type,
                        "text": entity.text,
                    }
            
            # Run detection with signal capture
            engine.clear_captured_signals()
            try:
                engine.detect(sample.text, verify=False)
            except Exception as e:
                progress.advance(task)
                continue
            
            signals = engine.get_captured_signals()
            
            for signal in signals:
                span_key = (signal.span_start, signal.span_end)
                
                # Check if this span matches ground truth
                matched_gt = None
                is_correct = False
                
                if span_key in gt_spans:
                    matched_gt = gt_spans[span_key]
                    is_correct = True
                else:
                    # Try overlap matching (50% threshold)
                    for (gt_start, gt_end), gt_info in gt_spans.items():
                        overlap_start = max(signal.span_start, gt_start)
                        overlap_end = min(signal.span_end, gt_end)
                        
                        if overlap_start < overlap_end:
                            overlap_len = overlap_end - overlap_start
                            signal_len = signal.span_end - signal.span_start
                            gt_len = gt_end - gt_start
                            
                            if overlap_len / max(signal_len, gt_len) > 0.5:
                                matched_gt = gt_info
                                is_correct = True
                                break
                
                # Record PHI-BERT data
                if signal.phi_bert_detected:
                    phi_bert_data.append({
                        "raw_score": signal.phi_bert_conf,
                        "actual": 1 if is_correct else 0,
                        "predicted_type": signal.phi_bert_type,
                        "true_type": matched_gt["type"] if matched_gt else None,
                        "text": signal.span_text[:50],
                    })
                    
                    if is_correct:
                        type_stats[signal.phi_bert_type]["tp"] += 1
                    else:
                        type_stats[signal.phi_bert_type]["fp"] += 1
                
                # Record PII-BERT data
                if signal.pii_bert_detected:
                    pii_bert_data.append({
                        "raw_score": signal.pii_bert_conf,
                        "actual": 1 if is_correct else 0,
                        "predicted_type": signal.pii_bert_type,
                        "true_type": matched_gt["type"] if matched_gt else None,
                        "text": signal.span_text[:50],
                    })
                    
                    if is_correct:
                        type_stats[signal.pii_bert_type]["tp"] += 1
                    else:
                        type_stats[signal.pii_bert_type]["fp"] += 1
            
            progress.advance(task)
    
    # Save calibration data
    output = {
        "phi_bert": phi_bert_data,
        "pii_bert": pii_bert_data,
        "metadata": {
            "samples": num_samples,
            "phi_bert_signals": len(phi_bert_data),
            "pii_bert_signals": len(pii_bert_data),
            "collected_at": datetime.now().isoformat(),
        },
        "type_stats": dict(type_stats),
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    console.print(f"\n[bold green]═══ Collection Complete ═══[/bold green]\n")
    
    console.print(f"PHI-BERT signals: {len(phi_bert_data)}")
    phi_correct = sum(1 for d in phi_bert_data if d["actual"] == 1)
    console.print(f"  Correct: {phi_correct} ({phi_correct/len(phi_bert_data)*100:.1f}%)" if phi_bert_data else "  No data")
    
    console.print(f"\nPII-BERT signals: {len(pii_bert_data)}")
    pii_correct = sum(1 for d in pii_bert_data if d["actual"] == 1)
    console.print(f"  Correct: {pii_correct} ({pii_correct/len(pii_bert_data)*100:.1f}%)" if pii_bert_data else "  No data")
    
    # Show score distribution
    console.print(f"\n[bold]Score Distribution (PHI-BERT):[/bold]")
    if phi_bert_data:
        for bucket in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            in_bucket = [d for d in phi_bert_data if bucket[0] <= d["raw_score"] < bucket[1]]
            if in_bucket:
                correct = sum(1 for d in in_bucket if d["actual"] == 1)
                console.print(f"  {bucket[0]:.1f}-{bucket[1]:.1f}: {len(in_bucket)} signals, {correct/len(in_bucket)*100:.1f}% correct")
    
    console.print(f"\n[bold]Score Distribution (PII-BERT):[/bold]")
    if pii_bert_data:
        for bucket in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            in_bucket = [d for d in pii_bert_data if bucket[0] <= d["raw_score"] < bucket[1]]
            if in_bucket:
                correct = sum(1 for d in in_bucket if d["actual"] == 1)
                console.print(f"  {bucket[0]:.1f}-{bucket[1]:.1f}: {len(in_bucket)} signals, {correct/len(in_bucket)*100:.1f}% correct")
    
    console.print(f"\n[green]✓ Saved to {output_path}[/green]\n")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect calibration data for Platt scaling")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="calibration_data.json", help="Output file")
    
    args = parser.parse_args()
    
    collect_calibration_data(num_samples=args.samples, output_path=args.output)
