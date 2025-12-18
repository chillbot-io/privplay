#!/usr/bin/env python3
"""
Merge multiple PII datasets into one mega training set.

Combines:
- Golden dataset (curated AI4Privacy)
- Synthetic PII (diverse, perfect labels)
- Hard negatives (what ISN'T PII)

Usage:
    python3 merge_datasets.py --output mega_dataset.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import Counter

from rich.console import Console
from rich.table import Table

console = Console()


def load_dataset(path: str) -> List[Dict]:
    """Load a dataset file."""
    with open(path) as f:
        data = json.load(f)
    
    samples = data.get("samples", data)
    return samples


def merge_datasets(
    golden_path: str = "golden_dataset.json",
    synthetic_path: str = "synthetic_pii.json",
    negatives_path: str = "hard_negatives.json",
    output_path: str = "mega_dataset.json",
):
    """Merge all datasets."""
    console.print(f"\n[bold cyan]═══ Merging Datasets ═══[/bold cyan]\n")
    
    all_samples = []
    sources = {}
    
    # Load golden dataset
    if Path(golden_path).exists():
        golden = load_dataset(golden_path)
        for s in golden:
            s["_source"] = "golden"
        all_samples.extend(golden)
        sources["golden"] = len(golden)
        console.print(f"[green]✓[/green] Loaded {len(golden):,} samples from {golden_path}")
    else:
        console.print(f"[yellow]![/yellow] {golden_path} not found, skipping")
    
    # Load synthetic dataset
    if Path(synthetic_path).exists():
        synthetic = load_dataset(synthetic_path)
        for s in synthetic:
            s["_source"] = "synthetic"
        all_samples.extend(synthetic)
        sources["synthetic"] = len(synthetic)
        console.print(f"[green]✓[/green] Loaded {len(synthetic):,} samples from {synthetic_path}")
    else:
        console.print(f"[yellow]![/yellow] {synthetic_path} not found, skipping")
    
    # Load hard negatives
    if Path(negatives_path).exists():
        negatives = load_dataset(negatives_path)
        for s in negatives:
            s["_source"] = "hard_negative"
        all_samples.extend(negatives)
        sources["hard_negatives"] = len(negatives)
        console.print(f"[green]✓[/green] Loaded {len(negatives):,} samples from {negatives_path}")
    else:
        console.print(f"[yellow]![/yellow] {negatives_path} not found, skipping")
    
    if not all_samples:
        console.print("[red]No datasets found![/red]")
        return
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Count entity types
    type_counts = Counter()
    positive_samples = 0
    negative_samples = 0
    
    for sample in all_samples:
        entities = sample.get("entities", [])
        if entities:
            positive_samples += 1
            for e in entities:
                label = e.get("label", "UNKNOWN")
                type_counts[label] += 1
        else:
            negative_samples += 1
    
    # Save
    output = {
        "samples": all_samples,
        "metadata": {
            "total_samples": len(all_samples),
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "sources": sources,
            "entity_type_counts": dict(type_counts),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f)
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    
    table = Table()
    table.add_column("Source")
    table.add_column("Samples", justify="right")
    
    for source, count in sources.items():
        table.add_row(source, f"{count:,}")
    table.add_row("─" * 15, "─" * 8)
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(all_samples):,}[/bold]")
    
    console.print(table)
    
    console.print(f"\n[bold]Composition:[/bold]")
    console.print(f"  Positive samples (with PII): {positive_samples:,}")
    console.print(f"  Negative samples (no PII):   {negative_samples:,}")
    console.print(f"  Negative ratio: {negative_samples / len(all_samples) * 100:.1f}%")
    
    console.print(f"\n[bold]Entity Types:[/bold]")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        console.print(f"  {etype:20} {count:,}")
    
    console.print(f"\n[green]✓ Saved {len(all_samples):,} samples to {output_path}[/green]")
    console.print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="golden_dataset.json")
    parser.add_argument("--synthetic", type=str, default="synthetic_pii.json")
    parser.add_argument("--negatives", type=str, default="hard_negatives.json")
    parser.add_argument("--output", type=str, default="mega_dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    merge_datasets(
        golden_path=args.golden,
        synthetic_path=args.synthetic,
        negatives_path=args.negatives,
        output_path=args.output,
    )
