#!/usr/bin/env python3
"""
Benchmark Review Tool - Review FPs and FNs to create golden training signals.

Usage:
    python review_benchmark.py --samples 1000
    
Workflow:
    1. Runs AI4Privacy benchmark
    2. Collects all FPs and FNs
    3. Interactive review of each
    4. Saves golden signals for meta-classifier training
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()


@dataclass
class GoldenSignal:
    """A human-reviewed signal for training."""
    id: str
    span_text: str
    span_start: int
    span_end: int
    
    # What the system detected
    detected_type: Optional[str]
    detected_conf: float
    
    # Ground truth from benchmark
    benchmark_type: Optional[str]
    
    # Human decision
    human_label: str  # "TP", "FP", "FN", "SKIP"
    correct_type: Optional[str]  # For FNs or wrong labels
    
    # Context
    context_before: str
    context_after: str
    full_text: str
    
    # Detector signals (for training features)
    phi_bert_detected: bool = False
    phi_bert_conf: float = 0.0
    pii_bert_detected: bool = False
    pii_bert_conf: float = 0.0
    presidio_detected: bool = False
    presidio_conf: float = 0.0
    rule_detected: bool = False
    rule_conf: float = 0.0
    rule_has_checksum: bool = False
    
    # Metadata
    sample_id: str = ""
    reviewed_at: str = ""


class BenchmarkReviewer:
    """Interactive reviewer for benchmark FPs and FNs."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("golden_signals")
        self.output_dir.mkdir(exist_ok=True)
        
        self.signals: List[GoldenSignal] = []
        self.fps_reviewed = 0
        self.fns_reviewed = 0
        self.skipped = 0
    
    def run_benchmark(self, samples: int = 1000) -> Dict[str, Any]:
        """Run AI4Privacy benchmark and collect results."""
        console.print()
        console.print("[bold cyan]═══ Running AI4Privacy Benchmark ═══[/bold cyan]")
        console.print()
        
        from privplay.benchmark.datasets import load_ai4privacy_dataset
        from privplay.benchmark.runner import BenchmarkRunner
        from privplay.engine.classifier import ClassificationEngine
        
        # Load dataset
        console.print(f"Loading AI4Privacy dataset ({samples} samples)...")
        dataset = load_ai4privacy_dataset(max_samples=samples)
        
        # Create engine with signal capture
        console.print("Initializing detection engine...")
        engine = ClassificationEngine(
            capture_signals=True,
            use_coreference=False,  # Faster
        )
        
        # Run benchmark
        console.print("Running benchmark...")
        runner = BenchmarkRunner(engine)
        result = runner.run(dataset, verify=False, show_progress=True)
        
        # Summary
        console.print()
        console.print(f"[green]✓ Benchmark complete[/green]")
        console.print(f"  F1: {result.f1:.1%}")
        console.print(f"  Precision: {result.precision:.1%}")
        console.print(f"  Recall: {result.recall:.1%}")
        console.print(f"  TP: {result.true_positives} | FP: {result.false_positives} | FN: {result.false_negatives}")
        
        return {
            "result": result,
            "dataset": dataset,
            "engine": engine,
        }
    
    def collect_errors(self, result, dataset, engine) -> tuple:
        """Collect FPs and FNs from benchmark results."""
        console.print()
        console.print("Collecting FPs and FNs...")
        
        fps = []  # (sample, detected_entity, signals)
        fns = []  # (sample, ground_truth_entity)
        
        sample_lookup = {s.id: s for s in dataset.samples}
        
        for sample_result in result.sample_results:
            sample = sample_lookup.get(sample_result.sample_id)
            if not sample:
                continue
            
            # Re-run detection to get signals
            engine.clear_captured_signals()
            detected = engine.detect(sample.text, verify=False)
            signals = engine.get_captured_signals()
            
            # Build signal lookup by span
            signal_lookup = {}
            for sig in signals:
                signal_lookup[(sig.span_start, sig.span_end)] = sig
            
            for match in sample_result.matches:
                if match.match_type == "false_positive":
                    # FP: detected something that wasn't in ground truth
                    det = match.detected
                    sig = signal_lookup.get((det.start, det.end))
                    fps.append((sample, det, sig))
                    
                elif match.match_type == "false_negative":
                    # FN: missed something in ground truth
                    gt = match.ground_truth
                    fns.append((sample, gt))
        
        console.print(f"  Found {len(fps)} FPs and {len(fns)} FNs")
        
        return fps, fns
    
    def review_fps(self, fps: list):
        """Interactive review of false positives."""
        if not fps:
            console.print("[green]No FPs to review![/green]")
            return
        
        console.print()
        console.print(f"[bold cyan]═══ Reviewing {len(fps)} False Positives ═══[/bold cyan]")
        console.print()
        console.print("[dim]For each detection, decide if it's really a FP or if benchmark was wrong[/dim]")
        console.print()
        console.print("  [bold green][tp][/] Actually correct (benchmark missed it)")
        console.print("  [bold red][fp][/] Confirm false positive")
        console.print("  [bold yellow][wl][/] Detection correct but wrong type")
        console.print("  [dim][s][/]  Skip")
        console.print("  [dim][q][/]  Quit review")
        console.print()
        
        for i, (sample, detected, signals) in enumerate(fps, 1):
            if not self._review_fp(i, len(fps), sample, detected, signals):
                break
    
    def _review_fp(self, index: int, total: int, sample, detected, signals) -> bool:
        """Review a single FP. Returns False to quit."""
        console.print("─" * 70)
        console.print(f"[bold]FP {index}/{total}[/bold]")
        console.print()
        
        # Show context
        ctx_start = max(0, detected.start - 60)
        ctx_end = min(len(sample.text), detected.end + 60)
        
        before = sample.text[ctx_start:detected.start]
        span = sample.text[detected.start:detected.end]
        after = sample.text[detected.end:ctx_end]
        
        console.print(f"  {before}[bold yellow on dark_red]{span}[/]{after}")
        console.print()
        console.print(f"  Detected: [bold]{span}[/] as [cyan]{detected.entity_type.value}[/] ({detected.confidence:.0%})")
        
        # Show detector breakdown if available
        if signals:
            detectors = []
            if signals.phi_bert_detected:
                detectors.append(f"PHI-BERT:{signals.phi_bert_conf:.0%}")
            if signals.pii_bert_detected:
                detectors.append(f"PII-BERT:{signals.pii_bert_conf:.0%}")
            if signals.presidio_detected:
                detectors.append(f"Presidio:{signals.presidio_conf:.0%}")
            if signals.rule_detected:
                chk = "✓" if signals.rule_has_checksum else ""
                detectors.append(f"Rule:{signals.rule_conf:.0%}{chk}")
            if detectors:
                console.print(f"  [dim]Detectors: {', '.join(detectors)}[/dim]")
        
        console.print()
        
        choice = Prompt.ask("Decision", choices=["tp", "fp", "wl", "s", "q"], default="fp")
        
        if choice == "q":
            return False
        
        if choice == "s":
            self.skipped += 1
            return True
        
        # Create golden signal
        signal = GoldenSignal(
            id=f"fp_{index}_{datetime.now().strftime('%H%M%S')}",
            span_text=span,
            span_start=detected.start,
            span_end=detected.end,
            detected_type=detected.entity_type.value,
            detected_conf=detected.confidence,
            benchmark_type=None,  # FP means no ground truth
            human_label="TP" if choice == "tp" else "FP" if choice == "fp" else "WL",
            correct_type=None,
            context_before=before,
            context_after=after,
            full_text=sample.text,
            sample_id=sample.id,
            reviewed_at=datetime.now().isoformat(),
        )
        
        # Copy detector signals if available
        if signals:
            signal.phi_bert_detected = signals.phi_bert_detected
            signal.phi_bert_conf = signals.phi_bert_conf
            signal.pii_bert_detected = signals.pii_bert_detected
            signal.pii_bert_conf = signals.pii_bert_conf
            signal.presidio_detected = signals.presidio_detected
            signal.presidio_conf = signals.presidio_conf
            signal.rule_detected = signals.rule_detected
            signal.rule_conf = signals.rule_conf
            signal.rule_has_checksum = signals.rule_has_checksum
        
        # If wrong label, get correct type
        if choice == "wl":
            correct = self._get_entity_type()
            if correct:
                signal.correct_type = correct
                signal.human_label = "WL"
        
        self.signals.append(signal)
        self.fps_reviewed += 1
        
        console.print(f"  [green]✓ Labeled as {signal.human_label}[/green]")
        console.print()
        
        return True
    
    def review_fns(self, fns: list):
        """Interactive review of false negatives."""
        if not fns:
            console.print("[green]No FNs to review![/green]")
            return
        
        console.print()
        console.print(f"[bold cyan]═══ Reviewing {len(fns)} False Negatives ═══[/bold cyan]")
        console.print()
        console.print("[dim]For each missed entity, confirm if we should have detected it[/dim]")
        console.print()
        console.print("  [bold green][fn][/] Confirm - we should detect this")
        console.print("  [bold red][ok][/] Actually OK to miss (not really PHI)")
        console.print("  [dim][s][/]  Skip")
        console.print("  [dim][q][/]  Quit review")
        console.print()
        
        for i, (sample, gt) in enumerate(fns, 1):
            if not self._review_fn(i, len(fns), sample, gt):
                break
    
    def _review_fn(self, index: int, total: int, sample, gt) -> bool:
        """Review a single FN. Returns False to quit."""
        console.print("─" * 70)
        console.print(f"[bold]FN {index}/{total}[/bold]")
        console.print()
        
        # Show context
        ctx_start = max(0, gt.start - 60)
        ctx_end = min(len(sample.text), gt.end + 60)
        
        before = sample.text[ctx_start:gt.start]
        span = sample.text[gt.start:gt.end]
        after = sample.text[gt.end:ctx_end]
        
        console.print(f"  {before}[bold magenta on dark_blue]{span}[/]{after}")
        console.print()
        console.print(f"  Missed: [bold]{span}[/] (should be [cyan]{gt.normalized_type}[/])")
        console.print()
        
        choice = Prompt.ask("Decision", choices=["fn", "ok", "s", "q"], default="fn")
        
        if choice == "q":
            return False
        
        if choice == "s":
            self.skipped += 1
            return True
        
        # Create golden signal
        signal = GoldenSignal(
            id=f"fn_{index}_{datetime.now().strftime('%H%M%S')}",
            span_text=span,
            span_start=gt.start,
            span_end=gt.end,
            detected_type=None,  # We didn't detect it
            detected_conf=0.0,
            benchmark_type=gt.normalized_type,
            human_label="FN" if choice == "fn" else "OK",
            correct_type=gt.normalized_type if choice == "fn" else None,
            context_before=before,
            context_after=after,
            full_text=sample.text,
            sample_id=sample.id,
            reviewed_at=datetime.now().isoformat(),
            # No detector signals - we missed it entirely
        )
        
        self.signals.append(signal)
        self.fns_reviewed += 1
        
        label_text = "FN (should detect)" if choice == "fn" else "OK (fine to miss)"
        console.print(f"  [green]✓ Labeled as {label_text}[/green]")
        console.print()
        
        return True
    
    def _get_entity_type(self) -> Optional[str]:
        """Prompt for entity type selection."""
        from privplay.types import EntityType
        
        types = [t.value for t in EntityType if t.value != "OTHER"]
        
        console.print()
        console.print("Select correct type:")
        for i, t in enumerate(types[:20], 1):  # Show first 20
            console.print(f"  [{i:2}] {t}")
        
        choice = Prompt.ask("Enter number", default="")
        if not choice:
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(types):
                return types[idx]
        except ValueError:
            pass
        
        return None
    
    def save_signals(self):
        """Save golden signals to file."""
        if not self.signals:
            console.print("[yellow]No signals to save[/yellow]")
            return
        
        # Save as JSON
        output_file = self.output_dir / f"golden_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "created_at": datetime.now().isoformat(),
            "total_signals": len(self.signals),
            "fps_reviewed": self.fps_reviewed,
            "fns_reviewed": self.fns_reviewed,
            "skipped": self.skipped,
            "signals": [asdict(s) for s in self.signals],
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        console.print()
        console.print(f"[green]✓ Saved {len(self.signals)} signals to {output_file}[/green]")
        
        # Also save summary
        summary_file = self.output_dir / "latest_review_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Review Summary - {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total signals: {len(self.signals)}\n")
            f.write(f"FPs reviewed: {self.fps_reviewed}\n")
            f.write(f"FNs reviewed: {self.fns_reviewed}\n")
            f.write(f"Skipped: {self.skipped}\n\n")
            
            # Breakdown
            labels = {}
            for s in self.signals:
                labels[s.human_label] = labels.get(s.human_label, 0) + 1
            
            f.write("Label breakdown:\n")
            for label, count in sorted(labels.items()):
                f.write(f"  {label}: {count}\n")
        
        console.print(f"[dim]Summary saved to {summary_file}[/dim]")
        
        return output_file
    
    def show_summary(self):
        """Show review session summary."""
        console.print()
        console.print("[bold cyan]═══ Review Summary ═══[/bold cyan]")
        console.print()
        
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        
        table.add_row("Total signals", str(len(self.signals)))
        table.add_row("FPs reviewed", str(self.fps_reviewed))
        table.add_row("FNs reviewed", str(self.fns_reviewed))
        table.add_row("Skipped", str(self.skipped))
        
        console.print(table)
        
        # Label breakdown
        if self.signals:
            console.print()
            console.print("[bold]Label Distribution:[/bold]")
            labels = {}
            for s in self.signals:
                labels[s.human_label] = labels.get(s.human_label, 0) + 1
            
            for label, count in sorted(labels.items()):
                pct = count / len(self.signals) * 100
                console.print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Review benchmark FPs/FNs for golden signals")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to run")
    parser.add_argument("--fps-only", action="store_true", help="Only review FPs")
    parser.add_argument("--fns-only", action="store_true", help="Only review FNs")
    parser.add_argument("--output", type=str, default="golden_signals", help="Output directory")
    
    args = parser.parse_args()
    
    reviewer = BenchmarkReviewer(output_dir=Path(args.output))
    
    try:
        # Run benchmark
        bench_data = reviewer.run_benchmark(samples=args.samples)
        
        # Collect errors
        fps, fns = reviewer.collect_errors(
            bench_data["result"],
            bench_data["dataset"],
            bench_data["engine"],
        )
        
        # Review
        if not args.fns_only:
            reviewer.review_fps(fps)
        
        if not args.fps_only:
            reviewer.review_fns(fns)
        
        # Save and summarize
        reviewer.save_signals()
        reviewer.show_summary()
        
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Interrupted - saving progress...[/yellow]")
        reviewer.save_signals()
        reviewer.show_summary()


if __name__ == "__main__":
    main()
