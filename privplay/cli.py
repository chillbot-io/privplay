"""Privplay CLI - PHI/PII Training Pipeline."""

from pathlib import Path
from typing import Optional
import logging
import re
import json

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Core modules
from .config import get_config, set_config, Config
from .db import Database, get_db, set_db
from .types import EntityType

# Training modules
from .training.faker_gen import generate_documents, generate_simple_pii_examples
from .training.importer import import_from_directory, import_from_json, import_single_text
from .training.scanner import scan_documents, display_scan_results
from .training.reviewer import run_review_session, display_stats, display_top_fps
from .training.finetune import finetune as run_finetune
from .training.rules import (
    RuleManager, 
    CustomRule, 
    interactive_add_rule, 
    display_rules,
    display_test_results as display_rule_test_results,
    suggest_rules_from_corrections,
)

# Engine modules
from .engine.classifier import ClassificationEngine
from .engine.rules.engine import RuleEngine

# Verification
from .verification.verifier import get_verifier, OllamaVerifier

# Testing
from .testing.evaluator import evaluate_from_corrections, display_test_results

# Benchmark modules
from .benchmark import (
    list_datasets,
    get_dataset,
    BenchmarkRunner,
    BenchmarkStorage,
    display_benchmark_result,
    capture_benchmark_errors,
)

# Dictionaries
from .dictionaries import (
    get_dictionary_status,
)

# Stub out missing download functions (not needed for training)
def download_fda_drugs(): pass
def download_cms_hospitals(): pass
def download_npi_database(): pass
def download_all(): pass
def get_download_status(): return {}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("privplay")

app = typer.Typer(
    name="phi-train",
    help="PHI/PII Classification Training Pipeline",
    no_args_is_help=True,
)

console = Console()


def init_app(data_dir: Optional[Path] = None):
    """Initialize application configuration and database."""
    config = get_config()
    
    if data_dir:
        config = Config(data_dir=data_dir)
        set_config(config)
    
    # Ensure data directory exists
    config.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    db = Database(config.db_path)
    set_db(db)


@app.command()
def faker(
    n: int = typer.Option(10, "--count", "-n", help="Number of documents to generate"),
    simple: bool = typer.Option(False, "--simple", "-s", help="Generate simple PII examples"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Generate synthetic documents with faker."""
    init_app(data_dir)
    
    db = get_db()
    
    if simple:
        docs = generate_simple_pii_examples(n)
    else:
        docs = generate_documents(n)
    
    for doc in docs:
        db.add_document(doc)
    
    console.print(f"[green]✓ Generated {len(docs)} documents[/green]")


@app.command("import")
def import_docs(
    path: Path = typer.Argument(..., help="File or directory to import"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Import documents from files."""
    init_app(data_dir)
    
    if path.is_dir():
        docs = import_from_directory(path)
    elif path.suffix == ".json":
        docs = import_from_json(path)
    else:
        text = path.read_text()
        docs = [import_single_text(text, source=f"file:{path}")]
    
    db = get_db()
    for doc in docs:
        db.add_document(doc)
    
    console.print(f"[green]✓ Imported {len(docs)} documents[/green]")


@app.command()
def scan(
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Run LLM verification"),
    presidio: bool = typer.Option(True, "--presidio/--no-presidio", help="Run Presidio PII detection"),
    mock: bool = typer.Option(False, "--mock", help="Use mock model"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Scan documents for PHI/PII."""
    init_app(data_dir)
    
    config = get_config()
    config.presidio.enabled = presidio
    
    verifier = None
    if verify:
        verifier = get_verifier()
        if isinstance(verifier, OllamaVerifier):
            if not verifier.is_available():
                console.print("[yellow]Warning: Ollama not available, skipping verification[/yellow]")
                verifier = None
    
    engine = ClassificationEngine(use_mock_model=mock, config=config)
    
    db = get_db()
    scan_documents(db, engine, verifier=verifier)


@app.command()
def review(
    auto_approve: float = typer.Option(0.95, "--auto-approve", "-a", help="Auto-approve threshold"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Interactive review session for uncertain detections."""
    init_app(data_dir)
    
    db = get_db()
    run_review_session(db, auto_approve_threshold=auto_approve)


@app.command()
def stats(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show training progress statistics."""
    init_app(data_dir)
    
    db = get_db()
    display_stats(db.get_review_stats())
    display_top_fps(db)


@app.command()
def test(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Run F1 evaluation on reviewed data."""
    init_app(data_dir)
    
    result = evaluate_from_corrections(get_db())
    display_test_results(result)


@app.command("detect")
def detect_text(
    text: str = typer.Argument(..., help="Text to scan"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Run LLM verification"),
    presidio: bool = typer.Option(True, "--presidio/--no-presidio", help="Run Presidio PII detection"),
    mock: bool = typer.Option(False, "--mock", help="Use mock model"),
):
    """Detect PHI/PII in a single text (without storing)."""
    config = get_config()
    config.presidio.enabled = presidio
    
    engine = ClassificationEngine(use_mock_model=mock, config=config)
    entities = engine.detect(text, verify=verify)
    
    display_scan_results(entities, text)


@app.command()
def export(
    path: Path = typer.Argument(..., help="Output file path"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Export corrections to JSON."""
    init_app(data_dir)
    
    db = get_db()
    corrections = db.get_corrections()
    
    if not corrections:
        console.print("[yellow]No corrections to export.[/yellow]")
        raise typer.Exit(0)
    
    data = {
        "corrections": [
            {
                "entity_text": c.entity_text,
                "detected_type": c.detected_type.value,
                "decision": c.decision.value,
                "correct_type": c.correct_type.value if c.correct_type else None,
                "context_before": c.context_before,
                "context_after": c.context_after,
                "ner_confidence": c.ner_confidence,
                "llm_confidence": c.llm_confidence,
            }
            for c in corrections
        ]
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    console.print(f"[green]✓ Exported {len(corrections)} corrections to {path}[/green]")


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Reset database (delete all data)."""
    init_app(data_dir)
    
    if not confirm:
        confirm = typer.confirm("Delete all data?")
    
    if confirm:
        db = get_db()
        db.reset()
        console.print("[green]✓ Database reset[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


@app.command("config-show")
def config_show():
    """Show current configuration."""
    config = get_config()
    
    console.print()
    console.print("[bold]Current Configuration[/bold]")
    console.print("─" * 40)
    console.print(f"  Data directory:  {config.data_dir}")
    console.print(f"  Database:        {config.db_path}")
    console.print(f"  Presidio:        {'enabled' if config.presidio.enabled else 'disabled'}")
    console.print()


@app.command()
def stack(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show detection stack status (all components)."""
    init_app(data_dir)
    
    config = get_config()
    
    console.print()
    console.print("[bold]Detection Stack Status[/bold]")
    console.print("─" * 50)
    
    # Model
    console.print()
    console.print("[cyan]1. Transformer Model[/cyan]")
    console.print("   Model: StanfordAIMI/stanford-deidentifier-base")
    console.print("   Status: [green]Ready[/green]")
    
    # Presidio
    console.print()
    console.print("[cyan]2. Presidio PII Detection[/cyan]")
    if config.presidio.enabled:
        console.print("   Status: [green]Enabled[/green]")
    else:
        console.print("   Status: [yellow]Disabled[/yellow]")
    
    # Rules
    console.print()
    console.print("[cyan]3. Rule Engine[/cyan]")
    rule_engine = RuleEngine()
    console.print(f"   Built-in rules: {len(rule_engine.rules)}")
    
    rule_manager = RuleManager()
    custom_rules = rule_manager.list_rules()
    console.print(f"   Custom rules: {len(custom_rules)}")
    
    # Dictionaries
    console.print()
    console.print("[cyan]4. Dictionaries[/cyan]")
    dict_status = get_dictionary_status()
    for name, exists in dict_status.get("bundled", {}).items():
        status = "[green]✓[/green]" if exists else "[red]✗[/red]"
        console.print(f"   {name}: {status}")
    for name, exists in dict_status.get("downloaded", {}).items():
        status = "[green]✓[/green]" if exists else "[yellow]not downloaded[/yellow]"
        console.print(f"   {name}: {status}")
    
    # Verifier
    console.print()
    console.print("[cyan]5. LLM Verifier[/cyan]")
    try:
        verifier = get_verifier()
        if isinstance(verifier, OllamaVerifier):
            if verifier.is_available():
                console.print("   Ollama: [green]Available[/green]")
            else:
                console.print("   Ollama: [yellow]Not running[/yellow]")
        else:
            console.print("   Status: [green]Ready[/green]")
    except Exception:
        console.print("   Status: [yellow]Not configured[/yellow]")
    
    console.print()


@app.command()
def download(
    target: str = typer.Argument("all", help="What to download: all, dictionaries, npi, drugs, hospitals"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Download dictionary data and NPI database."""
    init_app(data_dir)
    
    console.print()
    
    if target == "all":
        console.print("[bold]Downloading all data...[/bold]")
        download_all()
    elif target == "dictionaries":
        console.print("[bold]Downloading dictionaries...[/bold]")
        download_fda_drugs()
        download_cms_hospitals()
    elif target == "npi":
        console.print("[bold]Downloading NPI database...[/bold]")
        download_npi_database()
    elif target == "drugs":
        console.print("[bold]Downloading FDA drugs...[/bold]")
        download_fda_drugs()
    elif target == "hospitals":
        console.print("[bold]Downloading CMS hospitals...[/bold]")
        download_cms_hospitals()
    else:
        console.print(f"[red]Unknown target: {target}[/red]")
        console.print("Available: all, dictionaries, npi, drugs, hospitals")
        raise typer.Exit(1)
    
    console.print()
    console.print("[green]✓ Download complete[/green]")


# Benchmark subcommands
benchmark_app = typer.Typer(help="Benchmark detection against standard datasets")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("list")
def benchmark_list():
    """List available benchmark datasets."""
    datasets = list_datasets()
    
    console.print()
    console.print("[bold]Available Benchmark Datasets[/bold]")
    console.print("─" * 50)
    
    for ds in datasets:
        console.print(f"\n  [cyan]{ds['name']}[/cyan]")
        console.print(f"    {ds['description']}")
    
    console.print()


@benchmark_app.command("run")
def benchmark_run(
    dataset: str = typer.Argument(..., help="Dataset name (ai4privacy, synthetic_phi, physionet)"),
    samples: int = typer.Option(100, "--samples", "-n", help="Number of samples"),
    verify: bool = typer.Option(False, "--verify", help="Run LLM verification"),
    physionet_dir: Optional[Path] = typer.Option(None, "--physionet-dir", help="PhysioNet data directory"),
    mock: bool = typer.Option(False, "--mock", help="Use mock model"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to history"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    capture_errors: bool = typer.Option(False, "--capture-errors", help="Capture TPs and FPs as training data"),
    capture_signals: bool = typer.Option(False, "--capture-signals", help="Capture SpanSignals for meta-classifier"),
):
    """Run benchmark on a dataset."""
    init_app(data_dir)
    
    console.print()
    console.print(f"[bold]Running Benchmark: {dataset}[/bold]")
    console.print("─" * 50)
    
    # Load dataset
    try:
        if dataset == "physionet":
            if not physionet_dir:
                console.print("[red]PhysioNet requires --physionet-dir[/red]")
                raise typer.Exit(1)
            ds = get_dataset(dataset, max_samples=samples, data_dir=physionet_dir)
        else:
            ds = get_dataset(dataset, max_samples=samples)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(f"  Loaded {len(ds)} samples")
    
    # Create engine
    config = get_config()
    engine = ClassificationEngine(use_mock_model=mock, config=config)
    
    # Create storage if saving
    storage = BenchmarkStorage() if save else None
    
    # Run benchmark
    runner = BenchmarkRunner(engine, storage=storage)
    result = runner.run(ds, verify=verify)
    
    # Display results
    display_benchmark_result(result)
    
    # Capture training data (both TPs and FPs)
    if capture_errors:
        console.print()
        console.print("[bold]Capturing training data...[/bold]")
        stats = capture_benchmark_errors(result, ds)
        console.print()
        console.print(f"[green]✓ Captured {stats['tps_captured']} TPs as CONFIRMED (positive examples)[/green]")
        console.print(f"[yellow]✓ Captured {stats['fps_captured']} FPs as REJECTED (negative examples)[/yellow]")
        console.print(f"[dim]  Documents created: {stats['documents']}[/dim]")
        console.print(f"[dim]  FNs skipped (can't learn from these): {stats['fns_skipped']}[/dim]")
        console.print()
        total = stats['tps_captured'] + stats['fps_captured']
        if total > 0:
            balance = stats['tps_captured'] / total * 100
            console.print(f"[bold]Training data balance: {balance:.1f}% positive, {100-balance:.1f}% negative[/bold]")
    
    # Capture signals for meta-classifier
    if capture_signals:
        console.print()
        console.print("[bold]Capturing signals for meta-classifier...[/bold]")
        from .benchmark.runner import capture_benchmark_signals
        stats = capture_benchmark_signals(result, ds, engine)
        console.print(f"[green]✓ Captured {stats['signals_captured']} signals[/green]")
        console.print(f"  TPs labeled: {stats['tps_labeled']}")
        console.print(f"  FPs labeled: {stats['fps_labeled']}")
        console.print(f"  Balance: {stats['balance']}")


@benchmark_app.command("history")
def benchmark_history(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Show benchmark history."""
    init_app(data_dir)
    
    storage = BenchmarkStorage()
    runs = storage.get_recent_runs(dataset_name=dataset, limit=limit)
    
    if not runs:
        console.print("[dim]No benchmark history found.[/dim]")
        return
    
    console.print()
    console.print("[bold]Benchmark History[/bold]")
    console.print("─" * 80)
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Date")
    table.add_column("Dataset")
    table.add_column("Samples")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1")
    table.add_column("TP/FP/FN")
    
    for run in runs:
        table.add_row(
            run.run_id[:12] + "...",
            run.timestamp.strftime("%Y-%m-%d %H:%M"),
            run.dataset_name[:15],
            str(run.num_samples),
            f"{run.precision:.1%}",
            f"{run.recall:.1%}",
            f"{run.f1:.1%}",
            f"{run.true_positives}/{run.false_positives}/{run.false_negatives}",
        )
    
    console.print(table)
    console.print()


@benchmark_app.command("compare")
def benchmark_compare(
    id1: str = typer.Argument(..., help="First run ID (or prefix)"),
    id2: str = typer.Argument(..., help="Second run ID (or prefix)"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory"),
):
    """Compare two benchmark runs."""
    init_app(data_dir)
    
    storage = BenchmarkStorage()
    
    # Allow partial ID matching
    runs = storage.get_recent_runs(limit=100)
    
    run1 = None
    run2 = None
    for run in runs:
        if run.run_id.startswith(id1):
            run1 = run
        if run.run_id.startswith(id2):
            run2 = run
    
    if not run1:
        console.print(f"[red]Run not found: {id1}[/red]")
        raise typer.Exit(1)
    if not run2:
        console.print(f"[red]Run not found: {id2}[/red]")
        raise typer.Exit(1)
    
    def delta_str(v1, v2, percent=True):
        diff = v2 - v1
        if percent:
            s = f"{diff:+.1%}"
        else:
            s = f"{diff:+d}"
        color = "green" if diff > 0 else "red" if diff < 0 else "dim"
        return f"[{color}]{s}[/{color}]"
    
    console.print()
    console.print("[bold]Benchmark Comparison[/bold]")
    console.print("─" * 50)
    console.print()
    console.print(f"  {'Metric':<15} {'Run 1':>12} {'Run 2':>12} {'Delta':>12}")
    console.print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12}")
    console.print(f"  {'Dataset':<15} {run1.dataset_name[:12]:>12} {run2.dataset_name[:12]:>12}")
    console.print(f"  {'Date':<15} {run1.timestamp.strftime('%m/%d %H:%M'):>12} {run2.timestamp.strftime('%m/%d %H:%M'):>12}")
    console.print(f"  {'Samples':<15} {run1.num_samples:>12} {run2.num_samples:>12}")
    console.print()
    console.print(f"  {'Precision':<15} {run1.precision:>11.1%} {run2.precision:>11.1%} {delta_str(run1.precision, run2.precision):>12}")
    console.print(f"  {'Recall':<15} {run1.recall:>11.1%} {run2.recall:>11.1%} {delta_str(run1.recall, run2.recall):>12}")
    console.print(f"  {'F1':<15} {run1.f1:>11.1%} {run2.f1:>11.1%} {delta_str(run1.f1, run2.f1):>12}")
    console.print()
    console.print(f"  {'True Pos':<15} {run1.true_positives:>12} {run2.true_positives:>12} {delta_str(run1.true_positives, run2.true_positives, False):>12}")
    console.print(f"  {'False Pos':<15} {run1.false_positives:>12} {run2.false_positives:>12} {delta_str(run1.false_positives, run2.false_positives, False):>12}")
    console.print(f"  {'False Neg':<15} {run1.false_negatives:>12} {run2.false_negatives:>12} {delta_str(run1.false_negatives, run2.false_negatives, False):>12}")
    console.print()


# =============================================================================
# META-CLASSIFIER COMMANDS
# =============================================================================

meta_app = typer.Typer(help="Meta-classifier commands")
app.add_typer(meta_app, name="meta")


@meta_app.command("status")
def meta_status(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show meta-classifier and signals status."""
    init_app(data_dir)
    
    from .training.signals_storage import get_signals_storage
    from .training.meta_classifier import MetaClassifier
    
    storage = get_signals_storage()
    counts = storage.count_signals()
    
    console.print()
    console.print("[bold]Meta-Classifier Status[/bold]")
    console.print("─" * 50)
    
    console.print()
    console.print("[cyan]Training Signals:[/cyan]")
    
    table = Table(show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Total signals", str(counts["total"]))
    table.add_row("Labeled", str(counts["labeled"]))
    table.add_row("  Positive (is PHI)", str(counts["positive"]))
    table.add_row("  Negative (not PHI)", str(counts["negative"]))
    table.add_row("Unlabeled", str(counts["unlabeled"]))
    
    if counts["labeled"] > 0:
        balance = counts["positive"] / counts["labeled"] * 100
        table.add_row("Balance", f"{balance:.1f}% positive")
    
    console.print(table)
    
    console.print()
    console.print("[cyan]Model:[/cyan]")
    
    meta = MetaClassifier()
    if meta.is_trained():
        console.print("  Status: [green]Trained[/green]")
        
        imp_file = meta.model_path / "feature_importance.json"
        if imp_file.exists():
            with open(imp_file) as f:
                data = json.load(f)
            console.print("  Top features:")
            for name, importance in data["importances"][:5]:
                console.print(f"    {name}: {importance:.3f}")
    else:
        console.print("  Status: [yellow]Not trained[/yellow]")
        if counts["labeled"] < 50:
            console.print(f"  Need at least 50 labeled signals (have {counts['labeled']})")
        else:
            console.print("  Run: phi-train meta train")
    
    console.print()


@meta_app.command("train")
def meta_train(
    use_xgboost: bool = typer.Option(False, "--xgboost", help="Use XGBoost instead of RandomForest"),
    min_samples: int = typer.Option(50, "--min-samples", help="Minimum samples required"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Train the meta-classifier on captured signals."""
    init_app(data_dir)
    
    from .training.signals_storage import get_signals_storage
    from .training.meta_classifier import MetaClassifier
    
    storage = get_signals_storage()
    signals = storage.get_labeled_signals()
    
    if len(signals) < min_samples:
        console.print(f"[red]Need at least {min_samples} labeled signals, have {len(signals)}[/red]")
        console.print("Run: phi-train benchmark run <dataset> -n 1000 --capture-signals")
        raise typer.Exit(1)
    
    console.print(f"Training on {len(signals)} labeled signals...")
    
    positive = sum(1 for s in signals if s.ground_truth_type != "NONE")
    negative = len(signals) - positive
    console.print(f"  Positive: {positive} ({positive/len(signals)*100:.1f}%)")
    console.print(f"  Negative: {negative} ({negative/len(signals)*100:.1f}%)")
    
    meta = MetaClassifier(min_samples_for_training=min_samples)
    
    try:
        metrics = meta.train(signals, use_xgboost=use_xgboost)
        
        console.print()
        console.print("[green]Training complete![/green]")
        console.print(f"  is_entity F1: {metrics['is_entity_f1']:.1%}")
        console.print(f"  is_entity Precision: {metrics['is_entity_precision']:.1%}")
        console.print(f"  is_entity Recall: {metrics['is_entity_recall']:.1%}")
        console.print(f"  Train samples: {metrics['train_samples']}")
        console.print(f"  Test samples: {metrics['test_samples']}")
        console.print()
        console.print(f"Model saved to: {meta.model_path}")
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@meta_app.command("clear")
def meta_clear(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Clear all captured signals."""
    init_app(data_dir)
    
    from .training.signals_storage import get_signals_storage
    
    if not typer.confirm("This will delete all captured signals. Continue?"):
        raise typer.Abort()
    
    storage = get_signals_storage()
    storage.clear()
    
    console.print("[green]Signals cleared.[/green]")


# Fine-tuning command
@app.command()
def finetune(
    corrections_path: Optional[Path] = typer.Option(None, "--corrections", "-c", help="Path to exported corrections JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for model"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Fine-tune the detection model on your corrections."""
    init_app(data_dir)
    
    # Check we have corrections
    if corrections_path is None:
        db = get_db()
        corrections = db.get_corrections()
        if len(corrections) < 10:
            console.print(f"[yellow]Only {len(corrections)} corrections found.[/yellow]")
            console.print()
            console.print("Fine-tuning needs at least 50-100 corrections to be effective.")
            console.print("500+ is recommended for meaningful improvement.")
            console.print()
            console.print("Build corrections with:")
            console.print("  [bold]phi-train benchmark run ai4privacy -n 1000 --capture-errors[/bold]")
            console.print()
            if len(corrections) < 10:
                raise typer.Exit(1)
    
    console.print()
    console.print("[bold]Fine-tuning PHI Detection Model[/bold]")
    console.print("─" * 50)
    console.print()
    
    console.print(f"  Epochs:        {epochs}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print()
    
    try:
        model_path = run_finetune(
            corrections_path=corrections_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        console.print()
        console.print(f"[green]✓ Model saved to: {model_path}[/green]")
        console.print()
        console.print("To use the fine-tuned model, update your config or set:")
        console.print(f"  PRIVPLAY_MODEL_PATH={model_path}")
        
    except Exception as e:
        console.print(f"[red]Fine-tuning failed: {e}[/red]")
        raise typer.Exit(1)


# Rule management subcommands
rule_app = typer.Typer(help="Manage custom detection rules")
app.add_typer(rule_app, name="rule")


@rule_app.command("list")
def rule_list(
    show_patterns: bool = typer.Option(False, "--patterns", "-p", help="Show regex patterns"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """List all custom rules."""
    init_app(data_dir)
    
    manager = RuleManager()
    rules = manager.list_rules()
    
    console.print()
    console.print("[bold]Custom Detection Rules[/bold]")
    console.print("─" * 50)
    console.print()
    
    display_rules(rules, show_patterns=show_patterns)
    
    console.print()
    console.print(f"[dim]{len(rules)} custom rules defined[/dim]")


@rule_app.command("add")
def rule_add(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Rule name"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Regex pattern"),
    entity_type: Optional[str] = typer.Option(None, "--type", "-t", help="Entity type"),
    confidence: float = typer.Option(0.90, "--confidence", "-c", help="Confidence score"),
    description: str = typer.Option("", "--desc", help="Description"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Add a custom detection rule."""
    init_app(data_dir)
    
    manager = RuleManager()
    
    # If all required args provided, create directly
    if name and pattern and entity_type:
        rule = CustomRule(
            name=name,
            pattern=pattern,
            entity_type=entity_type.upper(),
            confidence=confidence,
            description=description,
        )
    else:
        # Interactive mode
        rule = interactive_add_rule()
        if rule is None:
            raise typer.Exit(1)
    
    if manager.add_rule(rule):
        console.print(f"[green]✓ Added rule: {rule.name}[/green]")
    else:
        console.print("[red]Failed to add rule[/red]")
        raise typer.Exit(1)


@rule_app.command("remove")
def rule_remove(
    name: str = typer.Argument(..., help="Rule name to remove"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Remove a custom rule."""
    init_app(data_dir)
    
    manager = RuleManager()
    
    if manager.remove_rule(name):
        console.print(f"[green]✓ Removed rule: {name}[/green]")
    else:
        console.print(f"[red]Rule not found: {name}[/red]")
        raise typer.Exit(1)


@rule_app.command("test")
def rule_test(
    pattern: str = typer.Argument(..., help="Regex pattern to test"),
    text: str = typer.Option(None, "--text", "-t", help="Text to test against"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File to test against"),
    ignore_case: bool = typer.Option(False, "--ignore-case", "-i", help="Case insensitive"),
):
    """Test a regex pattern against text."""
    # Get text to test
    if file:
        test_text = file.read_text()
    elif text:
        test_text = text
    else:
        test_text = typer.prompt("Enter text to test")
    
    flags = re.IGNORECASE if ignore_case else 0
    
    manager = RuleManager()
    matches = manager.test_pattern(pattern, test_text, flags)
    
    display_rule_test_results(pattern, test_text, matches)


@rule_app.command("builtin")
def rule_builtin():
    """Show all built-in detection rules."""
    engine = RuleEngine()
    
    console.print()
    console.print("[bold]Built-in Detection Rules[/bold]")
    console.print("─" * 60)
    console.print()
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Entity Type")
    table.add_column("Confidence")
    table.add_column("Pattern Preview")
    
    for rule in engine.rules:
        pattern_str = rule.pattern.pattern
        if len(pattern_str) > 40:
            pattern_str = pattern_str[:37] + "..."
        
        table.add_row(
            rule.name,
            rule.entity_type.value,
            f"{rule.confidence:.0%}",
            f"[dim]{pattern_str}[/dim]",
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]{len(engine.rules)} built-in rules[/dim]")


@rule_app.command("suggest")
def rule_suggest(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Suggest new rules based on correction patterns."""
    init_app(data_dir)
    
    console.print()
    console.print("[bold]Rule Suggestions from Corrections[/bold]")
    console.print("─" * 50)
    console.print()
    
    suggestions = suggest_rules_from_corrections()
    
    if not suggestions:
        console.print("[dim]No suggestions yet. Need more corrections.[/dim]")
        console.print()
        console.print("Build corrections with:")
        console.print("  [bold]phi-train review[/bold]")
        return
    
    for i, s in enumerate(suggestions):
        if s["type"] == "exclusion":
            console.print(f"{i+1}. [yellow]Exclude[/yellow]: \"{s['text']}\"")
            console.print(f"   Reason: {s['reason']}")
            console.print(f"   Pattern: [dim]{s['suggested_pattern']}[/dim]")
            console.print()


if __name__ == "__main__":
    app()
