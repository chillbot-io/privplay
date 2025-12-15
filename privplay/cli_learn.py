"""Learn CLI Commands - Continuous Learning for Meta-Classifier.

Add to cli.py:
    from .cli_learn import learn_app
    app.add_typer(learn_app, name="learn")

Commands:
    phi-train learn status    - Show learning status
    phi-train learn review    - Interactive HIL review TUI
    phi-train learn retrain   - Trigger training run
    phi-train learn models    - Show model history
    phi-train learn rollback  - Rollback to previous model
    phi-train learn daemon    - Run background training daemon
"""

from pathlib import Path
from typing import Optional
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
logger = logging.getLogger(__name__)

learn_app = typer.Typer(
    help="Continuous learning - review signals and retrain model",
    no_args_is_help=True,
)


def _init_learning(data_dir: Optional[Path] = None):
    """Initialize learning module."""
    from .config import get_config, set_config, Config
    
    config = get_config()
    if data_dir:
        config = Config(data_dir=data_dir)
        set_config(config)
    
    config.data_dir.mkdir(parents=True, exist_ok=True)


@learn_app.command("status")
def learn_status(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show continuous learning status.
    
    Displays:
    - Pending signals awaiting review
    - Labeled signals ready for training
    - Active model info
    - Whether training threshold is met
    """
    _init_learning(data_dir)
    
    from .learning import check_training_status, get_signal_store
    
    console.print()
    console.print("[bold cyan]═══ Continuous Learning Status ═══[/bold cyan]")
    console.print()
    
    status = check_training_status()
    store = get_signal_store()
    stats = store.get_stats()
    
    # Signal stats
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Pending review", str(stats['pending_review']))
    table.add_row("Labeled signals", str(stats['reviewed']))
    table.add_row("False negatives pending", str(stats['false_negatives_pending']))
    table.add_row("Archived", str(stats['archived']))
    
    console.print(table)
    console.print()
    
    # Training status
    threshold = status['threshold']
    labeled = status['labeled_signals']
    progress = min(100, labeled / threshold * 100)
    
    console.print(f"Training progress: {labeled}/{threshold} signals ({progress:.0f}%)")
    
    if status['should_train']:
        console.print("[bold green]✓ Ready for training![/bold green]")
        console.print()
        console.print("Run: [bold]phi-train learn retrain[/bold]")
    else:
        remaining = threshold - labeled
        console.print(f"[dim]{remaining} more labeled signals needed[/dim]")
    
    console.print()
    
    # Active model
    if status['active_model']:
        model = status['active_model']
        console.print("[bold]Active Model[/bold]")
        console.print(f"  ID: {model['id']}")
        console.print(f"  F1: {model['f1']:.1%}")
        console.print(f"  Created: {model['created_at']}")
    else:
        console.print("[yellow]No active model[/yellow]")
        console.print("[dim]Run initial training with: phi-train train run[/dim]")
    
    # Staged models
    if status['staged_models']:
        console.print()
        console.print("[bold]Staged Models (shadow period)[/bold]")
        for m in status['staged_models']:
            console.print(f"  {m['id']} (F1={m['f1']:.1%}) - {m['shadow_remaining']} remaining")
    
    console.print()


@learn_app.command("review")
def learn_review(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max signals to review"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Interactive review of uncertain detections.
    
    Opens TUI to review signals with confidence < 0.95.
    
    Menu options:
      [tp] True Positive   - Correct detection, correct label
      [wl] Wrong Label     - Correct detection, wrong type
      [fp] False Positive  - Not PHI at all
      [fn] False Negative  - Report missed PHI
      [s]  Skip
      [q]  Quit
    """
    _init_learning(data_dir)
    
    from .learning import run_review_tui, get_signal_store
    
    store = get_signal_store()
    pending = store.get_pending_count()
    
    if pending == 0:
        console.print()
        console.print("[green]✓ No signals pending review![/green]")
        console.print()
        console.print("[dim]Signals are captured when detection confidence < 0.95[/dim]")
        console.print("[dim]Use LearningClassifier to enable signal capture[/dim]")
        console.print()
        return
    
    reviewed = run_review_tui(limit=limit)
    
    # Check if training threshold reached
    labeled = store.get_labeled_signal_count()
    if labeled >= 1000:
        console.print()
        console.print("[bold green]Training threshold reached![/bold green]")
        console.print("Run: [bold]phi-train learn retrain[/bold]")


@learn_app.command("retrain")
def learn_retrain(
    force: bool = typer.Option(False, "--force", "-f", help="Train even if below threshold"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Trigger meta-classifier retraining.
    
    Training uses:
    - All labeled production signals
    - 100% of adversarial cases (always included)
    - 20% sample of original synthetic corpus
    
    New model is staged for 24h shadow period before promotion.
    If F1 drops > 2%, model is rejected automatically.
    """
    _init_learning(data_dir)
    
    from .learning import trigger_training, check_training_status
    
    status = check_training_status()
    
    if not status['should_train'] and not force:
        console.print()
        console.print(f"[yellow]Below threshold: {status['labeled_signals']}/{status['threshold']}[/yellow]")
        console.print()
        console.print("Use [bold]--force[/bold] to train anyway")
        console.print()
        raise typer.Exit(1)
    
    console.print()
    console.print("[bold cyan]═══ Continuous Learning - Retraining ═══[/bold cyan]")
    console.print()
    
    console.print(f"Labeled signals: {status['labeled_signals']}")
    console.print()
    
    try:
        model = trigger_training(force=force)
        
        if model:
            console.print()
            console.print("[bold green]✓ Training complete![/bold green]")
            console.print()
            
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("", style="cyan")
            table.add_column("")
            
            table.add_row("Model ID", model.id)
            table.add_row("F1 Score", f"{model.f1_score:.1%}")
            table.add_row("Precision", f"{model.precision:.1%}")
            table.add_row("Recall", f"{model.recall:.1%}")
            table.add_row("Status", model.status)
            table.add_row("Path", model.model_path)
            
            console.print(table)
            console.print()
            
            if model.status == "staged":
                console.print("[dim]Model is staged for 24h shadow period[/dim]")
                console.print("[dim]Will auto-promote if no issues detected[/dim]")
        else:
            console.print()
            console.print("[yellow]Training skipped or model rejected[/yellow]")
            console.print("[dim]Check logs for details[/dim]")
            
    except Exception as e:
        console.print()
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)
    
    console.print()


@learn_app.command("models")
def learn_models(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of models to show"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show model version history.
    
    Displays recent models with their status:
    - active: Currently in use
    - staged: In shadow period (24h)
    - rolled_back: Demoted
    """
    _init_learning(data_dir)
    
    from .learning import get_signal_store
    
    store = get_signal_store()
    models = store.get_model_history(limit=limit)
    
    console.print()
    console.print("[bold cyan]═══ Model History ═══[/bold cyan]")
    console.print()
    
    if not models:
        console.print("[dim]No models found[/dim]")
        console.print()
        console.print("Train initial model with: [bold]phi-train train run[/bold]")
        console.print()
        return
    
    table = Table(box=box.SIMPLE)
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Created")
    
    for m in models:
        status_style = {
            "active": "bold green",
            "staged": "yellow",
            "rolled_back": "dim",
        }.get(m.status, "")
        
        table.add_row(
            m.id[:12],
            f"[{status_style}]{m.status}[/]",
            f"{m.f1_score:.1%}",
            f"{m.precision:.1%}",
            f"{m.recall:.1%}",
            str(m.training_signals),
            m.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)
    console.print()


@learn_app.command("rollback")
def learn_rollback(
    reason: str = typer.Option("manual", "--reason", "-r", help="Rollback reason"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Rollback to previous model.
    
    Demotes current active model and promotes the previous one.
    Use when new model is performing poorly in production.
    """
    _init_learning(data_dir)
    
    from .learning import get_signal_store
    
    store = get_signal_store()
    
    # Confirm
    current = store.get_active_model()
    if not current:
        console.print()
        console.print("[yellow]No active model to rollback[/yellow]")
        console.print()
        raise typer.Exit(1)
    
    console.print()
    console.print(f"Current model: {current.id} (F1={current.f1_score:.1%})")
    console.print()
    
    if not typer.confirm("Rollback to previous model?"):
        raise typer.Abort()
    
    previous_id = store.rollback_model(reason=reason)
    
    if previous_id:
        console.print()
        console.print(f"[green]✓ Rolled back to {previous_id}[/green]")
        console.print(f"[dim]Reason: {reason}[/dim]")
    else:
        console.print()
        console.print("[yellow]No previous model available for rollback[/yellow]")
    
    console.print()


@learn_app.command("promote")
def learn_promote(
    model_id: str = typer.Argument(..., help="Model ID to promote"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Manually promote a staged model to active.
    
    Skips the 24h shadow period. Use with caution.
    """
    _init_learning(data_dir)
    
    from .learning import get_signal_store
    
    store = get_signal_store()
    
    console.print()
    console.print(f"Promoting model: {model_id}")
    
    if store.promote_model(model_id):
        console.print()
        console.print(f"[green]✓ Model {model_id} is now active[/green]")
    else:
        console.print()
        console.print(f"[red]Failed to promote model {model_id}[/red]")
        console.print("[dim]Model may not exist or already be active[/dim]")
        raise typer.Exit(1)
    
    console.print()


@learn_app.command("archive")
def learn_archive(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Archive old signals (> 90 days).
    
    Moves old signals from active DB to archive DB.
    This is normally done automatically by the daemon.
    """
    _init_learning(data_dir)
    
    from .learning import get_signal_store
    
    store = get_signal_store()
    
    console.print()
    console.print("Archiving old signals...")
    
    archived = store.archive_old_signals()
    
    console.print()
    console.print(f"[green]✓ Archived {archived} signals[/green]")
    console.print()


@learn_app.command("daemon")
def learn_daemon(
    start_hour: int = typer.Option(2, "--start-hour", help="Training window start (0-23)"),
    end_hour: int = typer.Option(6, "--end-hour", help="Training window end (0-23)"),
    check_interval: int = typer.Option(60, "--interval", help="Check interval in minutes"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Run continuous learning daemon.
    
    Monitors labeled signal count and triggers training overnight
    when threshold (1000) is reached.
    
    Default training window: 2 AM - 6 AM
    
    Example:
        phi-train learn daemon --start-hour 1 --end-hour 5
    """
    _init_learning(data_dir)
    
    from .learning import TrainingDaemon, get_trainer
    
    console.print()
    console.print("[bold cyan]═══ Continuous Learning Daemon ═══[/bold cyan]")
    console.print()
    console.print(f"Training window: {start_hour}:00 - {end_hour}:00")
    console.print(f"Check interval: {check_interval} minutes")
    console.print()
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()
    
    daemon = TrainingDaemon(
        trainer=get_trainer(),
        check_interval_minutes=check_interval,
        training_start_hour=start_hour,
        training_end_hour=end_hour,
    )
    
    try:
        daemon.run()
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Daemon stopped[/yellow]")


@learn_app.command("stats")
def learn_stats(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
):
    """Show detailed signal store statistics."""
    _init_learning(data_dir)
    
    from .learning import show_stats_tui
    
    show_stats_tui()


# For integration into main cli.py
def register_learn_commands(app: typer.Typer):
    """Register learn commands with main app."""
    app.add_typer(learn_app, name="learn")
