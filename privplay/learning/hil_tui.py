"""Human-in-the-Loop TUI for Signal Review.

Developer interface for reviewing uncertain detections and providing
feedback to improve the meta-classifier.

Menu:
  [tp] True Positive   - Correct detection, correct label
  [wl] Wrong Label     - Correct detection, wrong label
  [fp] False Positive  - Not PHI
  [fn] False Negative  - Report missed PHI
  [s]  Skip
  [q]  Quit
"""

import sys
from typing import Optional, List, Tuple
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import box

from .signal_store import (
    SignalStore, 
    CapturedSignal, 
    FeedbackType, 
    get_signal_store,
)
from ..types import EntityType

console = Console()


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def highlight_span_in_context(
    text: str,
    start: int,
    end: int,
    context_window: int = 80,
) -> Text:
    """Create highlighted text showing span in context."""
    # Calculate context bounds
    ctx_start = max(0, start - context_window)
    ctx_end = min(len(text), end + context_window)
    
    result = Text()
    
    # Leading ellipsis
    if ctx_start > 0:
        result.append("...", style="dim")
    
    # Before span
    result.append(text[ctx_start:start])
    
    # Highlighted span
    result.append(text[start:end], style="bold yellow on dark_red")
    
    # After span
    result.append(text[end:ctx_end])
    
    # Trailing ellipsis
    if ctx_end < len(text):
        result.append("...", style="dim")
    
    return result


def display_signal_for_review(
    signal: CapturedSignal,
    index: int,
    total: int,
    document_text: Optional[str] = None,
) -> None:
    """Display a signal for human review."""
    console.print()
    console.print("═" * 70, style="dim")
    
    # Header
    progress_pct = (index / total * 100) if total > 0 else 0
    header = f"Signal {index}/{total}  │  {progress_pct:.0f}% complete"
    console.print(header, style="bold cyan")
    console.print()
    
    # Context panel
    if document_text:
        highlighted = highlight_span_in_context(
            document_text,
            signal.span_start,
            signal.span_end,
        )
    else:
        # Fallback: just show the span text
        highlighted = Text()
        highlighted.append("[", style="dim")
        highlighted.append(signal.span_text, style="bold yellow")
        highlighted.append("]", style="dim")
    
    console.print(Panel(
        highlighted,
        title="Context",
        box=box.ROUNDED,
        padding=(0, 2),
    ))
    
    # Detection info
    console.print()
    console.print(f"  Detected: ", style="dim", end="")
    console.print(f"[{signal.span_text}]", style="bold yellow")
    
    console.print(f"  As type:  ", style="dim", end="")
    console.print(signal.merged_type, style="cyan bold")
    
    console.print(f"  Confidence: ", style="dim", end="")
    conf_style = "green" if signal.merged_conf > 0.8 else "yellow" if signal.merged_conf > 0.5 else "red"
    console.print(f"{signal.merged_conf:.1%}", style=conf_style)
    
    # Detector breakdown
    console.print()
    console.print("  Detectors:", style="dim")
    
    detectors = []
    if signal.phi_bert_detected:
        detectors.append(f"PHI-BERT ({signal.phi_bert_conf:.0%})")
    if signal.pii_bert_detected:
        detectors.append(f"PII-BERT ({signal.pii_bert_conf:.0%})")
    if signal.presidio_detected:
        detectors.append(f"Presidio ({signal.presidio_conf:.0%})")
    if signal.rule_detected:
        rule_info = f"Rule ({signal.rule_conf:.0%})"
        if signal.rule_has_checksum:
            rule_info += " ✓checksum"
        detectors.append(rule_info)
    
    if detectors:
        for d in detectors:
            console.print(f"    • {d}", style="dim")
    else:
        console.print("    (none)", style="dim italic")
    
    # Coreference info
    if signal.in_coref_cluster:
        console.print()
        console.print("  Coreference:", style="dim")
        console.print(f"    Cluster size: {signal.coref_cluster_size}", style="dim")
        if signal.coref_anchor_type:
            console.print(f"    Anchor type: {signal.coref_anchor_type}", style="dim")
        if signal.coref_is_pronoun:
            console.print("    Is pronoun: yes", style="dim")
    
    console.print()


def display_menu() -> None:
    """Display the feedback menu."""
    console.print("  [bold green][tp][/] True Positive   [dim]- Correct detection, correct label[/]")
    console.print("  [bold yellow][wl][/] Wrong Label     [dim]- Correct detection, wrong label[/]")
    console.print("  [bold red][fp][/] False Positive  [dim]- Not PHI at all[/]")
    console.print("  [bold magenta][fn][/] False Negative  [dim]- Report missed PHI[/]")
    console.print("  [dim][s][/]  Skip")
    console.print("  [dim][q][/]  Quit")
    console.print()


def get_entity_type_selection() -> Optional[str]:
    """Prompt user to select correct entity type."""
    console.print()
    console.print("Select correct entity type:", style="bold")
    console.print()
    
    # Get all entity types
    types = [t for t in EntityType]
    
    # Display in columns
    for i, etype in enumerate(types, 1):
        console.print(f"  [{i:2}] {etype.value}")
    
    console.print()
    choice = Prompt.ask("Enter number", default="")
    
    if not choice:
        return None
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(types):
            return types[idx].value
    except ValueError:
        pass
    
    console.print("[red]Invalid selection[/]")
    return None


def prompt_for_missed_text() -> Tuple[Optional[str], Optional[str]]:
    """Prompt user to enter missed PHI text and type.
    
    Returns (missed_text, entity_type) or (None, None) if cancelled.
    """
    console.print()
    console.print("Report missed PHI:", style="bold")
    console.print()
    
    missed_text = Prompt.ask("Enter the missed text", default="")
    if not missed_text:
        return None, None
    
    entity_type = get_entity_type_selection()
    if not entity_type:
        return None, None
    
    return missed_text, entity_type


# =============================================================================
# REVIEW SESSION
# =============================================================================

class ReviewSession:
    """Interactive review session."""
    
    def __init__(
        self,
        store: Optional[SignalStore] = None,
        document_loader: Optional[callable] = None,
    ):
        """Initialize review session.
        
        Args:
            store: Signal store (uses global if not provided)
            document_loader: Function to load document text by ID
        """
        self.store = store or get_signal_store()
        self.document_loader = document_loader
        
        self.reviewed_count = 0
        self.session_start = datetime.utcnow()
    
    def run(self, limit: Optional[int] = None) -> int:
        """Run interactive review session.
        
        Args:
            limit: Maximum signals to review (None for all pending)
            
        Returns:
            Number of signals reviewed
        """
        # Get pending signals
        signals = self.store.get_pending_signals(limit=limit or 1000)
        total = len(signals)
        
        if total == 0:
            console.print()
            console.print("[green]✓ No signals pending review![/]")
            console.print()
            return 0
        
        console.print()
        console.print(f"[bold]Starting review session: {total} signals pending[/]")
        console.print("[dim]Press 'q' at any time to quit[/]")
        
        # Document text cache
        doc_cache = {}
        
        for i, signal in enumerate(signals, 1):
            # Load document text if available
            doc_text = None
            if self.document_loader and signal.document_id:
                if signal.document_id not in doc_cache:
                    try:
                        doc_cache[signal.document_id] = self.document_loader(signal.document_id)
                    except Exception:
                        doc_cache[signal.document_id] = None
                doc_text = doc_cache.get(signal.document_id)
            
            # Display signal
            display_signal_for_review(signal, i, total, doc_text)
            display_menu()
            
            # Get input
            choice = Prompt.ask("Choice", choices=["tp", "wl", "fp", "fn", "s", "q"], default="s")
            
            if choice == "q":
                console.print()
                console.print("[yellow]Session ended by user[/]")
                break
            
            if choice == "s":
                continue
            
            # Process feedback
            if choice == "tp":
                self.store.record_feedback(signal.id, FeedbackType.TRUE_POSITIVE)
                console.print("[green]✓ Marked as True Positive[/]")
                self.reviewed_count += 1
                
            elif choice == "wl":
                correct_type = get_entity_type_selection()
                if correct_type:
                    self.store.record_feedback(
                        signal.id, 
                        FeedbackType.WRONG_LABEL,
                        correct_type=correct_type,
                    )
                    console.print(f"[yellow]✓ Marked as Wrong Label → {correct_type}[/]")
                    self.reviewed_count += 1
                else:
                    console.print("[dim]Cancelled[/]")
                    
            elif choice == "fp":
                self.store.record_feedback(signal.id, FeedbackType.FALSE_POSITIVE)
                console.print("[red]✓ Marked as False Positive[/]")
                self.reviewed_count += 1
                
            elif choice == "fn":
                missed_text, entity_type = prompt_for_missed_text()
                if missed_text and entity_type:
                    self.store.report_false_negative(
                        document_id=signal.document_id,
                        missed_text=missed_text,
                        correct_type=entity_type,
                        conversation_id=signal.conversation_id,
                    )
                    console.print(f"[magenta]✓ Reported missed: '{missed_text}' as {entity_type}[/]")
                    self.reviewed_count += 1
                else:
                    console.print("[dim]Cancelled[/]")
        
        # Summary
        self._display_summary(total)
        
        return self.reviewed_count
    
    def _display_summary(self, total: int):
        """Display session summary."""
        duration = datetime.utcnow() - self.session_start
        minutes = duration.total_seconds() / 60
        
        console.print()
        console.print("═" * 70, style="dim")
        console.print("[bold]Session Summary[/]")
        console.print()
        console.print(f"  Reviewed: {self.reviewed_count}/{total}")
        console.print(f"  Duration: {minutes:.1f} minutes")
        if self.reviewed_count > 0:
            rate = minutes / self.reviewed_count
            console.print(f"  Rate: {rate:.1f} min/signal")
        
        # Show remaining
        remaining = self.store.get_pending_count()
        console.print()
        console.print(f"  Pending: {remaining} signals remaining")
        
        # Check training threshold
        labeled = self.store.get_labeled_signal_count()
        console.print(f"  Labeled: {labeled} total (1000 triggers retrain)")
        
        if labeled >= 1000:
            console.print()
            console.print("[bold green]✓ Training threshold reached![/]")
            console.print("[dim]Run 'phi-train learn retrain' to train new model[/]")
        
        console.print()


# =============================================================================
# STATS DISPLAY
# =============================================================================

def display_stats(store: Optional[SignalStore] = None):
    """Display signal store statistics."""
    store = store or get_signal_store()
    stats = store.get_stats()
    
    console.print()
    console.print("[bold]Signal Store Statistics[/]")
    console.print("─" * 40)
    console.print()
    
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Active signals", str(stats['total_active']))
    table.add_row("  Pending review", str(stats['pending_review']))
    table.add_row("  Reviewed", str(stats['reviewed']))
    table.add_row("False negatives pending", str(stats['false_negatives_pending']))
    table.add_row("Archived signals", str(stats['archived']))
    table.add_row("", "")
    table.add_row("Active model", stats['active_model'] or "none")
    if stats['active_model_f1']:
        table.add_row("  F1 score", f"{stats['active_model_f1']:.1%}")
    
    console.print(table)
    
    # Training status
    labeled = store.get_labeled_signal_count()
    console.print()
    console.print(f"Labeled signals: {labeled}/1000 for next training run")
    
    if labeled >= 1000:
        console.print("[green]✓ Ready for training![/]")
    else:
        console.print(f"[dim]{1000 - labeled} more needed[/]")
    
    console.print()


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def run_review_tui(
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    """Run the review TUI.
    
    Args:
        limit: Max signals to review
        db_path: Path to signal store database
        
    Returns:
        Number of signals reviewed
    """
    if db_path:
        store = SignalStore(db_path)
    else:
        store = get_signal_store()
    
    session = ReviewSession(store=store)
    return session.run(limit=limit)


def show_stats_tui(db_path: Optional[str] = None):
    """Show signal store stats."""
    if db_path:
        store = SignalStore(db_path)
    else:
        store = get_signal_store()
    
    display_stats(store)
