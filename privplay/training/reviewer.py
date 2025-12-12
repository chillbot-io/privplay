"""Interactive review session for training."""

from typing import Optional, List, Tuple
import uuid
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from ..types import (
    Entity, Document, Correction, EntityType, 
    DecisionType, ReviewStats
)
from ..db import Database, get_db
from ..config import get_config

console = Console()


def highlight_entity_in_context(
    text: str, 
    entity: Entity, 
    context_window: int = 100
) -> Text:
    """Create highlighted text showing entity in context."""
    start = max(0, entity.start - context_window)
    end = min(len(text), entity.end + context_window)
    
    # Build rich text
    result = Text()
    
    # Add ellipsis if truncated
    if start > 0:
        result.append("...", style="dim")
    
    # Before entity
    result.append(text[start:entity.start])
    
    # Entity (highlighted)
    result.append(f"[{entity.text}]", style="bold yellow on dark_red")
    
    # After entity
    result.append(text[entity.end:end])
    
    # Add ellipsis if truncated
    if end < len(text):
        result.append("...", style="dim")
    
    return result


def display_entity_for_review(
    entity: Entity, 
    document: Document,
    index: int,
    total: int,
    session_reviewed: int,
) -> None:
    """Display an entity for human review."""
    # Header
    progress_pct = (index / total * 100) if total > 0 else 0
    header = f"Document {index}/{total}  |  Progress: {progress_pct:.1f}%  |  Session: {session_reviewed} reviewed"
    
    console.print()
    console.print("─" * 70)
    console.print(header, style="dim")
    console.print()
    
    # Context
    highlighted = highlight_entity_in_context(
        document.content, 
        entity,
        context_window=get_config().context_window
    )
    console.print(Panel(highlighted, title="Context", box=box.ROUNDED))
    
    # Entity info
    console.print()
    console.print(f"  [{entity.text}]", style="bold yellow")
    console.print(f"  detected as: {entity.entity_type.value}", style="cyan")
    console.print()
    
    # Confidence scores
    console.print(f"    NER confidence: {entity.confidence:.2f}", style="dim")
    if entity.llm_confidence is not None:
        llm_style = "green" if entity.llm_confidence > 0.7 else "red" if entity.llm_confidence < 0.3 else "yellow"
        console.print(f"    LLM confidence: {entity.llm_confidence:.2f}", style=llm_style)
        if entity.llm_reasoning:
            console.print(f"    LLM reasoning: {entity.llm_reasoning}", style="dim italic")
    
    console.print()


def prompt_for_decision() -> Tuple[str, Optional[EntityType]]:
    """Prompt user for review decision."""
    console.print("  [a] Correct - this IS PHI/PII")
    console.print("  [r] Reject  - this is NOT PHI/PII")
    console.print("  [c] Change type - PHI but wrong category")
    console.print("  [s] Skip for now")
    console.print("  [q] Quit and save progress")
    console.print()
    
    choice = Prompt.ask("> ", choices=["a", "r", "c", "s", "q"], default="s")
    
    new_type = None
    if choice == "c":
        new_type = prompt_for_entity_type()
    
    return choice, new_type


def prompt_for_entity_type() -> EntityType:
    """Prompt user to select correct entity type."""
    console.print()
    console.print("Select correct type:")
    
    types = list(EntityType)
    for i, t in enumerate(types, 1):
        console.print(f"  [{i:2d}] {t.value}")
    
    while True:
        try:
            choice = Prompt.ask("Enter number")
            idx = int(choice) - 1
            if 0 <= idx < len(types):
                return types[idx]
        except ValueError:
            pass
        console.print("Invalid choice, try again", style="red")


def run_review_session(
    threshold: float = 0.95,
    entity_type: Optional[EntityType] = None,
    limit: int = 100,
    db: Optional[Database] = None,
) -> int:
    """
    Run interactive review session.
    
    Returns number of items reviewed.
    """
    if db is None:
        db = get_db()
    
    config = get_config()
    
    # Get entities needing review
    items = db.get_entities_for_review(
        threshold=threshold,
        entity_type=entity_type,
        limit=limit
    )
    
    if not items:
        console.print()
        console.print("[green]✓ No items pending review![/green]")
        console.print()
        return 0
    
    console.print()
    console.print(f"[bold]Starting review session[/bold]")
    console.print(f"  {len(items)} items below {threshold:.0%} confidence")
    if entity_type:
        console.print(f"  Filtering: {entity_type.value}")
    console.print()
    
    session_reviewed = 0
    
    for i, (entity, document) in enumerate(items, 1):
        display_entity_for_review(
            entity=entity,
            document=document,
            index=i,
            total=len(items),
            session_reviewed=session_reviewed,
        )
        
        choice, new_type = prompt_for_decision()
        
        if choice == "q":
            console.print()
            console.print(f"[yellow]Session ended. Reviewed {session_reviewed} items.[/yellow]")
            break
        
        if choice == "s":
            continue
        
        # Record decision
        decision = {
            "a": DecisionType.CONFIRMED,
            "r": DecisionType.REJECTED,
            "c": DecisionType.CHANGED,
        }[choice]
        
        # Get context
        context_before, context_after = _get_context(
            document.content, 
            entity.start, 
            entity.end,
            config.context_window
        )
        
        correction = Correction(
            id=str(uuid.uuid4()),
            entity_id=entity.id,
            document_id=document.id,
            entity_text=entity.text,
            entity_start=entity.start,
            entity_end=entity.end,
            detected_type=entity.entity_type,
            decision=decision,
            correct_type=new_type,
            context_before=context_before,
            context_after=context_after,
            ner_confidence=entity.confidence,
            llm_confidence=entity.llm_confidence,
            reviewed_at=datetime.utcnow(),
        )
        
        db.add_correction(correction)
        session_reviewed += 1
        
        # Show confirmation
        if decision == DecisionType.REJECTED:
            console.print(f"  [red]✗ Marked as NOT PHI[/red]")
        elif decision == DecisionType.CONFIRMED:
            console.print(f"  [green]✓ Confirmed as {entity.entity_type.value}[/green]")
        else:
            console.print(f"  [yellow]→ Changed to {new_type.value}[/yellow]")
    
    console.print()
    console.print(f"[bold green]Session complete. Reviewed {session_reviewed} items.[/bold green]")
    console.print()
    
    return session_reviewed


def _get_context(text: str, start: int, end: int, window: int) -> Tuple[str, str]:
    """Get context before and after entity."""
    context_before = text[max(0, start - window):start]
    context_after = text[end:min(len(text), end + window)]
    return context_before, context_after


def display_stats(stats: ReviewStats, threshold: float = 0.95) -> None:
    """Display review statistics."""
    console.print()
    console.print("[bold]Training Progress[/bold]")
    console.print("─" * 40)
    console.print(f"  Total documents:  {stats.total_documents}")
    console.print(f"  Total entities:   {stats.total_entities}")
    console.print()
    console.print(f"  [green]Auto-approved (≥{threshold:.0%}): {stats.auto_approved}[/green]")
    console.print(f"  [yellow]Pending review:     {stats.pending}[/yellow]")
    console.print(f"  [blue]Reviewed:           {stats.reviewed}[/blue]")
    console.print()
    
    if stats.reviewed > 0:
        console.print("[bold]Decisions breakdown:[/bold]")
        total_decisions = stats.confirmed + stats.rejected + stats.changed
        if total_decisions > 0:
            console.print(f"  [green]Confirmed correct:  {stats.confirmed} ({stats.confirmed/total_decisions*100:.0f}%)[/green]")
            console.print(f"  [red]Rejected (FP):      {stats.rejected} ({stats.rejected/total_decisions*100:.0f}%)[/red]")
            console.print(f"  [yellow]Changed type:       {stats.changed} ({stats.changed/total_decisions*100:.0f}%)[/yellow]")
    
    console.print()


def display_top_fps(db: Optional[Database] = None, limit: int = 10) -> None:
    """Display top false positive patterns."""
    if db is None:
        db = get_db()
    
    fps = db.get_top_fp_patterns(limit=limit)
    
    if not fps:
        console.print("[dim]No false positives recorded yet.[/dim]")
        return
    
    console.print()
    console.print("[bold]Top False Positive Patterns[/bold]")
    console.print("─" * 40)
    
    for text, detected_type, count in fps:
        console.print(f"  {text:20s} {detected_type:15s} ({count}x)")
    
    console.print()
