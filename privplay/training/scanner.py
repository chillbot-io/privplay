"""Batch scanning of documents."""

from typing import Optional, List
from datetime import datetime
import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..types import Document, Entity
from ..db import Database, get_db
from ..config import get_config
from ..engine.classifier import ClassificationEngine

console = Console()
logger = logging.getLogger(__name__)


def scan_documents(
    engine: Optional[ClassificationEngine] = None,
    db: Optional[Database] = None,
    verify: bool = True,
    rescan: bool = False,
) -> tuple[int, int]:
    """
    Scan all unscanned documents.
    
    Args:
        engine: Classification engine to use
        db: Database instance
        verify: Whether to run LLM verification
        rescan: If True, rescan all documents (not just unscanned)
        
    Returns:
        Tuple of (documents_scanned, entities_found)
    """
    if db is None:
        db = get_db()
    
    if engine is None:
        engine = ClassificationEngine()
    
    config = get_config()
    
    # Get documents to scan
    if rescan:
        # TODO: implement get_all_documents
        documents = db.get_unscanned_documents()
    else:
        documents = db.get_unscanned_documents()
    
    if not documents:
        console.print("[dim]No documents to scan.[/dim]")
        return 0, 0
    
    total_entities = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning documents...", total=len(documents))
        
        for doc in documents:
            try:
                # Detect entities
                entities = engine.detect(
                    doc.content, 
                    verify=verify,
                    threshold=config.confidence_threshold
                )
                
                # Store entities
                if entities:
                    db.add_entities(entities, doc.id)
                    total_entities += len(entities)
                
                # Mark document as scanned
                db.mark_document_scanned(doc.id)
                
            except Exception as e:
                logger.error(f"Failed to scan document {doc.id}: {e}")
            
            progress.advance(task)
    
    # Summary
    above_threshold = db.count_entities_above_threshold(config.confidence_threshold)
    below_threshold = db.count_entities_needing_review(config.confidence_threshold)
    
    console.print()
    console.print(f"[green]✓ Scanned {len(documents)} documents[/green]")
    console.print(f"  Found {total_entities} entities")
    console.print(f"    [green]≥{config.confidence_threshold:.0%} confidence: {above_threshold} (auto-approved)[/green]")
    console.print(f"    [yellow]<{config.confidence_threshold:.0%} confidence: {below_threshold} (need review)[/yellow]")
    console.print()
    
    return len(documents), total_entities


def scan_single_document(
    content: str,
    engine: Optional[ClassificationEngine] = None,
    verify: bool = True,
) -> List[Entity]:
    """
    Scan a single piece of text (without storing).
    
    Useful for testing/debugging.
    """
    if engine is None:
        engine = ClassificationEngine()
    
    return engine.detect(content, verify=verify)


def display_scan_results(entities: List[Entity], text: str) -> None:
    """Display scan results for a single text."""
    config = get_config()
    
    if not entities:
        console.print("[dim]No PHI/PII detected.[/dim]")
        return
    
    console.print()
    console.print(f"[bold]Found {len(entities)} entities:[/bold]")
    console.print()
    
    for entity in sorted(entities, key=lambda e: e.start):
        # Determine status
        if entity.confidence >= config.confidence_threshold:
            status = "[green]✓[/green]"
        else:
            status = "[yellow]?[/yellow]"
        
        # Get short context
        start = max(0, entity.start - 20)
        end = min(len(text), entity.end + 20)
        context = text[start:entity.start] + f"[{entity.text}]" + text[entity.end:end]
        
        console.print(f"  {status} [{entity.text}]")
        console.print(f"      Type: {entity.entity_type.value}")
        console.print(f"      NER: {entity.confidence:.2f}", end="")
        
        if entity.llm_confidence is not None:
            console.print(f"  LLM: {entity.llm_confidence:.2f}")
        else:
            console.print()
        
        console.print(f"      Context: ...{context}...")
        console.print()
