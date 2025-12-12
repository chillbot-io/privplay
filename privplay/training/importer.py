"""Import documents from files."""

from pathlib import Path
from typing import List, Optional
import uuid
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..types import Document
from ..db import Database, get_db

console = Console()


def import_from_directory(
    path: Path,
    db: Optional[Database] = None,
    extensions: tuple = (".txt", ".text"),
) -> int:
    """
    Import documents from a directory.
    
    Args:
        path: Directory path
        db: Database instance
        extensions: File extensions to import
        
    Returns:
        Number of documents imported
    """
    if db is None:
        db = get_db()
    
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        return 0
    
    if not path.is_dir():
        console.print(f"[red]Error: Not a directory: {path}[/red]")
        return 0
    
    # Find files
    files = []
    for ext in extensions:
        files.extend(path.glob(f"*{ext}"))
        files.extend(path.glob(f"**/*{ext}"))
    
    if not files:
        console.print(f"[yellow]No files found with extensions {extensions}[/yellow]")
        return 0
    
    imported = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Importing files...", total=len(files))
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                
                doc = Document(
                    id=str(uuid.uuid4()),
                    content=content,
                    source=f"file:{file_path}",
                )
                
                db.add_document(doc)
                imported += 1
                
            except Exception as e:
                console.print(f"[red]Failed to import {file_path}: {e}[/red]")
            
            progress.advance(task)
    
    console.print(f"[green]✓ Imported {imported} documents[/green]")
    return imported


def import_from_json(
    path: Path,
    db: Optional[Database] = None,
) -> int:
    """
    Import documents from a JSON file.
    
    Expected format:
    {
        "documents": [
            {"content": "...", "source": "..."},
            ...
        ]
    }
    
    Or array format:
    [
        {"content": "..."},
        ...
    ]
    """
    if db is None:
        db = get_db()
    
    if not path.exists():
        console.print(f"[red]Error: File does not exist: {path}[/red]")
        return 0
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON: {e}[/red]")
        return 0
    
    # Handle different formats
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "documents" in data:
        items = data["documents"]
    else:
        console.print("[red]Error: Invalid format. Expected list or {\"documents\": [...]}[/red]")
        return 0
    
    imported = 0
    
    for item in items:
        if isinstance(item, str):
            content = item
            source = f"json:{path}"
        elif isinstance(item, dict):
            content = item.get("content", item.get("text", ""))
            source = item.get("source", f"json:{path}")
        else:
            continue
        
        if not content:
            continue
        
        doc = Document(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
        )
        
        db.add_document(doc)
        imported += 1
    
    console.print(f"[green]✓ Imported {imported} documents from JSON[/green]")
    return imported


def import_single_text(
    content: str,
    source: str = "manual",
    db: Optional[Database] = None,
) -> str:
    """
    Import a single text document.
    
    Returns document ID.
    """
    if db is None:
        db = get_db()
    
    doc = Document(
        id=str(uuid.uuid4()),
        content=content,
        source=source,
    )
    
    db.add_document(doc)
    return doc.id
