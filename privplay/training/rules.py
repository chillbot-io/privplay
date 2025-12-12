"""Rule management for adding custom detection patterns."""

import json
import re
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from ..types import EntityType

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CustomRule:
    """A custom detection rule."""
    name: str
    pattern: str
    entity_type: str
    confidence: float = 0.90
    description: str = ""
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CustomRule":
        return cls(**d)


class RuleManager:
    """Manages custom rules - add, test, save, load."""
    
    def __init__(self, rules_path: Optional[Path] = None):
        if rules_path is None:
            from ..config import get_config
            rules_path = get_config().data_dir / "custom_rules.json"
        
        self.rules_path = rules_path
        self.custom_rules: List[CustomRule] = []
        self._load()
    
    def _load(self):
        """Load custom rules from file."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path) as f:
                    data = json.load(f)
                self.custom_rules = [
                    CustomRule.from_dict(r) for r in data.get("rules", [])
                ]
                logger.info(f"Loaded {len(self.custom_rules)} custom rules")
            except Exception as e:
                logger.error(f"Failed to load custom rules: {e}")
                self.custom_rules = []
        else:
            self.custom_rules = []
    
    def save(self):
        """Save custom rules to file."""
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "rules": [r.to_dict() for r in self.custom_rules]
        }
        
        with open(self.rules_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.custom_rules)} rules to {self.rules_path}")
    
    def add_rule(self, rule: CustomRule) -> bool:
        """Add a new custom rule."""
        # Validate pattern compiles
        try:
            re.compile(rule.pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return False
        
        # Validate entity type
        try:
            EntityType(rule.entity_type)
        except ValueError:
            logger.error(f"Invalid entity type: {rule.entity_type}")
            return False
        
        # Check for duplicate names
        if any(r.name == rule.name for r in self.custom_rules):
            logger.error(f"Rule with name '{rule.name}' already exists")
            return False
        
        self.custom_rules.append(rule)
        self.save()
        return True
    
    def remove_rule(self, name: str) -> bool:
        """Remove a custom rule by name."""
        for i, rule in enumerate(self.custom_rules):
            if rule.name == name:
                del self.custom_rules[i]
                self.save()
                return True
        return False
    
    def get_rule(self, name: str) -> Optional[CustomRule]:
        """Get a rule by name."""
        for rule in self.custom_rules:
            if rule.name == name:
                return rule
        return None
    
    def list_rules(self) -> List[CustomRule]:
        """Get all custom rules."""
        return self.custom_rules
    
    def test_pattern(
        self, 
        pattern: str, 
        text: str,
        flags: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Test a regex pattern against text.
        
        Returns list of matches with position info.
        """
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return [{"error": str(e)}]
        
        matches = []
        for match in regex.finditer(text):
            matches.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "groups": match.groups() if match.groups() else None,
            })
        
        return matches
    
    def register_with_engine(self, rule_engine):
        """Register all custom rules with the rule engine."""
        from ..engine.rules.engine import Rule
        
        for custom in self.custom_rules:
            if not custom.enabled:
                continue
            
            try:
                rule_engine.add_rule(Rule(
                    name=f"custom_{custom.name}",
                    pattern=re.compile(custom.pattern),
                    entity_type=EntityType(custom.entity_type),
                    confidence=custom.confidence,
                ))
                logger.debug(f"Registered custom rule: {custom.name}")
            except Exception as e:
                logger.error(f"Failed to register rule {custom.name}: {e}")


def display_rules(rules: List[CustomRule], show_patterns: bool = False):
    """Display rules in a nice table."""
    if not rules:
        console.print("[dim]No custom rules defined.[/dim]")
        return
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Entity Type")
    table.add_column("Confidence")
    table.add_column("Enabled")
    if show_patterns:
        table.add_column("Pattern")
    table.add_column("Description")
    
    for rule in rules:
        enabled = "[green]✓[/green]" if rule.enabled else "[red]✗[/red]"
        row = [
            rule.name,
            rule.entity_type,
            f"{rule.confidence:.0%}",
            enabled,
        ]
        if show_patterns:
            # Truncate long patterns
            pattern = rule.pattern if len(rule.pattern) < 40 else rule.pattern[:37] + "..."
            row.append(f"[dim]{pattern}[/dim]")
        row.append(rule.description or "[dim]-[/dim]")
        table.add_row(*row)
    
    console.print(table)


def display_test_results(pattern: str, text: str, matches: List[Dict]):
    """Display pattern test results."""
    console.print()
    console.print("[bold]Pattern Test Results[/bold]")
    console.print("─" * 50)
    
    if matches and "error" in matches[0]:
        console.print(f"[red]Invalid pattern: {matches[0]['error']}[/red]")
        return
    
    console.print(f"Pattern: [cyan]{pattern}[/cyan]")
    console.print(f"Matches: [{'green' if matches else 'yellow'}]{len(matches)}[/]")
    console.print()
    
    if matches:
        # Show text with highlights
        highlighted = text
        offset = 0
        
        for m in sorted(matches, key=lambda x: x["start"]):
            start = m["start"] + offset
            end = m["end"] + offset
            
            before = highlighted[:start]
            match_text = highlighted[start:end]
            after = highlighted[end:]
            
            highlighted = before + f"[bold red]{match_text}[/bold red]" + after
            offset += len("[bold red]") + len("[/bold red]")
        
        console.print(Panel(highlighted, title="Matches highlighted"))
        console.print()
        
        # Show match details
        for i, m in enumerate(matches):
            console.print(f"  {i+1}. \"{m['text']}\" @ {m['start']}-{m['end']}")
    else:
        console.print("[yellow]No matches found.[/yellow]")
        console.print()
        console.print(Panel(text, title="Input text"))


def interactive_add_rule() -> Optional[CustomRule]:
    """Interactive rule addition wizard."""
    import typer
    
    console.print()
    console.print("[bold]Add Custom Rule[/bold]")
    console.print("─" * 40)
    console.print()
    
    # Name
    name = typer.prompt("Rule name (e.g., 'company_id')")
    if not name:
        console.print("[red]Name required[/red]")
        return None
    
    # Pattern
    console.print()
    console.print("[dim]Enter regex pattern. Examples:[/dim]")
    console.print("[dim]  SSN:    \\b\\d{3}-\\d{2}-\\d{4}\\b[/dim]")
    console.print("[dim]  Email:  \\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b[/dim]")
    console.print()
    
    pattern = typer.prompt("Regex pattern")
    
    # Validate pattern
    try:
        re.compile(pattern)
    except re.error as e:
        console.print(f"[red]Invalid regex: {e}[/red]")
        return None
    
    # Entity type
    console.print()
    console.print("[dim]Available entity types:[/dim]")
    types = [e.value for e in EntityType]
    for i, t in enumerate(types):
        if i % 4 == 0:
            console.print()
        console.print(f"  {t}", end="")
    console.print()
    console.print()
    
    entity_type = typer.prompt("Entity type").upper()
    
    try:
        EntityType(entity_type)
    except ValueError:
        console.print(f"[red]Invalid entity type: {entity_type}[/red]")
        return None
    
    # Confidence
    confidence = typer.prompt("Confidence (0.0-1.0)", default="0.90")
    try:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError()
    except ValueError:
        console.print("[red]Invalid confidence[/red]")
        return None
    
    # Description
    description = typer.prompt("Description (optional)", default="")
    
    # Test it?
    console.print()
    if typer.confirm("Test pattern before saving?", default=True):
        test_text = typer.prompt("Enter test text")
        matches = RuleManager().test_pattern(pattern, test_text)
        display_test_results(pattern, test_text, matches)
        
        if not matches:
            if not typer.confirm("No matches found. Save anyway?", default=False):
                return None
    
    return CustomRule(
        name=name,
        pattern=pattern,
        entity_type=entity_type,
        confidence=confidence,
        description=description,
        enabled=True,
    )


def suggest_rules_from_corrections() -> List[CustomRule]:
    """
    Analyze corrections to suggest new rules.
    
    Looks for patterns in false positives (rejected) that could 
    become negative rules, and patterns in confirmed entities
    that could become positive rules.
    """
    from ..db import get_db
    from collections import Counter
    
    db = get_db()
    corrections = db.get_corrections()
    
    suggestions = []
    
    # Find common rejected patterns (false positives)
    rejected = [c for c in corrections if c.decision.value == "rejected"]
    rejected_texts = Counter(c.entity_text.lower() for c in rejected)
    
    # If same text rejected multiple times, suggest exclusion
    for text, count in rejected_texts.most_common(10):
        if count >= 2:
            # Create a pattern that matches this exact text
            escaped = re.escape(text)
            suggestions.append({
                "type": "exclusion",
                "text": text,
                "count": count,
                "suggested_pattern": f"\\b{escaped}\\b",
                "reason": f"Rejected {count} times",
            })
    
    # Find patterns in confirmed entities that rules might miss
    confirmed = [c for c in corrections if c.decision.value == "confirmed"]
    
    # Group by type and look for common patterns
    by_type = {}
    for c in confirmed:
        etype = c.correct_type or c.detected_type
        if etype.value not in by_type:
            by_type[etype.value] = []
        by_type[etype.value].append(c.entity_text)
    
    # TODO: More sophisticated pattern mining
    # For now, just return rejection suggestions
    
    return suggestions
