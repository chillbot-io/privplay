#!/usr/bin/env python3
"""
Create FULLY ANNOTATED golden PII dataset from AI4Privacy.

Key difference: Keeps ALL entities per sample, not just one.
Each sample has complete annotations for every PII element.

This prevents the model from learning that unlabeled text is "not PII"
when it actually is PII that we just didn't annotate.
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# =============================================================================
# TYPE ALIASES - Map AI4Privacy labels to our standard
# =============================================================================

TYPE_ALIASES = {
    # Names
    "GIVENNAME": "NAME_PERSON",
    "SURNAME": "NAME_PERSON",
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON", 
    "MIDDLENAME": "NAME_PERSON",
    "NAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "TITLE": "NAME_PERSON",
    
    # Contact
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    
    # IDs
    "SSN": "SSN",
    "SOCIALNUM": "SSN",
    "IDCARDNUM": "ID_NUMBER",
    "DRIVERLICENSENUM": "DRIVER_LICENSE",
    "TAXNUM": "TAX_ID",
    
    # Financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN": "IBAN",
    "ACCOUNTNUM": "ACCOUNT_NUMBER",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "BIC": "ACCOUNT_NUMBER",
    
    # Location
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "COUNTRY": "LOCATION",
    "ZIPCODE": "ZIP",
    "POSTCODE": "ZIP",
    
    # Online
    "USERNAME": "USERNAME",
    "USERAGENT": "USERNAME",
    "PASSWORD": "PASSWORD",
    "PIN": "PASSWORD",
    "URL": "URL",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IP": "IP_ADDRESS",
    "MAC": "MAC_ADDRESS",
    
    # Dates
    "DATEOFBIRTH": "DATE",
    "DATE": "DATE",
    "DOB": "DATE",
    "TIME": "TIME",
    
    # Other
    "COMPANYNAME": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "VEHICLEVIN": "VIN",
    "VEHICLEVRM": "VIN",
    "IMEI": "DEVICE_ID",
}

# Types we want to include (skip obscure ones)
INCLUDE_TYPES = {
    "NAME_PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD", 
    "ADDRESS", "LOCATION", "ZIP", "DATE", "TIME",
    "USERNAME", "PASSWORD", "URL", "IP_ADDRESS",
    "ACCOUNT_NUMBER", "IBAN", "DRIVER_LICENSE", "ID_NUMBER",
    "ORGANIZATION", "TAX_ID", "VIN", "DEVICE_ID", "MAC_ADDRESS",
}

# =============================================================================
# VALIDATORS - Type-specific quality checks
# =============================================================================

def validate_name(text: str) -> Tuple[bool, float]:
    """Validate person name."""
    text = text.strip()
    if len(text) < 2:
        return False, 0.0
    if not any(c.isalpha() for c in text):
        return False, 0.0
    # Reject common non-names
    lower = text.lower()
    reject_words = {'the', 'a', 'an', 'mr', 'mrs', 'ms', 'dr', 'miss', 'mister', 
                    'master', 'madame', 'sir', 'prof', 'sen', 'rep'}
    if lower in reject_words:
        return False, 0.0
    return True, 0.90


def validate_email(text: str) -> Tuple[bool, float]:
    """Validate email address."""
    if '@' not in text or '.' not in text.split('@')[-1]:
        return False, 0.0
    return True, 0.95


def validate_phone(text: str) -> Tuple[bool, float]:
    """Validate phone number."""
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) < 7:
        return False, 0.0
    return True, 0.90


def validate_ssn(text: str) -> Tuple[bool, float]:
    """Validate SSN."""
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) != 9:
        return False, 0.0
    area = int(digits[:3])
    if area == 0 or area == 666 or area >= 900:
        return False, 0.0
    return True, 0.95


def validate_credit_card(text: str) -> Tuple[bool, float]:
    """Validate credit card (Luhn check)."""
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) < 13 or len(digits) > 19:
        return False, 0.0
    
    # Luhn check
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    
    if total % 10 != 0:
        return False, 0.0
    return True, 0.95


def validate_address(text: str) -> Tuple[bool, float]:
    """Validate address."""
    if len(text) < 3:
        return False, 0.0
    # Should have some content
    if text.isdigit():  # Just a number isn't an address
        return False, 0.0
    return True, 0.80


def validate_date(text: str) -> Tuple[bool, float]:
    """Validate date."""
    # Basic check - has digits
    if not any(c.isdigit() for c in text):
        return False, 0.0
    return True, 0.85


def validate_username(text: str) -> Tuple[bool, float]:
    """Validate username."""
    if len(text) < 2:
        return False, 0.0
    return True, 0.85


def validate_password(text: str) -> Tuple[bool, float]:
    """Validate password."""
    if len(text) < 4:
        return False, 0.0
    return True, 0.80


def validate_url(text: str) -> Tuple[bool, float]:
    """Validate URL."""
    if not (text.startswith('http') or text.startswith('www.') or '.' in text):
        return False, 0.0
    return True, 0.90


def validate_ip(text: str) -> Tuple[bool, float]:
    """Validate IP address."""
    parts = text.split('.')
    if len(parts) == 4:
        try:
            if all(0 <= int(p) <= 255 for p in parts):
                return True, 0.95
        except:
            pass
    return False, 0.0


def validate_generic(text: str) -> Tuple[bool, float]:
    """Generic validation for other types."""
    if len(text) < 2:
        return False, 0.0
    return True, 0.75


# Validator map
VALIDATORS = {
    "NAME_PERSON": validate_name,
    "EMAIL": validate_email,
    "PHONE": validate_phone,
    "SSN": validate_ssn,
    "CREDIT_CARD": validate_credit_card,
    "ADDRESS": validate_address,
    "LOCATION": validate_address,
    "ZIP": validate_generic,
    "DATE": validate_date,
    "TIME": validate_generic,
    "USERNAME": validate_username,
    "PASSWORD": validate_password,
    "URL": validate_url,
    "IP_ADDRESS": validate_ip,
    "ACCOUNT_NUMBER": validate_generic,
    "IBAN": validate_generic,
    "DRIVER_LICENSE": validate_generic,
    "ID_NUMBER": validate_generic,
    "ORGANIZATION": validate_generic,
    "TAX_ID": validate_generic,
    "VIN": validate_generic,
    "DEVICE_ID": validate_generic,
    "MAC_ADDRESS": validate_generic,
}


def normalize_type(label: str) -> Optional[str]:
    """Normalize entity type, return None if we should skip."""
    label_upper = label.upper().replace(" ", "").replace("_", "")
    
    # Try lookup
    normalized = TYPE_ALIASES.get(label_upper, TYPE_ALIASES.get(label.upper()))
    
    if normalized and normalized in INCLUDE_TYPES:
        return normalized
    
    return None


def validate_entity(entity_type: str, value: str) -> Tuple[bool, float]:
    """Validate an entity."""
    validator = VALIDATORS.get(entity_type, validate_generic)
    return validator(value)


def curate_golden_dataset(
    max_samples: int = 25000,
    min_entities_per_sample: int = 1,
    output_path: str = "golden_dataset.json",
):
    """Create golden dataset with FULL annotations per sample."""
    
    console.print(f"\n[bold cyan]═══ Creating Fully Annotated Golden Dataset ═══[/bold cyan]\n")
    
    # Load AI4Privacy
    try:
        from datasets import load_dataset
        console.print("Loading AI4Privacy from Hugging Face...")
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
        console.print("[green]✓ Dataset loaded (streaming)[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    samples = []
    type_counts = Counter()
    skipped_samples = 0
    skipped_entities = Counter()
    
    console.print(f"\nProcessing samples (target: {max_samples})...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} samples"),
        console=console,
    ) as progress:
        task = progress.add_task("Curating...", total=max_samples)
        
        for row in ds:
            if len(samples) >= max_samples:
                break
            
            text = row.get("source_text", "")
            raw_entities = row.get("privacy_mask", [])
            
            if not text or not raw_entities:
                skipped_samples += 1
                continue
            
            # Process ALL entities in this sample
            valid_entities = []
            
            for entity in raw_entities:
                raw_label = entity.get("label", "")
                value = entity.get("value", "")
                start = entity.get("start")
                end = entity.get("end")
                
                if not value or start is None or end is None:
                    continue
                
                # Normalize type
                norm_type = normalize_type(raw_label)
                if not norm_type:
                    skipped_entities[raw_label] += 1
                    continue
                
                # Validate
                is_valid, confidence = validate_entity(norm_type, value)
                if not is_valid:
                    skipped_entities[f"{norm_type}_invalid"] += 1
                    continue
                
                # Verify span matches text
                if start >= 0 and end <= len(text):
                    actual_text = text[start:end]
                    # Allow some flexibility for whitespace
                    if actual_text.strip() == value.strip() or value in actual_text or actual_text in value:
                        valid_entities.append({
                            "label": norm_type,
                            "value": value,
                            "start": start,
                            "end": end,
                            "quality_confidence": confidence,
                        })
                        type_counts[norm_type] += 1
            
            # Only keep sample if it has valid entities
            if len(valid_entities) >= min_entities_per_sample:
                samples.append({
                    "text": text,
                    "entities": valid_entities,
                    "quality_score": sum(e["quality_confidence"] for e in valid_entities) / len(valid_entities),
                    "source": "ai4privacy",
                })
                progress.update(task, completed=len(samples))
    
    # Shuffle
    random.shuffle(samples)
    
    # Calculate stats
    total_entities = sum(type_counts.values())
    avg_entities = total_entities / len(samples) if samples else 0
    
    # Save
    output = {
        "samples": samples,
        "metadata": {
            "total_samples": len(samples),
            "total_entities": total_entities,
            "avg_entities_per_sample": round(avg_entities, 2),
            "entity_type_counts": dict(type_counts),
            "fully_annotated": True,
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f)
    
    # Print summary
    console.print(f"\n[bold green]═══ Golden Dataset Complete ═══[/bold green]\n")
    
    console.print(f"[bold]Samples:[/bold] {len(samples):,}")
    console.print(f"[bold]Total entities:[/bold] {total_entities:,}")
    console.print(f"[bold]Avg entities/sample:[/bold] {avg_entities:.1f}")
    
    # Entity distribution table
    table = Table(title="\nEntity Distribution")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")
    
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_entities * 100
        table.add_row(etype, f"{count:,}", f"{pct:.1f}%")
    
    console.print(table)
    
    # Skipped entities
    if skipped_entities:
        console.print(f"\n[dim]Skipped entity types (top 10):[/dim]")
        for label, count in sorted(skipped_entities.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  [dim]{label}: {count:,}[/dim]")
    
    console.print(f"\n[green]✓ Saved to {output_path}[/green]\n")
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create fully annotated golden PII dataset")
    parser.add_argument("--max-samples", type=int, default=25000, help="Max samples to collect")
    parser.add_argument("--min-entities", type=int, default=1, help="Min entities per sample")
    parser.add_argument("--output", type=str, default="golden_dataset.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    curate_golden_dataset(
        max_samples=args.max_samples,
        min_entities_per_sample=args.min_entities,
        output_path=args.output,
    )
