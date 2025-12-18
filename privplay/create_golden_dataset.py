#!/usr/bin/env python3
"""
Create golden PII dataset from AI4Privacy.

Smart curation that:
1. Samples by type - ensures balanced representation
2. Type-specific validation - different rules per type
3. Context validation - keeps entities with clear context
4. Quality scoring - ranks samples by cleanliness

Target: ~10,000 high-quality samples
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

# Type aliases - map AI4Privacy labels to our standard types
TYPE_ALIASES = {
    # Names
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON", 
    "MIDDLENAME": "NAME_PERSON",
    "NAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "SUFFIX": "NAME_PERSON",
    
    # Contact
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONENUMBER": "PHONE",
    
    # IDs
    "SSN": "SSN",
    "SOCIALNUM": "SSN",
    "SOCIAL_SECURITY": "SSN",
    
    # Financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN": "IBAN",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "ACCOUNTNAME": "ACCOUNT_NUMBER",
    "BIC": "ACCOUNT_NUMBER",
    
    # Location
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
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
    "DATE": "DATE",
    "DOB": "DATE",
    "TIME": "DATE",
    
    # Other
    "VEHICLEVIN": "VIN",
    "VEHICLEVRM": "VIN",
    "IMEI": "DEVICE_ID",
}

# Target counts per type
TARGET_COUNTS = {
    # Tier 1 - High priority (1200 each)
    "NAME_PERSON": 1200,
    "EMAIL": 1200,
    "PHONE": 1200,
    "SSN": 1200,
    "CREDIT_CARD": 1200,
    "ADDRESS": 1200,
    "DATE": 1200,
    
    # Tier 2 - Medium priority (250 each)
    "USERNAME": 250,
    "PASSWORD": 250,
    "IP_ADDRESS": 250,
    "ACCOUNT_NUMBER": 250,
    "URL": 250,
    "ZIP": 250,
    "LOCATION": 250,
    "IBAN": 250,
}

# Common words that should NOT be names
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mr', 'mrs', 'ms',
    'dr', 'miss', 'sir', 'madam', 'master', 'mister', 'madame', 'sen', 'rep',
    'north', 'south', 'east', 'west', 'new', 'old', 'customer', 'client',
    'patient', 'user', 'account', 'student', 'child', 'contact', 'person',
    'emergency', 'primary', 'secondary', 'home', 'work', 'mobile', 'cell',
}


# =============================================================================
# VALIDATORS - Type-specific quality checks
# =============================================================================

def validate_name(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate person name."""
    text = text.strip()
    
    # Must have letters
    if not any(c.isalpha() for c in text):
        return False, 0.0
    
    # Reject common words
    if text.lower() in COMMON_WORDS:
        return False, 0.0
    
    # Reject single characters
    if len(text) <= 1:
        return False, 0.0
    
    # Should be capitalized (or all caps)
    if not (text[0].isupper() or text.isupper()):
        return False, 0.0
    
    # Bonus for multiple words (full name)
    words = text.split()
    if len(words) >= 2:
        # Check each word is name-like
        for word in words:
            if word.lower() in COMMON_WORDS:
                return False, 0.0
        return True, 0.95
    
    # Single name is OK but lower confidence
    return True, 0.75


def validate_email(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate email address."""
    text = text.strip()
    
    if '@' not in text:
        return False, 0.0
    
    parts = text.split('@')
    if len(parts) != 2:
        return False, 0.0
    
    local, domain = parts
    
    # Must have domain with dot
    if '.' not in domain:
        return False, 0.0
    
    # Local part must have content
    if len(local) < 1:
        return False, 0.0
    
    # Domain must have content after dot
    if domain.endswith('.') or domain.startswith('.'):
        return False, 0.0
    
    return True, 0.95


def validate_phone(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate phone number."""
    digits = ''.join(c for c in text if c.isdigit())
    
    # Must have enough digits
    if len(digits) < 7:
        return False, 0.0
    
    # 10+ digits is strong phone signal
    if len(digits) >= 10:
        return True, 0.90
    
    # 7-9 digits with formatting characters suggests phone
    if len(digits) >= 7 and any(c in text for c in '-().+ '):
        return True, 0.80
    
    return False, 0.0


def validate_ssn(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate SSN."""
    digits = ''.join(c for c in text if c.isdigit())
    
    # Must be 9 digits
    if len(digits) != 9:
        return False, 0.0
    
    area = int(digits[:3])
    group = int(digits[3:5])
    serial = int(digits[5:])
    
    # Invalid area numbers
    if area == 0 or area == 666 or area >= 900:
        return False, 0.0
    
    # Invalid group/serial
    if group == 0 or serial == 0:
        return False, 0.0
    
    # Formatted is higher confidence
    if '-' in text or ' ' in text:
        return True, 0.95
    
    # Unformatted but in SSN context
    context_lower = context.lower()
    if 'ssn' in context_lower or 'social' in context_lower:
        return True, 0.90
    
    return True, 0.75


def validate_credit_card(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate credit card with Luhn."""
    digits = ''.join(c for c in text if c.isdigit())
    
    # Must be 13-19 digits
    if len(digits) < 13 or len(digits) > 19:
        return False, 0.0
    
    # Luhn check
    def luhn(num):
        digits_list = [int(d) for d in num]
        odd = digits_list[-1::-2]
        even = digits_list[-2::-2]
        total = sum(odd) + sum(sum(divmod(d * 2, 10)) for d in even)
        return total % 10 == 0
    
    if not luhn(digits):
        return False, 0.0
    
    return True, 0.95


def validate_address(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate street address."""
    text_lower = text.lower()
    
    # Street indicators
    street_words = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 
                   'blvd', 'drive', 'dr', 'lane', 'ln', 'way', 'court', 'ct',
                   'place', 'pl', 'circle', 'cir', 'highway', 'hwy', 'apt',
                   'suite', 'floor', 'unit', 'box', 'apartment', 'building'}
    
    has_street_word = any(word in text_lower.split() for word in street_words)
    has_number = bool(re.search(r'\d+', text))
    
    # Strong: has number AND street word
    if has_number and has_street_word:
        return True, 0.90
    
    # Medium: has street word
    if has_street_word and len(text) >= 10:
        return True, 0.75
    
    # Weak: just a number (could be building number)
    if has_number and len(text.split()) >= 2:
        return True, 0.60
    
    return False, 0.0


def validate_date(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate date."""
    # Date patterns
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}',
        r'\d{1,2}[\s,]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        r'\d{1,2}(?:st|nd|rd|th)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.I):
            return True, 0.85
    
    # ISO format
    if re.match(r'\d{4}-\d{2}-\d{2}', text):
        return True, 0.90
    
    return False, 0.0


def validate_username(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate username."""
    text = text.strip()
    
    # Username patterns
    if re.match(r'^[A-Za-z][A-Za-z0-9._-]{2,30}$', text):
        # Bonus for common patterns
        if '_' in text or '.' in text:
            return True, 0.90
        return True, 0.75
    
    # Handle format
    if text.startswith('@'):
        return True, 0.90
    
    return False, 0.0


def validate_password(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate password."""
    text = text.strip()
    
    if len(text) < 4:
        return False, 0.0
    
    # Check complexity
    has_upper = any(c.isupper() for c in text)
    has_lower = any(c.islower() for c in text)
    has_digit = any(c.isdigit() for c in text)
    has_special = any(not c.isalnum() for c in text)
    
    complexity = sum([has_upper, has_lower, has_digit, has_special])
    
    # High complexity = likely password
    if complexity >= 3:
        return True, 0.90
    
    # Medium complexity with good length
    if complexity >= 2 and len(text) >= 8:
        return True, 0.80
    
    # Context check
    context_lower = context.lower()
    if 'password' in context_lower or 'pwd' in context_lower or 'pin' in context_lower:
        if len(text) >= 4:
            return True, 0.75
    
    return False, 0.0


def validate_ip(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate IP address."""
    # IPv4
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', text):
        octets = text.split('.')
        if all(0 <= int(o) <= 255 for o in octets):
            return True, 0.95
    
    # IPv6 (simplified)
    if ':' in text and len(text) >= 6:
        return True, 0.80
    
    return False, 0.0


def validate_url(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate URL."""
    if text.startswith(('http://', 'https://')):
        return True, 0.95
    if text.startswith('www.'):
        return True, 0.90
    if re.match(r'^[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', text):
        return True, 0.75
    return False, 0.0


def validate_zip(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate ZIP/postal code."""
    text = text.strip()
    
    # US ZIP
    if re.match(r'^\d{5}(-\d{4})?$', text):
        return True, 0.90
    
    # UK postcode
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$', text, re.I):
        return True, 0.85
    
    return False, 0.0


def validate_account(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate account number."""
    digits = ''.join(c for c in text if c.isdigit())
    
    if len(digits) >= 8:
        return True, 0.75
    
    return False, 0.0


def validate_iban(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate IBAN."""
    cleaned = text.replace(' ', '').upper()
    
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False, 0.0
    
    # Must start with 2 letters
    if not cleaned[:2].isalpha():
        return False, 0.0
    
    # Mod-97 check
    rearranged = cleaned[4:] + cleaned[:4]
    numeric = ''
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - 55)
    
    if int(numeric) % 97 == 1:
        return True, 0.95
    
    return False, 0.0


def validate_location(text: str, context: str = "") -> Tuple[bool, float]:
    """Validate location (city, state, country)."""
    text = text.strip()
    
    # Should be capitalized words
    if text[0].isupper() and len(text) >= 2:
        # Reject common non-location words
        if text.lower() not in COMMON_WORDS:
            return True, 0.70
    
    return False, 0.0


# Validator map
VALIDATORS = {
    "NAME_PERSON": validate_name,
    "EMAIL": validate_email,
    "PHONE": validate_phone,
    "SSN": validate_ssn,
    "CREDIT_CARD": validate_credit_card,
    "ADDRESS": validate_address,
    "DATE": validate_date,
    "USERNAME": validate_username,
    "PASSWORD": validate_password,
    "IP_ADDRESS": validate_ip,
    "URL": validate_url,
    "ZIP": validate_zip,
    "ACCOUNT_NUMBER": validate_account,
    "IBAN": validate_iban,
    "LOCATION": validate_location,
}


# =============================================================================
# MAIN CURATION
# =============================================================================

def normalize_type(label: str) -> str:
    """Normalize entity type to standard form."""
    label_upper = label.upper().replace(" ", "").replace("_", "")
    return TYPE_ALIASES.get(label_upper, TYPE_ALIASES.get(label.upper(), label.upper()))


def curate_golden_dataset(
    max_total: int = 10000,
    output_path: str = "golden_dataset.json",
):
    """Create golden dataset with balanced, validated samples."""
    
    console.print(f"\n[bold cyan]═══ Creating Golden Dataset ═══[/bold cyan]\n")
    
    # Load AI4Privacy
    try:
        from datasets import load_dataset
        console.print("Loading AI4Privacy from Hugging Face...")
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
        console.print(f"Loaded {len(ds)} samples")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    # Storage by type
    samples_by_type: Dict[str, List[Dict]] = defaultdict(list)
    
    # Calculate adjusted targets (scale to max_total)
    total_target = sum(TARGET_COUNTS.values())
    scale = max_total / total_target
    adjusted_targets = {k: int(v * scale) for k, v in TARGET_COUNTS.items()}
    
    console.print(f"\nTarget distribution (scaled to {max_total}):")
    for etype, count in adjusted_targets.items():
        console.print(f"  {etype:20} {count}")
    
    # Process samples
    console.print(f"\nProcessing samples...")
    
    processed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Curating...", total=len(ds))
        
        for row in ds:
            processed += 1
            
            text = row.get("source_text", "")
            entities = row.get("privacy_mask", [])
            
            if not text or not entities:
                progress.advance(task)
                continue
            
            # Process each entity
            for entity in entities:
                raw_label = entity.get("label", "")
                value = entity.get("value", "")
                start = entity.get("start")
                end = entity.get("end")
                
                if not value or start is None:
                    continue
                
                # Normalize type
                norm_type = normalize_type(raw_label)
                
                # Skip if not in our target types
                if norm_type not in adjusted_targets:
                    continue
                
                # Skip if we have enough
                if len(samples_by_type[norm_type]) >= adjusted_targets[norm_type]:
                    continue
                
                # Validate
                validator = VALIDATORS.get(norm_type)
                if validator:
                    is_valid, confidence = validator(value, text)
                    if not is_valid:
                        continue
                else:
                    confidence = 0.70
                
                # Store sample
                samples_by_type[norm_type].append({
                    "text": text,
                    "entity": {
                        "label": norm_type,
                        "value": value,
                        "start": start,
                        "end": end,
                        "original_label": raw_label,
                        "quality_confidence": confidence,
                    },
                    "source_index": processed,
                })
            
            # Check if we're done
            all_done = all(
                len(samples_by_type[t]) >= adjusted_targets[t]
                for t in adjusted_targets
            )
            if all_done:
                console.print(f"\n[green]All targets met at sample {processed}[/green]")
                break
            
            progress.advance(task)
    
    # Build final dataset - one sample per entity to avoid duplicates
    console.print(f"\nBuilding final dataset...")
    
    final_samples = []
    for etype, samples in samples_by_type.items():
        # Shuffle and take up to target
        random.shuffle(samples)
        for sample in samples[:adjusted_targets.get(etype, 0)]:
            final_samples.append({
                "text": sample["text"],
                "entities": [sample["entity"]],
                "quality_score": sample["entity"]["quality_confidence"],
            })
    
    # Shuffle final dataset
    random.shuffle(final_samples)
    
    # Save
    output = {
        "samples": final_samples,
        "metadata": {
            "total_samples": len(final_samples),
            "source": "ai4privacy/pii-masking-400k",
            "target_counts": adjusted_targets,
            "actual_counts": {t: len(s) for t, s in samples_by_type.items()},
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    console.print(f"\n[bold green]═══ Golden Dataset Complete ═══[/bold green]\n")
    
    table = Table(title="Entity Distribution")
    table.add_column("Type")
    table.add_column("Target", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Status", justify="center")
    
    total_actual = 0
    for etype in sorted(adjusted_targets.keys()):
        target = adjusted_targets[etype]
        actual = len(samples_by_type[etype])
        total_actual += actual
        
        if actual >= target:
            status = "[green]✓[/green]"
        elif actual >= target * 0.8:
            status = "[yellow]~[/yellow]"
        else:
            status = "[red]✗[/red]"
        
        table.add_row(etype, str(target), str(actual), status)
    
    table.add_row("─" * 15, "─" * 6, "─" * 6, "─" * 6)
    table.add_row("[bold]TOTAL[/bold]", str(sum(adjusted_targets.values())), str(total_actual), "")
    
    console.print(table)
    console.print(f"\n[green]✓ Saved to {output_path}[/green]\n")
    
    return final_samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create golden PII dataset")
    parser.add_argument("--max-total", type=int, default=10000, help="Total samples")
    parser.add_argument("--output", type=str, default="golden_dataset.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    curate_golden_dataset(
        max_total=args.max_total,
        output_path=args.output,
    )
