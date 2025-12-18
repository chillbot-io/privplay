#!/usr/bin/env python3
"""
Curate AI4Privacy dataset - filter noise, keep clean samples.

Garbage patterns to filter:
- 4-digit "passwords" with no context
- 5-digit "addresses" that are just numbers
- 3-digit "credit cards" (CVV without card)
- Common words labeled as names
- Truncated/partial entities

Quality signals to keep:
- Emails with @ symbol
- Phones with 10+ digits
- SSNs with XXX-XX-XXXX pattern
- Credit cards with 13-19 digits
- Names that are actual names (capitalized, not common words)
- Addresses with street indicators
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# Common English words that shouldn't be names
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which',
    'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
    'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
    'here', 'there', 'then', 'once', 'new', 'old', 'good', 'bad', 'first',
    'last', 'long', 'great', 'little', 'own', 'right', 'big', 'high', 'small',
    'large', 'next', 'early', 'young', 'important', 'public', 'national',
    'mr', 'mrs', 'ms', 'dr', 'miss', 'sir', 'north', 'south', 'east', 'west',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'today', 'tomorrow',
    'yesterday', 'forward', 'backward', 'internal', 'external', 'senior', 'junior',
    'customer', 'client', 'patient', 'user', 'account', 'payment', 'service',
    'team', 'sales', 'support', 'trans', 'card', 'bank', 'credit', 'debit',
}

# Street indicators
STREET_INDICATORS = {
    'street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd',
    'drive', 'dr', 'lane', 'ln', 'way', 'court', 'ct', 'place', 'pl',
    'circle', 'cir', 'highway', 'hwy', 'parkway', 'pkwy', 'suite', 'apt',
}


@dataclass
class EntityQuality:
    """Quality assessment for an entity."""
    entity_type: str
    text: str
    is_clean: bool
    reason: str
    confidence: float  # 0-1, how confident we are this is clean


def assess_entity_quality(entity_type: str, text: str, full_text: str = "") -> EntityQuality:
    """
    Assess whether an entity label is clean or garbage.
    
    Returns EntityQuality with is_clean=True for good samples.
    """
    text_clean = text.strip()
    text_lower = text_clean.lower()
    digits_only = ''.join(c for c in text_clean if c.isdigit())
    
    # === EMAIL ===
    if entity_type in ('EMAIL', 'email'):
        if '@' in text_clean and '.' in text_clean.split('@')[-1]:
            return EntityQuality(entity_type, text, True, "Valid email format", 0.95)
        return EntityQuality(entity_type, text, False, "Missing @ or domain", 0.0)
    
    # === PHONE ===
    if entity_type in ('PHONE', 'PHONE_NUMBER', 'phone', 'telephone'):
        if len(digits_only) >= 10:
            return EntityQuality(entity_type, text, True, "10+ digit phone", 0.90)
        if len(digits_only) >= 7 and any(c in text for c in '-().+ '):
            return EntityQuality(entity_type, text, True, "7+ digit formatted phone", 0.80)
        return EntityQuality(entity_type, text, False, f"Too few digits: {len(digits_only)}", 0.0)
    
    # === SSN ===
    if entity_type in ('SSN', 'US_SSN', 'SOCIAL_SECURITY'):
        if re.match(r'^\d{3}-\d{2}-\d{4}$', text_clean):
            return EntityQuality(entity_type, text, True, "SSN format XXX-XX-XXXX", 0.95)
        if len(digits_only) == 9:
            return EntityQuality(entity_type, text, True, "9-digit SSN", 0.85)
        return EntityQuality(entity_type, text, False, f"Invalid SSN format", 0.0)
    
    # === CREDIT CARD ===
    if entity_type in ('CREDIT_CARD', 'CREDITCARDNUMBER', 'credit_card'):
        if 13 <= len(digits_only) <= 19:
            return EntityQuality(entity_type, text, True, "Valid CC length", 0.90)
        if len(digits_only) == 3 or len(digits_only) == 4:
            return EntityQuality(entity_type, text, False, "CVV only, not full card", 0.0)
        return EntityQuality(entity_type, text, False, f"Invalid CC length: {len(digits_only)}", 0.0)
    
    # === PASSWORD ===
    if entity_type in ('PASSWORD', 'password', 'PIN', 'pin'):
        # Passwords need context OR complexity
        if len(text_clean) >= 8:
            has_upper = any(c.isupper() for c in text_clean)
            has_lower = any(c.islower() for c in text_clean)
            has_digit = any(c.isdigit() for c in text_clean)
            has_special = any(not c.isalnum() for c in text_clean)
            
            complexity = sum([has_upper, has_lower, has_digit, has_special])
            if complexity >= 3:
                return EntityQuality(entity_type, text, True, "Complex password", 0.85)
            if complexity >= 2 and len(text_clean) >= 10:
                return EntityQuality(entity_type, text, True, "Long mixed password", 0.75)
        
        # Check for context in surrounding text
        context_keywords = ['password', 'pwd', 'pin', 'passcode', 'secret']
        if any(kw in full_text.lower() for kw in context_keywords):
            if len(text_clean) >= 4:
                return EntityQuality(entity_type, text, True, "Password with context", 0.70)
        
        return EntityQuality(entity_type, text, False, "No complexity or context", 0.0)
    
    # === ADDRESS ===
    if entity_type in ('ADDRESS', 'STREET_ADDRESS', 'address'):
        # Must have street indicator or number + words
        has_street_word = any(ind in text_lower.split() for ind in STREET_INDICATORS)
        has_number = bool(re.search(r'\d+', text_clean))
        has_words = len(text_clean.split()) >= 2
        
        if has_street_word and (has_number or has_words):
            return EntityQuality(entity_type, text, True, "Street address", 0.85)
        if has_number and has_words and len(text_clean) >= 10:
            return EntityQuality(entity_type, text, True, "Address-like", 0.70)
        if text_clean.isdigit() and len(text_clean) <= 5:
            return EntityQuality(entity_type, text, False, "Just a number", 0.0)
        return EntityQuality(entity_type, text, False, "No street indicators", 0.0)
    
    # === NAME ===
    if entity_type in ('NAME', 'PERSON', 'NAME_PERSON', 'FIRSTNAME', 'LASTNAME', 'name', 'person'):
        # Filter common words
        if text_lower in COMMON_WORDS:
            return EntityQuality(entity_type, text, False, f"Common word: {text_lower}", 0.0)
        
        # Must be capitalized (unless all caps)
        if text_clean[0].isupper() or text_clean.isupper():
            # Should have letters
            if any(c.isalpha() for c in text_clean):
                # Not just an initial
                if len(text_clean) >= 2:
                    return EntityQuality(entity_type, text, True, "Capitalized name", 0.80)
        
        return EntityQuality(entity_type, text, False, "Not name-like", 0.0)
    
    # === USERNAME ===
    if entity_type in ('USERNAME', 'username', 'USER_ID'):
        # Usernames often have patterns: Name.Name, Name_Name, @handle
        if re.match(r'^[A-Za-z][A-Za-z0-9._-]{2,30}$', text_clean):
            return EntityQuality(entity_type, text, True, "Valid username pattern", 0.85)
        if text_clean.startswith('@'):
            return EntityQuality(entity_type, text, True, "Handle format", 0.90)
        return EntityQuality(entity_type, text, False, "Invalid username", 0.0)
    
    # === DATE ===
    if entity_type in ('DATE', 'DATE_TIME', 'DOB', 'date'):
        # Must have date-like pattern
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}',  # Month DD
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Month
        ]
        if any(re.search(p, text_clean, re.I) for p in date_patterns):
            return EntityQuality(entity_type, text, True, "Date pattern", 0.85)
        
        # Just two digits is probably not a date
        if len(text_clean) <= 2 and text_clean.isdigit():
            return EntityQuality(entity_type, text, False, "Too short for date", 0.0)
        
        return EntityQuality(entity_type, text, False, "No date pattern", 0.0)
    
    # === IP ADDRESS ===
    if entity_type in ('IP_ADDRESS', 'IP', 'IPV4', 'IPV6'):
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', text_clean):
            return EntityQuality(entity_type, text, True, "IPv4 format", 0.95)
        if ':' in text_clean and len(text_clean) >= 6:
            return EntityQuality(entity_type, text, True, "IPv6-like", 0.80)
        return EntityQuality(entity_type, text, False, "Not IP format", 0.0)
    
    # === URL ===
    if entity_type in ('URL', 'url'):
        if text_clean.startswith(('http://', 'https://', 'www.')):
            return EntityQuality(entity_type, text, True, "URL format", 0.95)
        if '.' in text_clean and '/' in text_clean:
            return EntityQuality(entity_type, text, True, "URL-like", 0.75)
        return EntityQuality(entity_type, text, False, "Not URL format", 0.0)
    
    # === ACCOUNT NUMBER / IBAN ===
    if entity_type in ('ACCOUNT_NUMBER', 'IBAN', 'BANK_ACCOUNT'):
        if len(text_clean) >= 8:
            return EntityQuality(entity_type, text, True, "Account number length", 0.75)
        return EntityQuality(entity_type, text, False, "Too short", 0.0)
    
    # === Default: require minimum length ===
    if len(text_clean) >= 3:
        return EntityQuality(entity_type, text, True, "Default accept", 0.50)
    return EntityQuality(entity_type, text, False, "Too short", 0.0)


def curate_ai4privacy(
    max_samples: int = 50000,
    min_confidence: float = 0.70,
    output_path: str = "ai4privacy_curated.json",
):
    """
    Load AI4Privacy and filter to clean samples.
    """
    console.print(f"\n[bold cyan]═══ AI4Privacy Curation ═══[/bold cyan]\n")
    
    # Load dataset
    try:
        from datasets import load_dataset
        console.print("Loading AI4Privacy from Hugging Face...")
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
        console.print(f"Loaded {len(ds)} total samples")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    # Process samples
    clean_samples = []
    rejected_counts = defaultdict(int)
    type_counts = defaultdict(lambda: {"clean": 0, "rejected": 0})
    
    console.print(f"\nProcessing up to {max_samples} samples...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Curating...", total=min(max_samples, len(ds)))
        
        for idx, row in enumerate(ds):
            if idx >= max_samples:
                break
            
            text = row.get("source_text", "")
            privacy_mask = row.get("privacy_mask", [])
            
            if not text or not privacy_mask:
                progress.advance(task)
                continue
            
            # Assess each entity in this sample
            clean_entities = []
            sample_quality = 0.0
            
            for entity in privacy_mask:
                entity_type = entity.get("label", "")
                entity_text = entity.get("value", "")
                
                quality = assess_entity_quality(entity_type, entity_text, text)
                
                if quality.is_clean and quality.confidence >= min_confidence:
                    clean_entities.append({
                        "label": entity_type,
                        "value": entity_text,
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                        "quality_confidence": quality.confidence,
                    })
                    type_counts[entity_type]["clean"] += 1
                    sample_quality += quality.confidence
                else:
                    rejected_counts[quality.reason] += 1
                    type_counts[entity_type]["rejected"] += 1
            
            # Keep sample if it has clean entities
            if clean_entities:
                avg_quality = sample_quality / len(clean_entities)
                clean_samples.append({
                    "text": text,
                    "entities": clean_entities,
                    "quality_score": avg_quality,
                    "original_index": idx,
                })
            
            progress.advance(task)
    
    # Sort by quality and take top samples
    clean_samples.sort(key=lambda x: x["quality_score"], reverse=True)
    
    # Save curated dataset
    output = {
        "samples": clean_samples,
        "metadata": {
            "total_processed": min(max_samples, len(ds)),
            "clean_samples": len(clean_samples),
            "min_confidence": min_confidence,
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    console.print(f"\n[bold green]═══ Curation Complete ═══[/bold green]\n")
    console.print(f"Processed: {min(max_samples, len(ds))} samples")
    console.print(f"Clean samples: {len(clean_samples)}")
    console.print(f"Clean rate: {len(clean_samples) / min(max_samples, len(ds)) * 100:.1f}%")
    
    # Show type breakdown
    console.print(f"\n[bold]Entity Types:[/bold]")
    table = Table()
    table.add_column("Type")
    table.add_column("Clean", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Rate", justify="right")
    
    for etype in sorted(type_counts.keys()):
        counts = type_counts[etype]
        total = counts["clean"] + counts["rejected"]
        rate = counts["clean"] / total * 100 if total > 0 else 0
        table.add_row(etype, str(counts["clean"]), str(counts["rejected"]), f"{rate:.0f}%")
    
    console.print(table)
    
    # Show rejection reasons
    console.print(f"\n[bold]Top Rejection Reasons:[/bold]")
    for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1])[:10]:
        console.print(f"  {count:6d}  {reason}")
    
    console.print(f"\n[green]✓ Saved to {output_path}[/green]\n")
    
    return clean_samples


def convert_to_ner_format(
    curated_path: str = "ai4privacy_curated.json",
    output_path: str = "ner_training_data.json",
    max_samples: int = 20000,
):
    """
    Convert curated data to NER training format (BIO tagging).
    """
    console.print(f"\n[bold cyan]═══ Converting to NER Format ═══[/bold cyan]\n")
    
    with open(curated_path) as f:
        data = json.load(f)
    
    samples = data["samples"][:max_samples]
    console.print(f"Converting {len(samples)} samples...")
    
    ner_samples = []
    
    for sample in samples:
        text = sample["text"]
        entities = sample["entities"]
        
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x.get("start", 0))
        
        # Build token list with BIO tags
        # Simple whitespace tokenization for now
        tokens = text.split()
        labels = ["O"] * len(tokens)
        
        # Map character positions to token indices
        char_to_token = {}
        char_pos = 0
        for i, token in enumerate(tokens):
            for j in range(len(token)):
                char_to_token[char_pos + j] = i
            char_pos += len(token) + 1  # +1 for space
        
        # Apply entity labels
        for entity in entities:
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label", "").upper()
            
            if start is None or end is None:
                continue
            
            # Find tokens that overlap with entity
            entity_tokens = set()
            for char_pos in range(start, min(end, len(text))):
                if char_pos in char_to_token:
                    entity_tokens.add(char_to_token[char_pos])
            
            # Apply BIO tags
            entity_tokens = sorted(entity_tokens)
            for i, token_idx in enumerate(entity_tokens):
                if token_idx < len(labels):
                    if i == 0:
                        labels[token_idx] = f"B-{label}"
                    else:
                        labels[token_idx] = f"I-{label}"
        
        ner_samples.append({
            "tokens": tokens,
            "labels": labels,
            "text": text,
        })
    
    # Save
    with open(output_path, "w") as f:
        json.dump(ner_samples, f, indent=2)
    
    # Collect unique labels
    all_labels = set()
    for sample in ner_samples:
        all_labels.update(sample["labels"])
    
    console.print(f"\n✓ Converted {len(ner_samples)} samples")
    console.print(f"✓ {len(all_labels)} unique labels: {sorted(all_labels)[:10]}...")
    console.print(f"✓ Saved to {output_path}")
    
    return ner_samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curate AI4Privacy dataset")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to process")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Min quality confidence")
    parser.add_argument("--output", type=str, default="ai4privacy_curated.json", help="Output file")
    parser.add_argument("--convert-ner", action="store_true", help="Also convert to NER format")
    
    args = parser.parse_args()
    
    samples = curate_ai4privacy(
        max_samples=args.max_samples,
        min_confidence=args.min_confidence,
        output_path=args.output,
    )
    
    if args.convert_ner and samples:
        convert_to_ner_format(
            curated_path=args.output,
            output_path="ner_training_data.json",
        )
