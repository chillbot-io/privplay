"""MTSamples dataset loader with PHI re-injection.

MTSamples contains ~5000 de-identified clinical notes with placeholder PHI like 
"Dr. Sample Doctor". We re-inject realistic synthetic PHI and track exact spans 
to create ground truth annotations for training.

This gives us REAL clinical language patterns (from actual medical transcriptionists)
with KNOWN PHI locations for training the meta-classifier.

Usage:
    from privplay.training.mtsamples_loader import load_mtsamples_dataset
    
    docs = load_mtsamples_dataset(max_samples=1000)
    # Returns List[LabeledDocument] compatible with train_meta_classifier
"""

import re
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Pattern, Set
from pathlib import Path
import uuid

from faker import Faker

# Import from your existing code
from .synthetic_generator import LabeledDocument, LabeledEntity

logger = logging.getLogger(__name__)
fake = Faker()


# =============================================================================
# PLACEHOLDER PATTERNS - What MTSamples uses for de-identification
# =============================================================================

# Compiled regex patterns for each PHI type placeholder
PHI_PLACEHOLDER_PATTERNS: Dict[str, List[Pattern]] = {}

def _compile_patterns():
    """Compile all placeholder patterns."""
    global PHI_PLACEHOLDER_PATTERNS
    
    patterns = {
        # Provider names
        "NAME_PROVIDER": [
            r"\bDr\.?\s+Sample\b",
            r"\bSample,?\s+M\.?D\.?\b",
            r"\bDoctor\s+Sample\b",
            r"\bDr\.?\s+[A-Z][a-z]+\s+Sample\b",
            r"\bSample\s+Doctor\b",
            r"\bAttending:\s*Sample\b",
            r"\bPhysician:\s*Sample\b",
        ],
        
        # Patient names
        "NAME_PATIENT": [
            r"\bSample\s+Patient\b",
            r"\bPatient\s+Sample\b",
            r"\bPatient:\s*Sample\b",
            r"\bMr\.?\s+Sample\b",
            r"\bMrs\.?\s+Sample\b",
            r"\bMs\.?\s+Sample\b",
            r"\bThe\s+patient,?\s+Sample\b",
            r"\bSample,\s+[A-Z][a-z]+\b",  # "Sample, John"
            r"\b[A-Z][a-z]+\s+Sample\b(?!\s+(?:Hospital|Medical|Clinic|Healthcare))",
        ],
        
        # Facility names
        "FACILITY": [
            r"\bSample\s+Hospital\b",
            r"\bSample\s+Medical\s+Center\b",
            r"\bSample\s+Clinic\b",
            r"\bSample\s+Healthcare\b",
            r"\bSample\s+Medical\s+Group\b",
            r"\bSample\s+Regional\s+Medical\b",
        ],
        
        # Placeholder dates (clearly fake)
        "DATE": [
            r"\b01/01/2000\b",
            r"\b1/1/2000\b",
            r"\b00/00/0000\b",
            r"\bXX/XX/XXXX\b",
            r"\bMM/DD/YYYY\b",
            r"\bDD/MM/YYYY\b",
            r"\b01-01-2000\b",
        ],
        
        # Placeholder MRNs
        "MRN": [
            r"\b0{6,10}\b",
            r"\b1{6,10}\b",
            r"(?:MRN|MR#|Medical\s+Record)[\s:#]*0+\b",
            r"(?:MRN|MR#)[\s:#]*1234567\d*\b",
        ],
        
        # Placeholder SSNs (all same digit or sequential)
        "SSN": [
            r"\b000-00-0000\b",
            r"\b111-11-1111\b",
            r"\b123-45-6789\b",
            r"\b999-99-9999\b",
        ],
        
        # Placeholder phones
        "PHONE": [
            r"\b\(000\)\s*000-0000\b",
            r"\b000-000-0000\b",
            r"\b\(555\)\s*555-5555\b",
            r"\b555-555-5555\b",
            r"\bxxx-xxx-xxxx\b",
        ],
        
        # Placeholder addresses
        "ADDRESS": [
            r"\b123\s+Sample\s+Street\b",
            r"\b123\s+Main\s+Street\b",
            r"\bSample\s+Address\b",
            r"\b1234\s+Sample\s+(?:Road|Ave|Avenue|Blvd|Drive|Dr|Lane|Ln)\b",
        ],
    }
    
    PHI_PLACEHOLDER_PATTERNS = {
        entity_type: [re.compile(p, re.IGNORECASE) for p in pattern_list]
        for entity_type, pattern_list in patterns.items()
    }

_compile_patterns()


# =============================================================================
# PHI GENERATORS - Create realistic replacements
# =============================================================================

def generate_provider_name() -> str:
    """Generate a realistic provider name."""
    templates = [
        f"Dr. {fake.last_name()}",
        f"Dr. {fake.first_name()} {fake.last_name()}",
        f"{fake.last_name()}, M.D.",
        f"{fake.first_name()} {fake.last_name()}, MD",
    ]
    return random.choice(templates)


def generate_patient_name() -> str:
    """Generate a realistic patient name."""
    return fake.name()


def generate_facility_name() -> str:
    """Generate a realistic facility name."""
    templates = [
        f"{fake.city()} General Hospital",
        f"{fake.city()} Medical Center",
        f"St. {fake.first_name()}'s Hospital",
        f"{fake.last_name()} Memorial Hospital",
        f"{fake.city()} Regional Medical Center",
        f"University of {fake.city()} Hospital",
    ]
    return random.choice(templates)


def generate_date() -> str:
    """Generate a realistic date."""
    date = fake.date_between(start_date='-2y', end_date='today')
    formats = ["%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"]
    return date.strftime(random.choice(formats))


def generate_mrn() -> str:
    """Generate a realistic MRN."""
    return str(random.randint(10000000, 99999999))


def generate_ssn() -> str:
    """Generate a realistic-looking SSN."""
    area = random.randint(1, 899)
    if area == 666:
        area = 667
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def generate_phone() -> str:
    """Generate a realistic phone number."""
    return fake.phone_number()


def generate_address() -> str:
    """Generate a realistic street address."""
    return fake.street_address()


PHI_GENERATORS = {
    "NAME_PROVIDER": generate_provider_name,
    "NAME_PATIENT": generate_patient_name,
    "FACILITY": generate_facility_name,
    "DATE": generate_date,
    "MRN": generate_mrn,
    "SSN": generate_ssn,
    "PHONE": generate_phone,
    "ADDRESS": generate_address,
}


# =============================================================================
# RE-INJECTION ENGINE
# =============================================================================

@dataclass
class PlaceholderMatch:
    """A detected placeholder in the text."""
    start: int
    end: int
    text: str
    entity_type: str
    pattern: str  # Which pattern matched (for debugging)


def detect_placeholders(text: str) -> List[PlaceholderMatch]:
    """Detect all placeholder PHI patterns in text."""
    matches = []
    seen_spans: Set[Tuple[int, int]] = set()
    
    for entity_type, patterns in PHI_PLACEHOLDER_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                
                # Skip if we already have a match at this location
                if span in seen_spans:
                    continue
                
                # Skip if this overlaps with an existing match
                overlaps = any(
                    not (span[1] <= existing[0] or span[0] >= existing[1])
                    for existing in seen_spans
                )
                if overlaps:
                    continue
                
                seen_spans.add(span)
                matches.append(PlaceholderMatch(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    entity_type=entity_type,
                    pattern=pattern.pattern,
                ))
    
    # Sort by position
    matches.sort(key=lambda m: m.start)
    return matches


def _reinject_with_tracking(
    text: str, 
    placeholders: List[PlaceholderMatch]
) -> Tuple[str, List[LabeledEntity]]:
    """Re-inject PHI while tracking exact positions."""
    
    if not placeholders:
        return text, []
    
    # Build new text piece by piece
    parts = []
    entities = []
    last_end = 0
    
    for placeholder in placeholders:
        # Add text before this placeholder
        parts.append(text[last_end:placeholder.start])
        
        # Generate replacement
        generator = PHI_GENERATORS.get(placeholder.entity_type)
        if generator is None:
            # Keep original if no generator
            parts.append(placeholder.text)
            last_end = placeholder.end
            continue
        
        replacement = generator()
        
        # Calculate position in new text
        new_start = sum(len(p) for p in parts)
        new_end = new_start + len(replacement)
        
        # Add replacement
        parts.append(replacement)
        
        # Track entity
        entities.append(LabeledEntity(
            start=new_start,
            end=new_end,
            text=replacement,
            entity_type=placeholder.entity_type,
        ))
        
        last_end = placeholder.end
    
    # Add remaining text after last placeholder
    parts.append(text[last_end:])
    
    new_text = "".join(parts)
    
    return new_text, entities


# =============================================================================
# VALIDATION
# =============================================================================

def validate_reinjection(doc: LabeledDocument) -> Tuple[bool, List[str]]:
    """
    Validate that all entity spans are correct.
    
    This is CRITICAL - bad training data will poison the model.
    """
    errors = []
    
    for entity in doc.entities:
        # Check bounds
        if entity.start < 0 or entity.end > len(doc.text):
            errors.append(
                f"OUT OF BOUNDS: [{entity.start}:{entity.end}] "
                f"for text length {len(doc.text)}"
            )
            continue
        
        if entity.start >= entity.end:
            errors.append(
                f"INVALID SPAN: start >= end [{entity.start}:{entity.end}]"
            )
            continue
        
        # Check text matches
        extracted = doc.text[entity.start:entity.end]
        if extracted != entity.text:
            errors.append(
                f"SPAN MISMATCH: expected '{entity.text}' "
                f"but got '{extracted}' at [{entity.start}:{entity.end}]"
            )
    
    return len(errors) == 0, errors


def validate_no_overlaps(doc: LabeledDocument) -> Tuple[bool, List[str]]:
    """Ensure no two entities overlap."""
    errors = []
    
    sorted_entities = sorted(doc.entities, key=lambda e: e.start)
    
    for i in range(len(sorted_entities) - 1):
        curr = sorted_entities[i]
        next_e = sorted_entities[i + 1]
        
        if curr.end > next_e.start:
            errors.append(
                f"OVERLAP: '{curr.text}' [{curr.start}:{curr.end}] "
                f"overlaps with '{next_e.text}' [{next_e.start}:{next_e.end}]"
            )
    
    return len(errors) == 0, errors


def validate_document(doc: LabeledDocument) -> Tuple[bool, List[str]]:
    """Run all validations on a document."""
    all_errors = []
    
    valid1, errors1 = validate_reinjection(doc)
    all_errors.extend(errors1)
    
    valid2, errors2 = validate_no_overlaps(doc)
    all_errors.extend(errors2)
    
    return len(all_errors) == 0, all_errors


def validate_dataset(docs: List[LabeledDocument]) -> Tuple[int, int, List[str]]:
    """
    Validate entire dataset.
    
    Returns: (valid_count, invalid_count, sample_errors)
    """
    valid = 0
    invalid = 0
    sample_errors = []
    
    for doc in docs:
        is_valid, errors = validate_document(doc)
        if is_valid:
            valid += 1
        else:
            invalid += 1
            if len(sample_errors) < 10:  # Keep first 10 errors for review
                sample_errors.extend([f"[{doc.id}] {e}" for e in errors[:3]])
    
    return valid, invalid, sample_errors


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_mtsamples_raw(max_samples: Optional[int] = None) -> List[Dict]:
    """Load raw MTSamples data from local CSV."""
    import pandas as pd
    
    local_paths = [
        Path("/mnt/d/privplay/privplay/training/data/mtsamples.csv"),
        Path.home() / ".privplay" / "data" / "mtsamples.csv",
        Path("./data/mtsamples.csv"),
        Path("./mtsamples.csv"),
    ]
    
    csv_path = None
    for path in local_paths:
        if path.exists():
            csv_path = path
            logger.info(f"Found MTSamples CSV at: {csv_path}")
            break
    
    if csv_path is None:
        searched = "\n".join(f"  - {p}" for p in local_paths)
        raise RuntimeError(
            f"Could not find mtsamples.csv. Searched:\n{searched}"
        )
    
    logger.info(f"Loading MTSamples from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    samples = []
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break
        
        text = str(row.get("transcription", ""))
        if not text or len(text) < 100:
            continue
            
        samples.append({
            "text": text,
            "specialty": str(row.get("medical_specialty", "")),
            "description": str(row.get("description", "")),
            "sample_name": str(row.get("sample_name", "")),
        })
    
    logger.info(f"Loaded {len(samples)} samples from CSV")
    return samples


def process_mtsamples_document(
    raw: Dict,
    doc_id: str,
) -> Optional[LabeledDocument]:
    """
    Process a single MTSamples document.
    
    1. Detect placeholder patterns
    2. Re-inject synthetic PHI
    3. Validate
    4. Return LabeledDocument or None if invalid
    """
    text = raw.get("text", "")
    
    if not text or len(text) < 100:
        return None
    
    # Re-inject PHI
    new_text, entities = _reinject_with_tracking(text, detect_placeholders(text))
    
    # Create document
    doc = LabeledDocument(
        id=doc_id,
        text=new_text,
        entities=entities,
        doc_type="mtsamples",
        metadata={
            "medical_specialty": raw.get("specialty", ""),
            "description": raw.get("description", ""),
            "sample_name": raw.get("sample_name", ""),
            "original_length": len(text),
            "reinjected_entities": len(entities),
        },
    )
    
    # Validate
    is_valid, errors = validate_document(doc)
    if not is_valid:
        logger.warning(f"Invalid document {doc_id}: {errors[:2]}")
        return None
    
    return doc


def load_mtsamples_dataset(
    max_samples: Optional[int] = None,
    min_entities: int = 0,
    validate: bool = True,
) -> List[LabeledDocument]:
    """
    Load MTSamples with PHI re-injection.
    
    Args:
        max_samples: Maximum number of samples to process
        min_entities: Minimum entities required per document (0 = include all)
        validate: Whether to validate each document
        
    Returns:
        List of LabeledDocuments with ground truth PHI annotations
    """
    logger.info("Loading MTSamples dataset...")
    
    # Load raw data (get extra in case some get filtered)
    raw_samples = load_mtsamples_raw(max_samples=max_samples * 2 if max_samples else None)
    
    # Process each document
    documents = []
    skipped_invalid = 0
    skipped_no_phi = 0
    
    for i, raw in enumerate(raw_samples):
        if max_samples and len(documents) >= max_samples:
            break
        
        doc_id = f"mtsamples_{i}"
        doc = process_mtsamples_document(raw, doc_id)
        
        if doc is None:
            skipped_invalid += 1
            continue
        
        if len(doc.entities) < min_entities:
            skipped_no_phi += 1
            continue
        
        documents.append(doc)
    
    logger.info(
        f"Processed {len(documents)} documents "
        f"(skipped: {skipped_invalid} invalid, {skipped_no_phi} no PHI)"
    )
    
    # Final validation
    if validate and documents:
        valid, invalid, errors = validate_dataset(documents)
        if invalid > 0:
            logger.error(f"VALIDATION FAILED: {invalid} invalid documents")
            for err in errors[:5]:
                logger.error(f"  {err}")
            raise ValueError(f"Dataset validation failed: {invalid} invalid documents")
        
        logger.info(f"Validation passed: {valid} documents OK")
    
    return documents


# =============================================================================
# STATISTICS AND DEBUGGING
# =============================================================================

def get_dataset_stats(docs: List[LabeledDocument]) -> Dict:
    """Get statistics about the dataset."""
    from collections import Counter
    
    if not docs:
        return {
            "total_documents": 0,
            "total_entities": 0,
            "avg_entities_per_doc": 0,
            "entity_types": {},
            "specialties": {},
        }
    
    total_entities = sum(len(doc.entities) for doc in docs)
    
    entity_types = Counter()
    specialties = Counter()
    
    for doc in docs:
        for entity in doc.entities:
            entity_types[entity.entity_type] += 1
        specialty = doc.metadata.get("medical_specialty", "unknown")
        specialties[specialty] += 1
    
    return {
        "total_documents": len(docs),
        "total_entities": total_entities,
        "avg_entities_per_doc": total_entities / len(docs) if docs else 0,
        "entity_types": dict(entity_types.most_common()),
        "specialties": dict(specialties.most_common(10)),
    }


def display_sample(doc: LabeledDocument, max_length: int = 500):
    """Display a sample document for debugging."""
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    console.print(f"\n{'='*60}")
    console.print(f"ID: {doc.id}")
    console.print(f"Specialty: {doc.metadata.get('medical_specialty', 'unknown')}")
    console.print(f"Entities: {len(doc.entities)}")
    
    # Show entity breakdown
    for entity in doc.entities:
        console.print(f"  [{entity.start}:{entity.end}] {entity.entity_type}: '{entity.text}'")
    
    # Show text preview with highlighted entities
    text = doc.text[:max_length]
    console.print(f"\nText preview:")
    console.print(Panel(text + ("..." if len(doc.text) > max_length else "")))


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"Loading {n} MTSamples documents...")
    
    try:
        docs = load_mtsamples_dataset(max_samples=n)
        
        print(f"\nLoaded {len(docs)} documents")
        
        stats = get_dataset_stats(docs)
        print(f"\nStatistics:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Avg per doc: {stats['avg_entities_per_doc']:.1f}")
        print(f"\nEntity types:")
        for etype, count in stats['entity_types'].items():
            print(f"  {etype}: {count}")
        
        print(f"\nSpecialties:")
        for specialty, count in list(stats['specialties'].items())[:5]:
            print(f"  {specialty}: {count}")
        
        print("\nSample documents:")
        for doc in docs[:3]:
            display_sample(doc)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
