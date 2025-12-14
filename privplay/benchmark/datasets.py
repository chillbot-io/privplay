"""Benchmark dataset loaders."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedEntity:
    """A ground-truth entity annotation."""
    text: str
    start: int
    end: int
    entity_type: str  # Original label from dataset
    normalized_type: str  # Mapped to our EntityType


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    text: str
    entities: List[AnnotatedEntity]
    source: str  # 'ai4privacy', 'synthetic_phi', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A benchmark dataset."""
    name: str
    description: str
    samples: List[BenchmarkSample]
    entity_types: List[str]
    source_url: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.samples)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_entities = sum(len(s.entities) for s in self.samples)
        entity_counts = {}
        for sample in self.samples:
            for entity in sample.entities:
                entity_counts[entity.normalized_type] = entity_counts.get(entity.normalized_type, 0) + 1
        
        return {
            "name": self.name,
            "num_samples": len(self.samples),
            "total_entities": total_entities,
            "entity_distribution": entity_counts,
            "avg_entities_per_sample": total_entities / len(self.samples) if self.samples else 0,
        }


# Mapping from AI4Privacy labels to our EntityType values
AI4PRIVACY_MAPPING = {
    # Names
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON", 
    "MIDDLENAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "SUFFIX": "NAME_PERSON",
    "USERNAME": "USERNAME",
    
    # Contact
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TEL": "PHONE",
    
    # Location
    "STREET": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "ZIPCODE": "ZIP",
    "ZIP": "ZIP",
    "COUNTY": "ADDRESS",
    "COUNTRY": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
    
    # Government IDs
    "SSN": "SSN",
    "SOCIALSECURITY": "SSN",
    "DRIVERLICENSE": "DRIVER_LICENSE",
    "PASSPORT": "PASSPORT",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "IBAN": "BANK_ACCOUNT",
    "BIC": "BANK_ACCOUNT",
    "SWIFT": "BANK_ACCOUNT",
    
    # Financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD",
    "CREDITCARDISSUER": "OTHER",
    "CURRENCY": "OTHER",
    "CURRENCYCODE": "OTHER",
    "CURRENCYSYMBOL": "OTHER",
    "CURRENCYNAME": "OTHER",
    "BITCOINADDRESS": "OTHER",
    "ETHEREUMADDRESS": "OTHER",
    "LITECOINADDRESS": "OTHER",
    "PIN": "OTHER",
    
    # Medical / Health
    "MEDICALRECORD": "MRN",
    "HEALTHPLAN": "HEALTH_PLAN",
    "BLOOD_TYPE": "OTHER",
    "HEIGHT": "OTHER",
    "WEIGHT": "OTHER",
    
    # Digital
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "MAC": "MAC_ADDRESS",
    "MACADDRESS": "MAC_ADDRESS",
    "URL": "URL",
    "USERAGENT": "OTHER",
    
    # Temporal
    "DATE": "DATE",
    "DOB": "DATE_DOB",
    "DATEOFBIRTH": "DATE_DOB",
    "TIME": "OTHER",
    "AGE": "AGE",
    
    # Organization
    "COMPANY": "OTHER",
    "COMPANYNAME": "OTHER",
    "ORGANIZATION": "OTHER",
    
    # Other identifiers
    "VEHICLEVIN": "DEVICE_ID",
    "VEHICLEVRM": "DEVICE_ID",
    "LICENSEPLATE": "OTHER",
    "IMEI": "DEVICE_ID",
    "IMEISV": "DEVICE_ID",
    "SERIALNUMBER": "DEVICE_ID",
    "NEARBYGPSCOORDINATE": "OTHER",
    "ORDINALDIRECTION": "OTHER",
    "GENDER": "OTHER",
    "SEX": "OTHER",
    "JOBAREA": "OTHER",
    "JOBTITLE": "OTHER",
    "JOBTYPE": "OTHER",
}


def _normalize_label(label: str) -> str:
    """Normalize a label to our EntityType format."""
    # Remove B-, I- prefixes (BIO tagging)
    if label.startswith(("B-", "I-")):
        label = label[2:]
    
    # Uppercase and remove spaces/underscores variations
    label = label.upper().replace(" ", "").replace("-", "").replace("_", "")
    
    # Check mapping
    for key, value in AI4PRIVACY_MAPPING.items():
        if label == key.upper().replace("_", ""):
            return value
    
    # Default to OTHER
    return "OTHER"


def load_ai4privacy_dataset(
    max_samples: int = 1000,
    cache_dir: Optional[Path] = None,
) -> BenchmarkDataset:
    """
    Load AI4Privacy PII dataset from Hugging Face.
    
    This dataset contains 200k+ samples with 54 PII classes.
    License: CC-BY-4.0
    
    Args:
        max_samples: Maximum number of samples to load
        cache_dir: Directory to cache downloaded data
        
    Returns:
        BenchmarkDataset with loaded samples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )
    
    logger.info(f"Loading AI4Privacy dataset (max {max_samples} samples)...")
    
    # Load from Hugging Face
    dataset = load_dataset(
        "ai4privacy/pii-masking-200k",
        split=f"train[:{max_samples}]",
        trust_remote_code=True,
    )
    
    samples = []
    entity_types_seen = set()
    
    for idx, item in enumerate(dataset):
        # Parse the masked text and privacy mask
        text = item.get("source_text", item.get("text", ""))
        
        # Get token labels if available (for token classification format)
        if "tokens" in item and "ner_tags" in item:
            entities = _parse_bio_tags(
                item["tokens"], 
                item["ner_tags"],
                item.get("label_names", [])
            )
        # Or parse from span annotations
        elif "privacy_mask" in item:
            entities = _parse_privacy_mask(text, item["privacy_mask"])
        elif "spans" in item:
            entities = _parse_spans(text, item["spans"])
        else:
            # Try to find entities from masked version
            entities = []
        
        # Track entity types
        for e in entities:
            entity_types_seen.add(e.normalized_type)
        
        sample_id = hashlib.md5(text.encode()).hexdigest()[:12]
        samples.append(BenchmarkSample(
            id=f"ai4privacy_{sample_id}",
            text=text,
            entities=entities,
            source="ai4privacy",
            metadata={"index": idx},
        ))
    
    logger.info(f"Loaded {len(samples)} samples with {len(entity_types_seen)} entity types")
    
    return BenchmarkDataset(
        name="AI4Privacy PII-Masking-200k",
        description="Large-scale PII masking dataset with 54 PII classes",
        samples=samples,
        entity_types=sorted(entity_types_seen),
        source_url="https://huggingface.co/datasets/ai4privacy/pii-masking-200k",
    )


def _parse_bio_tags(
    tokens: List[str],
    tags: List[int],
    label_names: List[str],
) -> List[AnnotatedEntity]:
    """Parse BIO-tagged tokens into entities."""
    entities = []
    current_entity = None
    position = 0
    
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # Get label name
        if tag < len(label_names):
            label = label_names[tag]
        else:
            label = f"LABEL_{tag}"
        
        # Handle BIO tags
        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = label[2:]
            current_entity = AnnotatedEntity(
                text=token,
                start=position,
                end=position + len(token),
                entity_type=entity_type,
                normalized_type=_normalize_label(entity_type),
            )
        elif label.startswith("I-") and current_entity:
            # Continue current entity
            current_entity.text += " " + token
            current_entity.end = position + len(token)
        else:
            # O tag or mismatch - save current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
        position += len(token) + 1  # +1 for space
    
    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


def _parse_privacy_mask(text: str, privacy_mask: List[Dict]) -> List[AnnotatedEntity]:
    """Parse privacy mask annotations."""
    entities = []
    
    for mask in privacy_mask:
        entity_type = mask.get("label", mask.get("type", "OTHER"))
        start = mask.get("start", 0)
        end = mask.get("end", 0)
        
        entities.append(AnnotatedEntity(
            text=text[start:end],
            start=start,
            end=end,
            entity_type=entity_type,
            normalized_type=_normalize_label(entity_type),
        ))
    
    return entities


def _parse_spans(text: str, spans: List[Dict]) -> List[AnnotatedEntity]:
    """Parse span annotations."""
    entities = []
    
    for span in spans:
        entity_type = span.get("label", span.get("type", "OTHER"))
        start = span.get("start", span.get("begin", 0))
        end = span.get("end", 0)
        
        entities.append(AnnotatedEntity(
            text=text[start:end],
            start=start,
            end=end,
            entity_type=entity_type,
            normalized_type=_normalize_label(entity_type),
        ))
    
    return entities


def load_synthetic_phi_dataset(
    num_samples: int = 100,
    seed: int = 42,
) -> BenchmarkDataset:
    """
    Generate synthetic clinical documents with ground-truth PHI.
    
    This creates realistic clinical notes using Faker with known
    PHI locations for testing detection accuracy.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        BenchmarkDataset with generated samples
    """
    try:
        from faker import Faker
    except ImportError:
        raise ImportError(
            "faker library required. Install with: pip install faker"
        )
    
    logger.info(f"Generating {num_samples} synthetic PHI samples...")
    
    fake = Faker()
    Faker.seed(seed)
    
    samples = []
    entity_types_seen = set()
    
    generators = [
        _generate_clinical_note,
        _generate_lab_report,
        _generate_radiology_report,
        _generate_prescription,
    ]
    
    for i in range(num_samples):
        # Rotate through generators
        generator = generators[i % len(generators)]
        text, entities = generator(fake, i)
        
        # Track entity types
        for e in entities:
            entity_types_seen.add(e.normalized_type)
        
        samples.append(BenchmarkSample(
            id=f"synthetic_{i:05d}",
            text=text,
            entities=entities,
            source="synthetic_phi",
            metadata={"generator": generator.__name__},
        ))
    
    logger.info(f"Generated {len(samples)} samples with {len(entity_types_seen)} entity types")
    
    return BenchmarkDataset(
        name="Synthetic PHI Dataset",
        description="Synthetically generated clinical notes with ground-truth PHI",
        samples=samples,
        entity_types=sorted(entity_types_seen),
    )


def _generate_clinical_note(fake: 'Faker', idx: int) -> tuple:
    """Generate a synthetic clinical note."""
    entities = []
    parts = []
    position = 0
    
    def add_text(text: str):
        nonlocal position
        parts.append(text)
        position += len(text)
    
    def add_entity(text: str, entity_type: str, normalized: str):
        nonlocal position
        start = position
        parts.append(text)
        position += len(text)
        entities.append(AnnotatedEntity(
            text=text,
            start=start,
            end=position,
            entity_type=entity_type,
            normalized_type=normalized,
        ))
    
    add_text("CLINICAL NOTE\n\nPatient: ")
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PATIENT")
    add_text("\nMRN: ")
    add_entity(fake.numerify("########"), "MRN", "MRN")
    add_text("\nDOB: ")
    add_entity(fake.date_of_birth().strftime("%m/%d/%Y"), "DOB", "DATE_DOB")
    add_text("\nSSN: ")
    add_entity(fake.ssn(), "SSN", "SSN")
    
    add_text("\n\nChief Complaint: ")
    add_text(fake.sentence())
    
    add_text("\n\nHistory: ")
    add_entity(fake.first_name(), "PATIENT_FIRST", "NAME_PATIENT")
    add_text(" reports symptoms for the past week. ")
    add_text("Contact email: ")
    add_entity(fake.email(), "EMAIL", "EMAIL")
    add_text("\nPhone: ")
    add_entity(fake.phone_number(), "PHONE", "PHONE")
    
    add_text("\n\nAddress: ")
    add_entity(fake.address().replace("\n", ", "), "ADDRESS", "ADDRESS")
    
    add_text("\n\nFamily History: Patient's wife ")
    add_entity(fake.first_name_female(), "RELATIVE_NAME", "NAME_RELATIVE")
    add_text(" confirms the symptoms began last week.")
    
    add_text("\n\nObjective:\nVitals stable. ")
    add_text(fake.sentence())
    
    add_text("\n\nAssessment:\n")
    add_text(fake.sentence())
    
    add_text("\n\nPlan:\n1. ")
    add_text(fake.sentence())
    add_text("\n2. Follow up with Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PROVIDER")
    add_text(" at ")
    add_entity(fake.phone_number(), "PHONE", "PHONE")
    
    add_text("\n\nSigned: Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PROVIDER")
    
    return "".join(parts), entities


def _generate_lab_report(fake: 'Faker', idx: int) -> tuple:
    """Generate a synthetic lab report."""
    entities = []
    parts = []
    position = 0
    
    def add_text(text: str):
        nonlocal position
        parts.append(text)
        position += len(text)
    
    def add_entity(text: str, entity_type: str, normalized: str):
        nonlocal position
        start = position
        parts.append(text)
        position += len(text)
        entities.append(AnnotatedEntity(
            text=text,
            start=start,
            end=position,
            entity_type=entity_type,
            normalized_type=normalized,
        ))
    
    add_text("LABORATORY REPORT\n\nPatient: ")
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PATIENT")
    add_text("\nAccount #: ")
    add_entity(fake.numerify("##########"), "ACCOUNT", "ACCOUNT_NUMBER")
    add_text("\nCollection Date: ")
    add_entity(fake.date_this_month().strftime("%m/%d/%Y"), "DATE", "DATE")
    
    add_text("\n\nOrdering Physician: ")
    add_entity(f"Dr. {fake.name()}", "PROVIDER_NAME", "NAME_PROVIDER")
    
    add_text("\n\nTEST RESULTS:\n")
    tests = ["CBC", "BMP", "Lipid Panel", "TSH", "HbA1c"]
    for test in fake.random_elements(tests, length=3, unique=True):
        add_text(f"\n{test}: {fake.pyfloat(min_value=0.1, max_value=100, right_digits=1)}")
    
    add_text("\n\nComments: Results reviewed. Call patient at ")
    add_entity(fake.phone_number(), "PHONE", "PHONE")
    add_text(" to discuss.")
    
    return "".join(parts), entities


def _generate_radiology_report(fake: 'Faker', idx: int) -> tuple:
    """Generate a synthetic radiology report."""
    entities = []
    parts = []
    position = 0
    
    def add_text(text: str):
        nonlocal position
        parts.append(text)
        position += len(text)
    
    def add_entity(text: str, entity_type: str, normalized: str):
        nonlocal position
        start = position
        parts.append(text)
        position += len(text)
        entities.append(AnnotatedEntity(
            text=text,
            start=start,
            end=position,
            entity_type=entity_type,
            normalized_type=normalized,
        ))
    
    modality = fake.random_element(["X-RAY", "CT", "MRI", "ULTRASOUND"])
    body_part = fake.random_element(["CHEST", "ABDOMEN", "HEAD", "SPINE", "KNEE"])
    
    add_text(f"RADIOLOGY REPORT\n\nExam: {modality} {body_part}\n")
    add_text("Date: ")
    add_entity(fake.date_this_month().strftime("%m/%d/%Y"), "DATE", "DATE")
    
    add_text("\n\nPatient: ")
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PATIENT")
    add_text("\nDOB: ")
    add_entity(fake.date_of_birth().strftime("%m/%d/%Y"), "DOB", "DATE_DOB")
    add_text("\nMRN: ")
    add_entity(fake.numerify("########"), "MRN", "MRN")
    
    add_text("\n\nClinical History: ")
    add_text(fake.sentence())
    
    add_text("\n\nFindings:\n")
    add_text(fake.paragraph())
    
    add_text("\n\nImpression:\n1. ")
    add_text(fake.sentence())
    
    add_text("\n\nDictated by: Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PROVIDER")
    add_text("\nTranscribed: ")
    add_entity(fake.date_this_month().strftime("%m/%d/%Y"), "DATE", "DATE")
    
    return "".join(parts), entities


def _generate_prescription(fake: 'Faker', idx: int) -> tuple:
    """Generate a synthetic prescription."""
    entities = []
    parts = []
    position = 0
    
    def add_text(text: str):
        nonlocal position
        parts.append(text)
        position += len(text)
    
    def add_entity(text: str, entity_type: str, normalized: str):
        nonlocal position
        start = position
        parts.append(text)
        position += len(text)
        entities.append(AnnotatedEntity(
            text=text,
            start=start,
            end=position,
            entity_type=entity_type,
            normalized_type=normalized,
        ))
    
    add_text("PRESCRIPTION\n\nDate: ")
    add_entity(fake.date_this_month().strftime("%m/%d/%Y"), "DATE", "DATE")
    
    add_text("\n\nPatient: ")
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PATIENT")
    add_text("\nAddress: ")
    add_entity(fake.address().replace("\n", ", "), "ADDRESS", "ADDRESS")
    add_text("\nPhone: ")
    add_entity(fake.phone_number(), "PHONE", "PHONE")
    
    add_text("\n\nRx:\n")
    meds = ["Lisinopril 10mg", "Metformin 500mg", "Atorvastatin 20mg", "Omeprazole 20mg"]
    add_text(fake.random_element(meds))
    add_text("\nSig: Take one tablet daily")
    add_text(f"\nDispense: #{fake.random_int(30, 90)}")
    add_text(f"\nRefills: {fake.random_int(0, 5)}")
    
    add_text("\n\nPrescriber: Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PROVIDER")
    add_text("\nDEA: ")
    add_entity(fake.bothify("??#######"), "DEA", "ACCOUNT_NUMBER")
    add_text("\nNPI: ")
    add_entity(fake.numerify("##########"), "NPI", "ACCOUNT_NUMBER")
    
    return "".join(parts), entities


def get_dataset(
    name: str,
    max_samples: int = 1000,
    **kwargs
) -> BenchmarkDataset:
    """
    Load a benchmark dataset by name.
    
    Args:
        name: Dataset name ('ai4privacy' or 'synthetic_phi')
        max_samples: Maximum number of samples
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        BenchmarkDataset
    """
    loaders = {
        "ai4privacy": lambda: load_ai4privacy_dataset(max_samples=max_samples, **kwargs),
        "synthetic_phi": lambda: load_synthetic_phi_dataset(num_samples=max_samples, **kwargs),
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    return loaders[name]()


def list_datasets():
    """List available benchmark datasets with descriptions."""
    return [
        {
            "name": "synthetic_phi",
            "description": "Synthetically generated clinical notes with ground-truth PHI annotations",
        },
        {
            "name": "ai4privacy",
            "description": "AI4Privacy PII-Masking-200k dataset from HuggingFace (54 PII types)",
        },
        {
            "name": "physionet",
            "description": "PhysioNet Gold Standard de-identification corpus (requires credentialed access)",
        },
    ]


def display_benchmark_result(result):
    """Display benchmark results in a formatted way."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print()
    console.print(f"[bold]Benchmark Results: {result.dataset_name}[/bold]")
    console.print("─" * 50)
    console.print()
    console.print(f"  Samples:   {result.num_samples}")
    console.print(f"  Precision: {result.precision:.1%}")
    console.print(f"  Recall:    {result.recall:.1%}")
    console.print(f"  F1 Score:  {result.f1:.1%}")
    console.print()
    console.print(f"  True Positives:  {result.true_positives}")
    console.print(f"  False Positives: {result.false_positives}")
    console.print(f"  False Negatives: {result.false_negatives}")
    
    # Per-entity breakdown if available
    if result.by_entity_type:
        console.print()
        console.print("[bold]Per-Entity Breakdown:[/bold]")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Entity Type")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1")
        table.add_column("TP/FP/FN")
        
        for etype, metrics in sorted(result.by_entity_type.items()):
            table.add_row(
                etype,
                f"{metrics.get('precision', 0):.0%}",
                f"{metrics.get('recall', 0):.0%}",
                f"{metrics.get('f1', 0):.0%}",
                f"{metrics.get('tp', 0)}/{metrics.get('fp', 0)}/{metrics.get('fn', 0)}",
            )
        
        console.print(table)
    
    console.print()
