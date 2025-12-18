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


# =============================================================================
# AI4Privacy Type Mapping
# =============================================================================
# Maps AI4Privacy labels to our canonical EntityType values.
# Must match what our PII-BERT v2 model outputs.
# None = exclude from evaluation (not PII)

AI4PRIVACY_MAPPING = {
    # Names -> NAME_PERSON
    "FIRSTNAME": "NAME_PERSON",
    "LASTNAME": "NAME_PERSON", 
    "MIDDLENAME": "NAME_PERSON",
    "PREFIX": "NAME_PERSON",
    "SUFFIX": "NAME_PERSON",
    "NAME": "NAME_PERSON",
    "GIVENNAME": "NAME_PERSON",
    "SURNAME": "NAME_PERSON",
    "FULLNAME": "NAME_PERSON",
    
    # Credentials
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",
    "PIN": "PASSWORD",
    
    # Contact
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TEL": "PHONE",
    "TELEPHONE": "PHONE",
    "MOBILE": "PHONE",
    "FAX": "FAX",
    
    # Location -> ADDRESS
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "COUNTY": "ADDRESS",
    "COUNTRY": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
    
    # ZIP codes
    "ZIPCODE": "ZIP",
    "ZIP": "ZIP",
    "POSTALCODE": "ZIP",
    "POSTCODE": "ZIP",
    
    # Government IDs
    "SSN": "SSN",
    "SOCIALSECURITY": "SSN",
    "SOCIALSECURITYNUMBER": "SSN",
    
    "DRIVERLICENSE": "DRIVER_LICENSE",
    "DRIVERSLICENSE": "DRIVER_LICENSE",
    "DRIVERS_LICENSE": "DRIVER_LICENSE",
    
    "PASSPORT": "PASSPORT",
    "PASSPORTNUMBER": "PASSPORT",
    
    # Account numbers
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    "ACCOUNTNUM": "ACCOUNT_NUMBER",
    "ACCOUNT": "ACCOUNT_NUMBER",
    "TAXID": "ACCOUNT_NUMBER",
    "TAX_ID": "ACCOUNT_NUMBER",
    "TAXNUM": "ACCOUNT_NUMBER",
    "IDCARDNUM": "ACCOUNT_NUMBER",
    "NATIONALID": "ACCOUNT_NUMBER",
    
    # Bank accounts
    "IBAN": "BANK_ACCOUNT",
    "BIC": "BANK_ACCOUNT",
    "SWIFT": "BANK_ACCOUNT",
    "BANKACCOUNT": "BANK_ACCOUNT",
    
    # Financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD",
    "CREDITCARDISSUER": None,  # Not PII - card brand name
    "CURRENCY": None,
    "CURRENCYCODE": None,
    "CURRENCYSYMBOL": None,
    "CURRENCYNAME": None,
    "AMOUNT": None,
    
    # Crypto
    "BITCOINADDRESS": "CRYPTO_ADDRESS",
    "ETHEREUMADDRESS": "CRYPTO_ADDRESS",
    "LITECOINADDRESS": "CRYPTO_ADDRESS",
    
    # Device identifiers -> DEVICE_ID
    "PHONEIMEI": "DEVICE_ID",
    "IMEI": "DEVICE_ID",
    "IMEISV": "DEVICE_ID",
    "SERIALNUMBER": "DEVICE_ID",
    "SERIAL": "DEVICE_ID",
    
    # Vehicle -> VIN
    "VEHICLEVIN": "VIN",
    "VIN": "VIN",
    "VEHICLEVRM": "VIN",
    "LICENSEPLATE": "VIN",
    "LICENSE_PLATE": "VIN",
    
    # Medical
    "MEDICALRECORD": "MRN",
    "MRN": "MRN",
    "HEALTHPLAN": "HEALTH_PLAN_ID",
    "BLOOD_TYPE": None,
    "HEIGHT": None,
    "WEIGHT": None,
    
    # Digital
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    
    "MAC": "MAC_ADDRESS",
    "MACADDRESS": "MAC_ADDRESS",
    "MAC_ADDRESS": "MAC_ADDRESS",
    
    "URL": "URL",
    "WEBSITE": "URL",
    "URI": "URL",
    
    "USERAGENT": "USER_AGENT",
    "USER_AGENT": "USER_AGENT",
    
    # Temporal
    "DATE": "DATE",
    "TIME": "DATE",
    "DATETIME": "DATE",
    
    "DOB": "DATE",
    "DATEOFBIRTH": "DATE",
    "BIRTHDATE": "DATE",
    
    "AGE": "AGE",
    
    # GPS
    "NEARBYGPSCOORDINATE": "GPS_COORDINATE",
    "GPS": "GPS_COORDINATE",
    "COORDINATE": "GPS_COORDINATE",
    "COORDINATES": "GPS_COORDINATE",
    
    # Organization - not PII per HIPAA Safe Harbor
    "COMPANY": None,
    "COMPANYNAME": None,
    "ORGANIZATION": None,
    "EMPLOYER": None,
    
    # Job info - not PII
    "ORDINALDIRECTION": None,
    "GENDER": None,
    "SEX": None,
    "JOBAREA": None,
    "JOBTITLE": None,
    "JOBTYPE": None,
    "JOBDESCRIPTOR": None,
}


def _normalize_label(label: str) -> Optional[str]:
    """Normalize a label to our EntityType format. Returns None if excluded."""
    # Remove B-, I- prefixes (BIO tagging)
    if label.startswith(("B-", "I-")):
        label = label[2:]
    
    # Try exact match first
    if label.upper() in AI4PRIVACY_MAPPING:
        return AI4PRIVACY_MAPPING[label.upper()]
    
    # Try without underscores/spaces
    normalized = label.upper().replace(" ", "").replace("-", "").replace("_", "")
    
    for key, value in AI4PRIVACY_MAPPING.items():
        if normalized == key.upper().replace("_", ""):
            return value
    
    # Unknown type - exclude
    logger.debug(f"Unknown label type: {label}")
    return None


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
        
        # Filter out None types (excluded non-PII) and track entity types
        filtered_entities = []
        for e in entities:
            if e.normalized_type is not None:
                filtered_entities.append(e)
                entity_types_seen.add(e.normalized_type)
        
        sample_id = hashlib.md5(text.encode()).hexdigest()[:12]
        samples.append(BenchmarkSample(
            id=sample_id,
            text=text,
            entities=filtered_entities,
            source="ai4privacy",
            metadata={"index": idx},
        ))
    
    logger.info(f"Loaded {len(samples)} samples with {len(entity_types_seen)} entity types")
    
    return BenchmarkDataset(
        name="ai4privacy",
        description="AI4Privacy PII-Masking-200k dataset",
        samples=samples,
        entity_types=sorted(entity_types_seen),
        source_url="https://huggingface.co/datasets/ai4privacy/pii-masking-200k",
    )


def _parse_privacy_mask(text: str, privacy_mask: List[Dict]) -> List[AnnotatedEntity]:
    """Parse privacy_mask format from AI4Privacy."""
    entities = []
    
    for mask in privacy_mask:
        start = mask.get("start")
        end = mask.get("end")
        label = mask.get("label", "")
        value = mask.get("value", "")
        
        if start is None or end is None:
            continue
        
        # Get text from span
        entity_text = text[start:end] if start < len(text) and end <= len(text) else value
        
        normalized = _normalize_label(label)
        
        entities.append(AnnotatedEntity(
            text=entity_text,
            start=start,
            end=end,
            entity_type=label,
            normalized_type=normalized,
        ))
    
    return entities


def _parse_spans(text: str, spans: List[Dict]) -> List[AnnotatedEntity]:
    """Parse spans format."""
    entities = []
    
    for span in spans:
        start = span.get("start", span.get("begin"))
        end = span.get("end")
        label = span.get("label", span.get("type", ""))
        
        if start is None or end is None:
            continue
        
        entity_text = text[start:end]
        normalized = _normalize_label(label)
        
        entities.append(AnnotatedEntity(
            text=entity_text,
            start=start,
            end=end,
            entity_type=label,
            normalized_type=normalized,
        ))
    
    return entities


def _parse_bio_tags(tokens: List[str], tags: List[int], label_names: List[str]) -> List[AnnotatedEntity]:
    """Parse BIO-tagged token sequences."""
    entities = []
    current_entity = None
    current_start = 0
    position = 0
    
    for i, (token, tag_id) in enumerate(zip(tokens, tags)):
        if tag_id >= len(label_names):
            tag = "O"
        else:
            tag = label_names[tag_id]
        
        if tag.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            
            label = tag[2:]
            normalized = _normalize_label(label)
            current_entity = AnnotatedEntity(
                text=token,
                start=position,
                end=position + len(token),
                entity_type=label,
                normalized_type=normalized,
            )
        elif tag.startswith("I-") and current_entity:
            # Continue entity
            current_entity.text += " " + token
            current_entity.end = position + len(token)
        else:
            # O tag or I- without B-
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
        position += len(token) + 1  # +1 for space
    
    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


# =============================================================================
# Synthetic PHI Dataset Generator
# =============================================================================

def load_synthetic_phi_dataset(
    num_samples: int = 100,
    seed: int = 42,
) -> BenchmarkDataset:
    """
    Generate synthetic PHI dataset for testing.
    
    Creates realistic clinical documents with known PHI.
    """
    try:
        from faker import Faker
    except ImportError:
        raise ImportError("faker library required. Install with: pip install faker")
    
    import random
    random.seed(seed)
    
    fake = Faker()
    Faker.seed(seed)
    
    samples = []
    entity_types = set()
    
    # Document generators
    generators = [
        _generate_clinical_note,
        _generate_lab_report,
        _generate_radiology_report,
        _generate_prescription,
    ]
    
    for i in range(num_samples):
        generator = random.choice(generators)
        text, entities = generator(fake, i)
        
        for e in entities:
            entity_types.add(e.normalized_type)
        
        samples.append(BenchmarkSample(
            id=f"synth_{i:05d}",
            text=text,
            entities=entities,
            source="synthetic_phi",
            metadata={"generator": generator.__name__},
        ))
    
    return BenchmarkDataset(
        name="synthetic_phi",
        description="Synthetically generated clinical notes with ground-truth PHI",
        samples=samples,
        entity_types=sorted(entity_types),
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
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PERSON")
    add_text("\nMRN: ")
    add_entity(fake.numerify("########"), "MRN", "MRN")
    add_text("\nDOB: ")
    add_entity(fake.date_of_birth().strftime("%m/%d/%Y"), "DOB", "DATE")
    add_text("\nSSN: ")
    add_entity(fake.ssn(), "SSN", "SSN")
    
    add_text("\n\nChief Complaint: ")
    add_text(fake.sentence())
    
    add_text("\n\nHistory of Present Illness:\n")
    add_text(fake.paragraph())
    
    add_text("\n\nPlan:\n1. ")
    add_text(fake.sentence())
    add_text("\n2. Follow up with Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PERSON")
    add_text(" at ")
    add_entity(fake.phone_number(), "PHONE", "PHONE")
    
    add_text("\n\nSigned: Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PERSON")
    
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
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PERSON")
    add_text("\nAccount #: ")
    add_entity(fake.numerify("##########"), "ACCOUNT", "ACCOUNT_NUMBER")
    add_text("\nCollection Date: ")
    add_entity(fake.date_this_month().strftime("%m/%d/%Y"), "DATE", "DATE")
    
    add_text("\n\nOrdering Physician: ")
    add_entity(f"Dr. {fake.name()}", "PROVIDER_NAME", "NAME_PERSON")
    
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
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PERSON")
    add_text("\nDOB: ")
    add_entity(fake.date_of_birth().strftime("%m/%d/%Y"), "DOB", "DATE")
    add_text("\nMRN: ")
    add_entity(fake.numerify("########"), "MRN", "MRN")
    
    add_text("\n\nClinical History: ")
    add_text(fake.sentence())
    
    add_text("\n\nFindings:\n")
    add_text(fake.paragraph())
    
    add_text("\n\nImpression:\n1. ")
    add_text(fake.sentence())
    
    add_text("\n\nDictated by: Dr. ")
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PERSON")
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
    add_entity(fake.name(), "PATIENT_NAME", "NAME_PERSON")
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
    add_entity(fake.name(), "PROVIDER_NAME", "NAME_PERSON")
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
