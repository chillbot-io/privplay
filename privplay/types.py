"""Core types for PHI/PII detection."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class EntityType(str, Enum):
    """All supported PHI/PII entity types."""
    
    # Identity
    NAME_PERSON = "NAME_PERSON"
    NAME_PATIENT = "NAME_PATIENT"
    NAME_PROVIDER = "NAME_PROVIDER"
    NAME_RELATIVE = "NAME_RELATIVE"
    SSN = "SSN"
    MRN = "MRN"
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    DEVICE_ID = "DEVICE_ID"
    
    # Contact
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    FAX = "FAX"
    ADDRESS = "ADDRESS"
    ZIP = "ZIP"
    
    # Temporal
    DATE = "DATE"
    DATE_DOB = "DATE_DOB"
    DATE_ADMISSION = "DATE_ADMISSION"
    DATE_DISCHARGE = "DATE_DISCHARGE"
    AGE = "AGE"
    
    # Financial
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    ROUTING_NUMBER = "ROUTING_NUMBER"
    
    # Digital
    IP_ADDRESS = "IP_ADDRESS"
    MAC_ADDRESS = "MAC_ADDRESS"
    URL = "URL"
    USERNAME = "USERNAME"
    
    # Biometric
    BIOMETRIC = "BIOMETRIC"
    
    # HIPAA Identifiers (additional)
    VIN = "VIN"                           # Vehicle Identification Number (#12)
    UDI = "UDI"                           # Unique Device Identifier (#13)
    CRYPTO_ADDRESS = "CRYPTO_ADDRESS"     # Bitcoin, Ethereum, Litecoin addresses
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"     # Health plan beneficiary number (#9)
    DEA_NUMBER = "DEA_NUMBER"             # DEA registration number (#11)
    MEDICAL_LICENSE = "MEDICAL_LICENSE"   # State medical license (#11)
    NPI = "NPI"                           # National Provider Identifier
    
    # Clinical (dictionary-based)
    DRUG = "DRUG"                         # Medication names
    DIAGNOSIS = "DIAGNOSIS"               # Diagnosis/condition names (ICD-10)
    LAB_TEST = "LAB_TEST"                 # Laboratory test names (LOINC)
    HEALTH_PLAN = "HEALTH_PLAN"           # Insurance company/payer names
    FACILITY = "FACILITY"                 # Healthcare facility names
    
    # Legacy alias - maps to FACILITY internally
    HOSPITAL = "HOSPITAL"                 # Deprecated: use FACILITY
    
    # Special
    OTHER = "OTHER"


class DecisionType(str, Enum):
    """Human review decisions."""
    CONFIRMED = "confirmed"      # Detection was correct
    REJECTED = "rejected"        # Not PHI/PII (false positive)
    CHANGED = "changed"          # PHI but wrong type


class SourceType(str, Enum):
    """Where the detection came from."""
    MODEL = "model"           # Transformer model (Stanford de-id)
    PRESIDIO = "presidio"     # Microsoft Presidio PII detection
    RULE = "rule"             # Regex patterns
    DICTIONARY = "dictionary"
    MERGED = "merged"         # Multiple sources agreed


class VerificationResult(str, Enum):
    """LLM verification decisions."""
    YES = "yes"          # Is PHI
    NO = "no"            # Not PHI
    UNCERTAIN = "uncertain"


@dataclass
class Entity:
    """A detected PHI/PII entity."""
    text: str
    start: int
    end: int
    entity_type: EntityType
    confidence: float
    source: SourceType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    llm_confidence: Optional[float] = None
    llm_reasoning: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.entity_type, str):
            # Handle HOSPITAL -> FACILITY mapping
            if self.entity_type == "HOSPITAL":
                self.entity_type = EntityType.FACILITY
            else:
                self.entity_type = EntityType(self.entity_type)
        if isinstance(self.source, str):
            self.source = SourceType(self.source)


@dataclass
class Document:
    """A document to scan."""
    id: str
    content: str
    source: str  # 'faker', 'import', 'file:path'
    scanned_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Correction:
    """A human review decision."""
    id: str
    entity_id: str
    document_id: str
    entity_text: str
    entity_start: int
    entity_end: int
    detected_type: EntityType
    decision: DecisionType
    correct_type: Optional[EntityType] = None  # If changed
    context_before: str = ""
    context_after: str = ""
    ner_confidence: float = 0.0
    llm_confidence: Optional[float] = None
    reviewed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerificationResponse:
    """LLM verification response."""
    decision: VerificationResult
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class TestResult:
    """F1 test results."""
    precision: float
    recall: float
    f1: float
    total_entities: int
    true_positives: int
    false_positives: int
    false_negatives: int
    by_type: dict = field(default_factory=dict)


@dataclass 
class ReviewStats:
    """Training progress statistics."""
    total_documents: int
    total_entities: int
    reviewed: int
    pending: int
    auto_approved: int
    confirmed: int
    rejected: int
    changed: int
