"""Adversarial and edge cases for meta-classifier training.

These are hand-crafted examples designed to confuse detectors.
The meta-classifier needs to learn from these patterns.
"""

from dataclasses import dataclass
from typing import List, Optional
import random
import uuid

from .synthetic_generator import LabeledDocument, LabeledEntity, DocumentBuilder


# =============================================================================
# DRUG NAMES THAT LOOK LIKE PERSON NAMES
# =============================================================================

DRUG_NAME_TRAPS = [
    # These are real drug names that could be detected as person names
    ("Flomax", "drug_as_name"),
    ("Cardizem", "drug_as_name"),
    ("Prozac", "drug_as_name"),
    ("Allegra", "drug_as_name"),
    ("Tamiflu", "drug_as_name"),
    ("Ambien", "drug_as_name"),
    ("Xanax", "drug_as_name"),
    ("Valium", "drug_as_name"),
    ("Lyrica", "drug_as_name"),
    ("Crestor", "drug_as_name"),
    ("Plavix", "drug_as_name"),
    ("Zoloft", "drug_as_name"),
    ("Lexapro", "drug_as_name"),
    ("Celebrex", "drug_as_name"),
    ("Cymbalta", "drug_as_name"),
    ("Januvia", "drug_as_name"),
    ("Humira", "drug_as_name"),
    ("Enbrel", "drug_as_name"),
    ("Remicade", "drug_as_name"),
    ("Botox", "drug_as_name"),
]


def generate_drug_name_trap() -> LabeledDocument:
    """Generate note with drug names that look like person names."""
    builder = DocumentBuilder()
    
    drug, trap_type = random.choice(DRUG_NAME_TRAPS)
    dose = random.choice(["10mg", "20mg", "25mg", "50mg", "100mg"])
    frequency = random.choice(["daily", "BID", "TID", "QHS", "PRN"])
    
    templates = [
        (f"Patient started on {drug} {dose} {frequency} for symptom management.", []),
        (f"Continue {drug} as prescribed.", []),
        (f"Increase {drug} to {dose} {frequency}.", []),
        (f"Hold {drug} due to side effects.", []),
        (f"Administer {drug} {dose} IV now.", []),
    ]
    
    template, entities = random.choice(templates)
    builder.add_line(template)
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # No PHI - drug names are NOT entities
        doc_type="adversarial",
        metadata={"adversarial_type": trap_type, "trap_value": drug},
    )


# =============================================================================
# COMMON WORDS THAT LOOK LIKE NAMES
# =============================================================================

COMMON_WORD_TRAPS = [
    "STABLE",
    "NORMAL",
    "ALERT",
    "ORIENTED",
    "NEGATIVE",
    "POSITIVE",
    "CHRONIC",
    "ACUTE",
    "MILD",
    "SEVERE",
    "BENIGN",
    "MALIGNANT",
    "PRIMARY",
    "SECONDARY",
    "BILATERAL",
    "UNILATERAL",
]


def generate_common_word_trap() -> LabeledDocument:
    """Generate note with common clinical words in all caps."""
    builder = DocumentBuilder()
    
    word = random.choice(COMMON_WORD_TRAPS)
    
    templates = [
        f"Patient is {word} and comfortable.",
        f"Condition: {word}.",
        f"Assessment: {word} findings on exam.",
        f"Results were {word} for infection.",
        f"Patient remains {word} overnight.",
    ]
    
    builder.add_line(random.choice(templates))
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # No PHI
        doc_type="adversarial",
        metadata={"adversarial_type": "common_word", "trap_value": word},
    )


# =============================================================================
# NUMBERS THAT LOOK LIKE SSN BUT AREN'T
# =============================================================================

def generate_fake_ssn_trap() -> LabeledDocument:
    """Generate text with SSN-like patterns that aren't SSNs."""
    builder = DocumentBuilder()
    
    # Generate something that looks like SSN but isn't
    traps = [
        ("ICD-10 code 123-45-6789", "icd_code"),
        ("Reference number: 987-65-4321", "reference"),
        ("Order #123-45-6789", "order_number"),
        ("Part number 234-56-7890", "part_number"),
        ("Case ID: 345-67-8901", "case_id"),
        ("Tracking: 456-78-9012", "tracking"),
        ("Protocol 567-89-0123", "protocol"),
    ]
    
    text, trap_type = random.choice(traps)
    builder.add_line(text)
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # No PHI - these aren't real SSNs
        doc_type="adversarial",
        metadata={"adversarial_type": "fake_ssn", "trap_value": trap_type},
    )


# =============================================================================
# DATES THAT ARE NOT PHI
# =============================================================================

def generate_non_phi_date_trap() -> LabeledDocument:
    """Generate text with dates that aren't PHI (publication dates, etc.)."""
    builder = DocumentBuilder()
    
    year = random.randint(2015, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    date_str = f"{month:02d}/{day:02d}/{year}"
    
    templates = [
        f"Study published {date_str}.",
        f"Guidelines updated {date_str}.",
        f"Protocol version {date_str}.",
        f"FDA approved {date_str}.",
        f"Research conducted in {year}.",
        f"Data collected between {year-1} and {year}.",
        f"As of {date_str}, recommendations include...",
    ]
    
    builder.add_line(random.choice(templates))
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # Publication dates are not PHI
        doc_type="adversarial",
        metadata={"adversarial_type": "non_phi_date"},
    )


# =============================================================================
# FACILITY NAMES (SHOULD DETECT, BUT DIFFERENT TYPE)
# =============================================================================

FACILITIES = [
    "Massachusetts General Hospital",
    "Johns Hopkins Hospital",
    "Mayo Clinic",
    "Cleveland Clinic",
    "Stanford Medical Center",
    "UCLA Medical Center",
    "Mount Sinai Hospital",
    "NYU Langone",
    "Northwestern Memorial",
    "UCSF Medical Center",
    "Duke University Hospital",
    "Cedars-Sinai",
    "Memorial Sloan Kettering",
    "MD Anderson",
    "Brigham and Women's",
]


def generate_facility_case() -> LabeledDocument:
    """Generate text with facility names (should be detected as FACILITY, not NAME)."""
    builder = DocumentBuilder()
    
    facility = random.choice(FACILITIES)
    
    templates = [
        (f"Patient transferred from ", facility, "."),
        (f"Records requested from ", facility, "."),
        (f"Consultation with ", facility, " oncology."),
        (f"Previously seen at ", facility, "."),
        (f"Referred to ", facility, " for surgery."),
    ]
    
    prefix, facility_name, suffix = random.choice(templates)
    
    builder.add_text(prefix)
    builder.add_entity(facility_name, "FACILITY")
    builder.add_line(suffix)
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={"adversarial_type": "facility"},
    )


# =============================================================================
# PROVIDER VS PATIENT NAME DISTINCTION
# =============================================================================

def generate_provider_patient_distinction() -> LabeledDocument:
    """Generate text where provider and patient names must be distinguished."""
    builder = DocumentBuilder()
    
    from faker import Faker
    fake = Faker()
    
    patient_name = fake.name()
    patient_first = patient_name.split()[0]
    provider_last = fake.last_name()
    
    builder.add_text("Patient ")
    builder.add_entity(patient_name, "NAME_PATIENT")
    builder.add_text(" was seen by Dr. ")
    builder.add_entity(provider_last, "NAME_PROVIDER")
    builder.add_line(" today.")
    builder.add_line()
    
    builder.add_entity(patient_first, "NAME_PATIENT")
    builder.add_text(" reports improvement. Dr. ")
    builder.add_entity(provider_last, "NAME_PROVIDER")
    builder.add_line(" recommends continued treatment.")
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={"adversarial_type": "provider_patient"},
    )


# =============================================================================
# AMBIGUOUS SINGLE NAMES
# =============================================================================

AMBIGUOUS_NAMES = [
    "Jordan",  # Name or Air Jordan reference?
    "Chase",   # Name or Chase bank?
    "Hunter",  # Name or occupation?
    "Mason",   # Name or occupation?
    "Taylor",  # Name or Taylor Swift reference?
    "Morgan",  # Name or Morgan Stanley?
    "Austin",  # Name or city?
    "Dallas",  # Name or city?
    "Phoenix", # Name or city?
    "Brooklyn",# Name or borough?
    "Christian", # Name or religion?
    "Faith",   # Name or concept?
    "Hope",    # Name or concept?
    "Grace",   # Name or concept?
]


def generate_ambiguous_name_as_patient() -> LabeledDocument:
    """Generate text with ambiguous name that IS a patient."""
    builder = DocumentBuilder()
    
    from faker import Faker
    fake = Faker()
    
    first_name = random.choice(AMBIGUOUS_NAMES)
    last_name = fake.last_name()
    full_name = f"{first_name} {last_name}"
    mrn = str(random.randint(10000000, 99999999))
    
    builder.add_text("Patient: ")
    builder.add_entity(full_name, "NAME_PATIENT")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    builder.add_line()
    
    builder.add_entity(first_name, "NAME_PATIENT")
    builder.add_line(" presents today for follow-up.")
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={"adversarial_type": "ambiguous_name_patient", "trap_value": first_name},
    )


def generate_ambiguous_name_as_city() -> LabeledDocument:
    """Generate text with ambiguous name that is NOT a patient (it's a city)."""
    builder = DocumentBuilder()
    
    # Cities that are also names
    city = random.choice(["Austin", "Dallas", "Phoenix", "Brooklyn", "Madison"])
    
    templates = [
        f"Patient relocated from {city}, TX.",
        f"Records transferred from {city} General Hospital.",
        f"Previously treated in {city}.",
    ]
    
    builder.add_line(random.choice(templates))
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # City names are not PHI in this context
        doc_type="adversarial",
        metadata={"adversarial_type": "ambiguous_name_city", "trap_value": city},
    )


# =============================================================================
# PHONE-LIKE NUMBERS THAT AREN'T PHONES
# =============================================================================

def generate_fake_phone_trap() -> LabeledDocument:
    """Generate text with phone-like patterns that aren't phone numbers."""
    builder = DocumentBuilder()
    
    traps = [
        "Room 555-1234",
        "Extension 555-4321", 
        "Code 555-9876",
        "Pager #555-1111",  # This one IS a phone/pager
        "Unit 555-2222",
        "Lab ID: 555-3333",
        "Ref: 555-4444",
    ]
    
    trap = random.choice(traps[:3] + traps[4:])  # Exclude pager for this test
    builder.add_line(trap)
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # These aren't real phone numbers
        doc_type="adversarial",
        metadata={"adversarial_type": "fake_phone"},
    )


# =============================================================================
# EMAIL-LIKE PATTERNS THAT AREN'T PERSONAL
# =============================================================================

def generate_fake_email_trap() -> LabeledDocument:
    """Generate text with generic/system email addresses."""
    builder = DocumentBuilder()
    
    # These are generic, not personal PHI
    generic_emails = [
        "noreply@hospital.org",
        "info@healthcare.com", 
        "support@clinic.net",
        "records@medical.org",
        "billing@hospital.com",
        "appointments@clinic.org",
    ]
    
    email = random.choice(generic_emails)
    builder.add_line(f"For questions, contact {email}")
    
    text, _ = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=[],  # Generic emails are not PHI
        doc_type="adversarial",
        metadata={"adversarial_type": "generic_email", "trap_value": email},
    )


def generate_personal_email_case() -> LabeledDocument:
    """Generate text with a personal email that IS PHI."""
    builder = DocumentBuilder()
    
    from faker import Faker
    fake = Faker()
    
    email = fake.email()
    
    builder.add_text("Patient's email: ")
    builder.add_entity(email, "EMAIL")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={"adversarial_type": "personal_email"},
    )


# =============================================================================
# MIXED CONTEXT (PII + MEDICAL = PHI MODE)
# =============================================================================

def generate_pii_with_medical_context() -> LabeledDocument:
    """Generate document with PII in medical context (triggers PHI mode)."""
    builder = DocumentBuilder()
    
    from faker import Faker
    fake = Faker()
    
    name = fake.name()
    ssn = f"{random.randint(100,899):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"
    phone = fake.phone_number()
    diagnosis = random.choice(["Type 2 Diabetes", "Hypertension", "COPD", "CHF"])
    
    builder.add_line("PATIENT REGISTRATION")
    builder.add_line()
    
    builder.add_text("Name: ")
    builder.add_entity(name, "NAME_PATIENT")
    builder.add_line()
    
    builder.add_text("SSN: ")
    builder.add_entity(ssn, "SSN")
    builder.add_line()
    
    builder.add_text("Phone: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    builder.add_line()
    builder.add_line(f"Primary Diagnosis: {diagnosis}")
    builder.add_line("Follow-up required in 2 weeks.")
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={
            "adversarial_type": "pii_medical_context",
            "has_ssn": True,
            "has_medical": True,
        },
    )


def generate_pii_without_medical_context() -> LabeledDocument:
    """Generate document with PII but no medical context."""
    builder = DocumentBuilder()
    
    from faker import Faker
    fake = Faker()
    
    name = fake.name()
    email = fake.email()
    phone = fake.phone_number()
    address = fake.street_address()
    
    builder.add_line("CONTACT INFORMATION UPDATE")
    builder.add_line()
    
    builder.add_text("Name: ")
    builder.add_entity(name, "NAME_PERSON")  # Note: NAME_PERSON not NAME_PATIENT
    builder.add_line()
    
    builder.add_text("Email: ")
    builder.add_entity(email, "EMAIL")
    builder.add_line()
    
    builder.add_text("Phone: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    builder.add_text("Address: ")
    builder.add_entity(address, "ADDRESS")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="adversarial",
        metadata={
            "adversarial_type": "pii_no_medical",
            "has_ssn": False,
            "has_medical": False,
        },
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================

ADVERSARIAL_GENERATORS = [
    (generate_drug_name_trap, 0.15),
    (generate_common_word_trap, 0.10),
    (generate_fake_ssn_trap, 0.10),
    (generate_non_phi_date_trap, 0.10),
    (generate_facility_case, 0.10),
    (generate_provider_patient_distinction, 0.10),
    (generate_ambiguous_name_as_patient, 0.08),
    (generate_ambiguous_name_as_city, 0.05),
    (generate_fake_phone_trap, 0.05),
    (generate_fake_email_trap, 0.05),
    (generate_personal_email_case, 0.04),
    (generate_pii_with_medical_context, 0.04),
    (generate_pii_without_medical_context, 0.04),
]


def generate_adversarial_case() -> LabeledDocument:
    """Generate a random adversarial case."""
    generators, weights = zip(*ADVERSARIAL_GENERATORS)
    generator = random.choices(generators, weights=weights, k=1)[0]
    return generator()


def generate_adversarial_dataset(n: int = 500) -> List[LabeledDocument]:
    """Generate n adversarial cases."""
    return [generate_adversarial_case() for _ in range(n)]


def get_adversarial_stats(docs: List[LabeledDocument]) -> dict:
    """Get statistics on adversarial case distribution."""
    from collections import Counter
    
    types = Counter()
    for doc in docs:
        adv_type = doc.metadata.get("adversarial_type", "unknown")
        types[adv_type] += 1
    
    return dict(types)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print(f"Generating {n} adversarial cases...")
    docs = generate_adversarial_dataset(n)
    
    print("\nDistribution:")
    stats = get_adversarial_stats(docs)
    for adv_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {adv_type}: {count}")
    
    print("\nSample cases:")
    for doc in docs[:3]:
        print(f"\n--- {doc.metadata.get('adversarial_type')} ---")
        print(doc.text.strip())
        if doc.entities:
            print(f"Entities: {[(e.text, e.entity_type) for e in doc.entities]}")
        else:
            print("Entities: (none - this is a trap)")
