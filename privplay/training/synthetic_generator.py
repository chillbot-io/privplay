"""Synthetic clinical document generator with labeled entity spans.

Generates training data for the meta-classifier by creating clinical notes
with known PHI/PII locations tracked by character offset.

UPDATED: Now includes pronouns and references (He/She, the patient, Mr./Ms. X)
as labeled entities for coreference training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid
import random
import re
from faker import Faker

fake = Faker()


@dataclass
class LabeledEntity:
    """An entity with known location and type."""
    start: int
    end: int
    text: str
    entity_type: str
    # NEW: Track coreference info
    coref_cluster_id: Optional[int] = None  # Which cluster this belongs to
    is_anchor: bool = False  # Is this the primary mention?
    
    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "entity_type": self.entity_type,
            "coref_cluster_id": self.coref_cluster_id,
            "is_anchor": self.is_anchor,
        }


@dataclass
class LabeledDocument:
    """A document with labeled entities."""
    id: str
    text: str
    entities: List[LabeledEntity]
    doc_type: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }


# =============================================================================
# ENTITY GENERATORS
# =============================================================================

def generate_patient_name() -> Tuple[str, str, str]:
    """Returns (full_name, first_name, last_name)"""
    first = fake.first_name()
    last = fake.last_name()
    return f"{first} {last}", first, last


def generate_provider_name() -> Tuple[str, str]:
    """Returns (full_name_with_title, last_name)"""
    last = fake.last_name()
    return f"Dr. {last}", last


def generate_dob() -> str:
    return fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y")


def generate_mrn() -> str:
    return str(random.randint(10000000, 99999999))


def generate_ssn() -> str:
    # Generate valid-looking SSN (not real validation, just format)
    area = random.randint(1, 899)
    if area == 666:
        area = 667
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def generate_phone() -> str:
    return fake.phone_number()


def generate_email() -> str:
    return fake.email()


def generate_address() -> str:
    return fake.street_address()


def generate_city_state() -> Tuple[str, str]:
    return fake.city(), fake.state_abbr()


def generate_date() -> str:
    return fake.date_this_year().strftime("%m/%d/%Y")


def generate_time() -> str:
    return fake.time(pattern="%H:%M")


def generate_account_number() -> str:
    return str(random.randint(1000000000, 9999999999))


def generate_ip_address() -> str:
    return fake.ipv4()


# =============================================================================
# CLINICAL CONTENT (NOT PHI)
# =============================================================================

COMPLAINTS = [
    "chest pain", "shortness of breath", "abdominal pain", "headache",
    "fever", "cough", "nausea and vomiting", "back pain", "dizziness",
    "weakness", "fatigue", "leg swelling", "palpitations", "syncope",
    "altered mental status", "fall", "urinary symptoms", "rash",
]

PMH_CONDITIONS = [
    "Hypertension", "Diabetes mellitus type 2", "Hyperlipidemia",
    "Coronary artery disease", "COPD", "Asthma", "CHF", "Atrial fibrillation",
    "Chronic kidney disease", "GERD", "Osteoarthritis", "Depression",
    "Hypothyroidism", "Anxiety", "Obesity", "Sleep apnea",
]

MEDICATIONS = [
    "Metformin 500mg BID", "Lisinopril 10mg daily", "Atorvastatin 40mg QHS",
    "Aspirin 81mg daily", "Metoprolol 25mg BID", "Omeprazole 20mg daily",
    "Amlodipine 5mg daily", "Levothyroxine 50mcg daily", "Gabapentin 300mg TID",
    "Losartan 50mg daily", "Furosemide 40mg daily", "Warfarin 5mg daily",
    "Prednisone 10mg daily", "Albuterol inhaler PRN", "Insulin glargine 20 units QHS",
]

EXAM_FINDINGS = [
    "Alert and oriented x3", "No acute distress", "Lungs clear bilaterally",
    "Heart regular rate and rhythm, no murmurs", "Abdomen soft, non-tender",
    "No peripheral edema", "Cranial nerves II-XII intact",
    "Normal gait and station", "Skin warm and dry", "No lymphadenopathy",
]

ALLERGIES = ["NKDA", "Penicillin", "Sulfa drugs", "Codeine", "Latex", "Aspirin", "Iodine contrast"]

LOCATIONS = ["MICU", "SICU", "CCU", "ED", "4 North", "3 South", "Oncology Unit", "Cardiac Care"]


# =============================================================================
# DOCUMENT BUILDER WITH COREFERENCE TRACKING
# =============================================================================

class DocumentBuilder:
    """Builds documents while tracking entity positions and coreference clusters."""
    
    def __init__(self):
        self.parts: List[str] = []
        self.entities: List[LabeledEntity] = []
        self.current_pos = 0
        self._coref_cluster_counter = 0
        self._active_clusters: Dict[str, int] = {}  # name -> cluster_id
    
    def add_text(self, text: str):
        """Add plain text (no entity)."""
        self.parts.append(text)
        self.current_pos += len(text)
    
    def add_entity(
        self, 
        text: str, 
        entity_type: str,
        coref_cluster_id: Optional[int] = None,
        is_anchor: bool = False,
    ):
        """Add text that is an entity."""
        start = self.current_pos
        end = start + len(text)
        
        self.entities.append(LabeledEntity(
            start=start,
            end=end,
            text=text,
            entity_type=entity_type,
            coref_cluster_id=coref_cluster_id,
            is_anchor=is_anchor,
        ))
        
        self.parts.append(text)
        self.current_pos = end
    
    def start_coref_cluster(self, name: str) -> int:
        """Start a new coreference cluster, returns cluster_id."""
        cluster_id = self._coref_cluster_counter
        self._coref_cluster_counter += 1
        self._active_clusters[name] = cluster_id
        return cluster_id
    
    def get_cluster_id(self, name: str) -> Optional[int]:
        """Get cluster_id for a named entity."""
        return self._active_clusters.get(name)
    
    def add_patient_anchor(self, full_name: str, entity_type: str = "NAME_PATIENT") -> int:
        """Add primary patient name mention, starts coreference cluster."""
        cluster_id = self.start_coref_cluster(full_name)
        self.add_entity(full_name, entity_type, coref_cluster_id=cluster_id, is_anchor=True)
        return cluster_id
    
    def add_patient_reference(
        self, 
        text: str, 
        cluster_name: str,
        entity_type: str = "NAME_PATIENT",
    ):
        """Add a reference to a patient (pronoun, partial name, etc.)."""
        cluster_id = self.get_cluster_id(cluster_name)
        self.add_entity(text, entity_type, coref_cluster_id=cluster_id, is_anchor=False)
    
    def add_line(self, text: str = ""):
        """Add text followed by newline."""
        if text:
            self.add_text(text)
        self.add_text("\n")
    
    def add_entity_line(self, prefix: str, value: str, entity_type: str, suffix: str = ""):
        """Add a line with prefix, entity value, and optional suffix."""
        self.add_text(prefix)
        self.add_entity(value, entity_type)
        if suffix:
            self.add_text(suffix)
        self.add_text("\n")
    
    def build(self) -> Tuple[str, List[LabeledEntity]]:
        """Return the complete text and entities."""
        return "".join(self.parts), self.entities


# =============================================================================
# PRONOUN/REFERENCE HELPERS
# =============================================================================

def get_pronoun(gender: str) -> str:
    """Get subject pronoun for gender."""
    return "He" if gender == "male" else "She"


def get_pronoun_object(gender: str) -> str:
    """Get object pronoun for gender."""
    return "him" if gender == "male" else "her"


def get_pronoun_possessive(gender: str) -> str:
    """Get possessive pronoun for gender."""
    return "his" if gender == "male" else "her"


def get_title(gender: str) -> str:
    """Get title for gender."""
    return "Mr." if gender == "male" else "Ms."


# =============================================================================
# DOCUMENT GENERATORS (WITH COREFERENCE)
# =============================================================================

def generate_admission_note() -> LabeledDocument:
    """Generate an admission note with labeled entities including pronouns."""
    builder = DocumentBuilder()
    
    # Generate values
    patient_name, first_name, last_name = generate_patient_name()
    provider_name, provider_last = generate_provider_name()
    dob = generate_dob()
    mrn = generate_mrn()
    admit_date = generate_date()
    phone = generate_phone()
    city, state = generate_city_state()
    age = random.randint(18, 95)
    gender = random.choice(["male", "female"])
    complaint = random.choice(COMPLAINTS)
    
    # Build document
    builder.add_line("ADMISSION NOTE")
    builder.add_line()
    
    # Primary patient mention (anchor)
    builder.add_text("Patient: ")
    cluster_id = builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("DOB: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    
    builder.add_text("Admission Date: ")
    builder.add_entity(admit_date, "DATE")
    builder.add_line()
    
    builder.add_line()
    builder.add_line(f"Chief Complaint: {complaint}")
    builder.add_line()
    
    # HPI with first name reference and pronoun
    builder.add_line("History of Present Illness:")
    builder.add_patient_reference(first_name, patient_name)
    builder.add_text(f" is a {age} year old {gender} who presents with {complaint}. ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(f" reports symptoms began {random.randint(1, 14)} days ago. ")
    builder.add_text("The patient denies any recent travel or sick contacts.")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Past Medical History:")
    for condition in random.sample(PMH_CONDITIONS, 3):
        builder.add_line(f"- {condition}")
    
    builder.add_line()
    builder.add_line("Medications:")
    for med in random.sample(MEDICATIONS, 4):
        builder.add_line(f"- {med}")
    
    builder.add_line()
    builder.add_line(f"Allergies: {random.choice(ALLERGIES)}")
    
    builder.add_line()
    builder.add_text("Social History: ")
    builder.add_patient_reference("The patient", patient_name)
    builder.add_text(" lives in ")
    builder.add_entity(city, "LOCATION")
    builder.add_text(", ")
    builder.add_entity(state, "LOCATION")
    builder.add_text(". ")
    builder.add_line(random.choice([
        "Non-smoker, occasional alcohol.",
        "Former smoker, quit 5 years ago.",
        "Denies tobacco, alcohol, illicit drugs.",
    ]))
    
    builder.add_line()
    builder.add_line("Physical Exam:")
    bp = f"{random.randint(110, 160)}/{random.randint(60, 95)}"
    hr = random.randint(60, 100)
    temp = f"{random.uniform(97.5, 100.5):.1f}"
    builder.add_line(f"Vitals: BP {bp}, HR {hr}, Temp {temp}F")
    for finding in random.sample(EXAM_FINDINGS, 3):
        builder.add_line(f"- {finding}")
    
    builder.add_line()
    builder.add_line("Assessment/Plan:")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(" will be admitted for further workup. ")
    builder.add_text(fake.sentence())
    builder.add_line()
    
    builder.add_line()
    builder.add_text("Attending: ")
    builder.add_entity(provider_name, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_text("Contact: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="admission_note",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": True},
    )


def generate_discharge_summary() -> LabeledDocument:
    """Generate a discharge summary with labeled entities including pronouns."""
    builder = DocumentBuilder()
    
    # Generate values
    patient_name, first_name, last_name = generate_patient_name()
    provider_name, provider_last = generate_provider_name()
    dob = generate_dob()
    mrn = generate_mrn()
    ssn = generate_ssn()
    admit_date = generate_date()
    discharge_date = generate_date()
    followup_date = generate_date()
    phone = generate_phone()
    email = generate_email()
    complaint = random.choice(COMPLAINTS)
    gender = random.choice(["male", "female"])
    
    # Build document
    builder.add_line("DISCHARGE SUMMARY")
    builder.add_line()
    
    builder.add_text("Patient: ")
    cluster_id = builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("DOB: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    
    builder.add_text("SSN: ")
    builder.add_entity(ssn, "SSN")
    builder.add_line()
    
    builder.add_text("Admission: ")
    builder.add_entity(admit_date, "DATE")
    builder.add_line()
    
    builder.add_text("Discharge: ")
    builder.add_entity(discharge_date, "DATE")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Discharge Diagnosis:")
    builder.add_line(random.choice(PMH_CONDITIONS))
    
    builder.add_line()
    builder.add_line("Hospital Course:")
    builder.add_patient_reference(first_name, patient_name)
    builder.add_text(f" was admitted on ")
    builder.add_entity(admit_date, "DATE")
    builder.add_text(f" for {complaint}. ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(" was started on appropriate therapy. ")
    builder.add_patient_reference("The patient", patient_name)
    builder.add_text(" showed steady improvement. ")
    builder.add_text(fake.sentence())
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Discharge Medications:")
    for med in random.sample(MEDICATIONS, 4):
        builder.add_line(f"- {med}")
    
    builder.add_line()
    builder.add_line("Follow-up:")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(f" should follow up with Dr. ")
    builder.add_entity(provider_last, "NAME_PROVIDER")
    builder.add_text(" on ")
    builder.add_entity(followup_date, "DATE")
    builder.add_line(".")
    
    builder.add_text("Call ")
    builder.add_entity(phone, "PHONE")
    builder.add_line(" with questions.")
    
    builder.add_line()
    builder.add_text("Discharge Physician: ")
    builder.add_entity(provider_name, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_text("Email: ")
    builder.add_entity(email, "EMAIL")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="discharge_summary",
        metadata={"has_ssn": True, "has_medical": True, "has_coreference": True},
    )


def generate_lab_report() -> LabeledDocument:
    """Generate a lab report with labeled entities."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    provider_name, _ = generate_provider_name()
    reviewer = fake.name()
    dob = generate_dob()
    mrn = generate_mrn()
    date = generate_date()
    time = generate_time()
    
    builder.add_line("LABORATORY REPORT")
    builder.add_line()
    
    builder.add_text("Patient: ")
    builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("DOB: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    
    builder.add_text("Collection Date: ")
    builder.add_entity(date, "DATE")
    builder.add_line()
    
    builder.add_text("Ordering Provider: ")
    builder.add_entity(provider_name, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Results:")
    
    lab_results = [
        f"WBC: {random.uniform(4.0, 12.0):.1f} (4.5-11.0)",
        f"Hgb: {random.uniform(10.0, 16.0):.1f} (12.0-16.0)",
        f"Plt: {random.randint(150, 400)} (150-400)",
        f"Na: {random.randint(135, 148)} (136-145)",
        f"K: {random.uniform(3.2, 5.5):.1f} (3.5-5.0)",
        f"Cr: {random.uniform(0.6, 2.0):.1f} (0.7-1.3)",
        f"Glucose: {random.randint(70, 200)} (70-100)",
    ]
    for result in random.sample(lab_results, 5):
        builder.add_line(f"  {result}")
    
    builder.add_line()
    builder.add_text("Reviewed by: ")
    builder.add_entity(reviewer, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_entity(date, "DATE")
    builder.add_text(" ")
    builder.add_entity(time, "TIME")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="lab_report",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": False},
    )


def generate_progress_note() -> LabeledDocument:
    """Generate a progress note with labeled entities including pronouns."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    provider_name, provider_last = generate_provider_name()
    mrn = generate_mrn()
    date = generate_date()
    pager = f"{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    gender = random.choice(["male", "female"])
    
    builder.add_line("PROGRESS NOTE")
    builder.add_line()
    
    builder.add_text("Date: ")
    builder.add_entity(date, "DATE")
    builder.add_line()
    
    builder.add_text("Patient: ")
    builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Subjective:")
    builder.add_patient_reference("The patient", patient_name)
    builder.add_text(" reports ")
    builder.add_text(fake.sentence().lower())
    builder.add_text(" ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(" denies ")
    builder.add_line(random.choice(["chest pain", "shortness of breath", "fever", "nausea"]) + ".")
    
    builder.add_line()
    builder.add_line("Objective:")
    bp = f"{random.randint(110, 160)}/{random.randint(60, 95)}"
    hr = random.randint(60, 100)
    builder.add_line(f"Vitals: BP {bp}, HR {hr}")
    builder.add_line(random.choice(EXAM_FINDINGS))
    
    builder.add_line()
    builder.add_line("Assessment:")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(" is doing well. ")
    builder.add_text(fake.sentence())
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Plan:")
    builder.add_text("Continue current management. Will see ")
    builder.add_patient_reference(get_pronoun_object(gender), patient_name)
    builder.add_text(" again in 1 week.")
    builder.add_line()
    
    builder.add_line()
    builder.add_entity(provider_name, "NAME_PROVIDER")
    builder.add_line(", MD")
    
    builder.add_text("Pager: ")
    builder.add_entity(pager, "PHONE")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="progress_note",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": True},
    )


def generate_referral_letter() -> LabeledDocument:
    """Generate a referral letter with labeled entities including pronouns."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    referring_doc, referring_last = generate_provider_name()
    receiving_doc, receiving_last = generate_provider_name()
    dob = generate_dob()
    mrn = generate_mrn()
    phone = generate_phone()
    fax = generate_phone()
    date = generate_date()
    address = generate_address()
    city, state = generate_city_state()
    gender = random.choice(["male", "female"])
    complaint = random.choice(COMPLAINTS)
    
    builder.add_line(f"Date: ")
    builder.add_entity(date, "DATE")
    builder.add_line()
    builder.add_line()
    
    builder.add_entity(receiving_doc, "NAME_PROVIDER")
    builder.add_line()
    builder.add_line("Department of Cardiology")
    builder.add_entity(address, "ADDRESS")
    builder.add_line()
    builder.add_entity(city, "LOCATION")
    builder.add_text(", ")
    builder.add_entity(state, "LOCATION")
    builder.add_line()
    builder.add_line()
    
    builder.add_text("RE: ")
    builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("DOB: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    builder.add_line()
    
    builder.add_text("Dear Dr. ")
    builder.add_entity(receiving_last, "NAME_PROVIDER")
    builder.add_line(",")
    builder.add_line()
    
    builder.add_text("I am referring ")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(f" for evaluation of {complaint}. ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(" has been experiencing symptoms for the past few weeks. ")
    builder.add_patient_reference("The patient", patient_name)
    builder.add_text(" would benefit from your expertise.")
    builder.add_line()
    builder.add_line()
    
    builder.add_text("Please see ")
    builder.add_patient_reference(get_pronoun_object(gender), patient_name)
    builder.add_text(" at your earliest convenience.")
    builder.add_line()
    builder.add_line()
    
    builder.add_line("Sincerely,")
    builder.add_line()
    builder.add_entity(referring_doc, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_text("Phone: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    builder.add_text("Fax: ")
    builder.add_entity(fax, "FAX")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="referral_letter",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": True},
    )


def generate_insurance_form() -> LabeledDocument:
    """Generate an insurance form with PII."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    dob = generate_dob()
    ssn = generate_ssn()
    phone = generate_phone()
    email = generate_email()
    address = generate_address()
    city, state = generate_city_state()
    account = generate_account_number()
    employer = fake.company()
    
    builder.add_line("INSURANCE ENROLLMENT FORM")
    builder.add_line()
    
    builder.add_text("Name: ")
    builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("Date of Birth: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("Social Security Number: ")
    builder.add_entity(ssn, "SSN")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Contact Information:")
    
    builder.add_text("Address: ")
    builder.add_entity(address, "ADDRESS")
    builder.add_line()
    
    builder.add_text("City: ")
    builder.add_entity(city, "LOCATION")
    builder.add_text(" State: ")
    builder.add_entity(state, "LOCATION")
    builder.add_line()
    
    builder.add_text("Phone: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    builder.add_text("Email: ")
    builder.add_entity(email, "EMAIL")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Employment Information:")
    builder.add_line(f"Employer: {employer}")
    
    builder.add_text("Member ID: ")
    builder.add_entity(account, "ACCOUNT_NUMBER")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="insurance_form",
        metadata={"has_ssn": True, "has_medical": False, "has_coreference": False},
    )


def generate_brief_note() -> LabeledDocument:
    """Generate a brief clinical note with pronouns."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    provider_last = fake.last_name()
    mrn = generate_mrn()
    date = generate_date()
    phone = generate_phone()
    location = random.choice(LOCATIONS)
    age = random.randint(18, 95)
    gender = random.choice(["male", "female"])
    complaint = random.choice(COMPLAINTS)
    
    builder.add_text("Patient ")
    builder.add_patient_anchor(patient_name)
    builder.add_text(" (")
    builder.add_entity(mrn, "MRN")
    builder.add_text(f") seen in {location} on ")
    builder.add_entity(date, "DATE")
    builder.add_line(".")
    builder.add_line()
    
    builder.add_line(f"CC: {complaint}")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(f" is {age}yo {gender}. ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(f" reports {fake.sentence().lower()}")
    builder.add_line()
    builder.add_line()
    
    builder.add_line(f"A: {random.choice(PMH_CONDITIONS)}")
    builder.add_text("P: Will follow up with ")
    builder.add_patient_reference("the patient", patient_name)
    builder.add_text(" in 2 weeks.")
    builder.add_line()
    builder.add_line()
    
    builder.add_text("- Dr. ")
    builder.add_entity(provider_last, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_text("  ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="brief_note",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": True},
    )


def generate_consultation_note() -> LabeledDocument:
    """Generate a consultation note with rich coreference patterns."""
    builder = DocumentBuilder()
    
    patient_name, first_name, last_name = generate_patient_name()
    consulting_doc, consulting_last = generate_provider_name()
    requesting_doc, requesting_last = generate_provider_name()
    dob = generate_dob()
    mrn = generate_mrn()
    date = generate_date()
    phone = generate_phone()
    age = random.randint(18, 95)
    gender = random.choice(["male", "female"])
    complaint = random.choice(COMPLAINTS)
    
    builder.add_line("CONSULTATION NOTE")
    builder.add_line()
    
    builder.add_text("Date: ")
    builder.add_entity(date, "DATE")
    builder.add_line()
    
    builder.add_text("Patient: ")
    builder.add_patient_anchor(patient_name)
    builder.add_line()
    
    builder.add_text("DOB: ")
    builder.add_entity(dob, "DATE_DOB")
    builder.add_line()
    
    builder.add_text("MRN: ")
    builder.add_entity(mrn, "MRN")
    builder.add_line()
    
    builder.add_text("Requesting Physician: ")
    builder.add_entity(requesting_doc, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Reason for Consultation:")
    builder.add_text(f"Thank you for asking me to see ")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(f" regarding {complaint}.")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("History:")
    builder.add_patient_reference(first_name, patient_name)
    builder.add_text(f" is a {age} year old {gender} with a history of ")
    builder.add_text(f"{random.choice(PMH_CONDITIONS).lower()}. ")
    builder.add_patient_reference(get_pronoun(gender), patient_name)
    builder.add_text(f" reports that {get_pronoun_possessive(gender)} symptoms began recently. ")
    builder.add_patient_reference("The patient", patient_name)
    builder.add_text(" has tried over-the-counter medications without relief.")
    builder.add_line()
    
    builder.add_line()
    builder.add_line("Physical Examination:")
    builder.add_text("On examination, ")
    builder.add_patient_reference(f"{get_title(gender)} {last_name}", patient_name)
    builder.add_text(" appears comfortable. ")
    for finding in random.sample(EXAM_FINDINGS, 2):
        builder.add_line(f"- {finding}")
    
    builder.add_line()
    builder.add_line("Assessment and Recommendations:")
    builder.add_text("Based on my evaluation, I believe ")
    builder.add_patient_reference(get_pronoun_object(gender), patient_name)
    builder.add_text(" would benefit from ")
    builder.add_text(f"{random.choice(MEDICATIONS).lower()}. ")
    builder.add_text("I will see ")
    builder.add_patient_reference("the patient", patient_name)
    builder.add_text(" again in 2 weeks.")
    builder.add_line()
    
    builder.add_line()
    builder.add_text("Consulting Physician: ")
    builder.add_entity(consulting_doc, "NAME_PROVIDER")
    builder.add_line()
    
    builder.add_text("Phone: ")
    builder.add_entity(phone, "PHONE")
    builder.add_line()
    
    text, entities = builder.build()
    
    return LabeledDocument(
        id=str(uuid.uuid4()),
        text=text,
        entities=entities,
        doc_type="consultation_note",
        metadata={"has_ssn": False, "has_medical": True, "has_coreference": True},
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================

GENERATORS = [
    (generate_admission_note, 0.18),
    (generate_discharge_summary, 0.18),
    (generate_lab_report, 0.12),
    (generate_progress_note, 0.15),
    (generate_referral_letter, 0.10),
    (generate_insurance_form, 0.08),
    (generate_brief_note, 0.09),
    (generate_consultation_note, 0.10),  # NEW
]


def generate_synthetic_document() -> LabeledDocument:
    """Generate a random synthetic document."""
    generators, weights = zip(*GENERATORS)
    generator = random.choices(generators, weights=weights, k=1)[0]
    return generator()


def generate_synthetic_dataset(n: int = 1000) -> List[LabeledDocument]:
    """Generate n synthetic documents."""
    return [generate_synthetic_document() for _ in range(n)]


def validate_document(doc: LabeledDocument) -> bool:
    """Validate that entity spans are correct."""
    for entity in doc.entities:
        extracted = doc.text[entity.start:entity.end]
        if extracted != entity.text:
            print(f"MISMATCH in {doc.id}:")
            print(f"  Expected: '{entity.text}'")
            print(f"  Got: '{extracted}'")
            print(f"  Span: [{entity.start}:{entity.end}]")
            return False
    return True


def validate_dataset(docs: List[LabeledDocument]) -> Tuple[int, int]:
    """Validate all documents. Returns (valid_count, invalid_count)."""
    valid = 0
    invalid = 0
    for doc in docs:
        if validate_document(doc):
            valid += 1
        else:
            invalid += 1
    return valid, invalid


def get_coreference_stats(docs: List[LabeledDocument]) -> Dict:
    """Get statistics about coreference in the dataset."""
    total_entities = 0
    anchor_entities = 0
    reference_entities = 0
    docs_with_coref = 0
    
    for doc in docs:
        has_coref = False
        for entity in doc.entities:
            total_entities += 1
            if entity.coref_cluster_id is not None:
                has_coref = True
                if entity.is_anchor:
                    anchor_entities += 1
                else:
                    reference_entities += 1
        if has_coref:
            docs_with_coref += 1
    
    return {
        "total_documents": len(docs),
        "docs_with_coreference": docs_with_coref,
        "total_entities": total_entities,
        "anchor_entities": anchor_entities,
        "reference_entities": reference_entities,
        "avg_references_per_doc": reference_entities / len(docs) if docs else 0,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    print(f"Generating {n} synthetic documents...")
    docs = generate_synthetic_dataset(n)
    
    print("Validating...")
    valid, invalid = validate_dataset(docs)
    print(f"Valid: {valid}, Invalid: {invalid}")
    
    print("\nCoreference stats:")
    coref_stats = get_coreference_stats(docs)
    for key, value in coref_stats.items():
        print(f"  {key}: {value}")
    
    if invalid == 0:
        print("\nSample document:")
        sample = docs[0]
        print(f"Type: {sample.doc_type}")
        print(f"Entities: {len(sample.entities)}")
        for e in sample.entities[:10]:
            coref_info = f" [cluster={e.coref_cluster_id}, anchor={e.is_anchor}]" if e.coref_cluster_id is not None else ""
            print(f"  [{e.start}:{e.end}] {e.entity_type}: '{e.text}'{coref_info}")
        print("\nText preview:")
        print(sample.text[:500])
