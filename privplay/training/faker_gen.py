"""Faker-based synthetic clinical document generator."""

from typing import List, Optional
import uuid
from faker import Faker
import random

from ..types import Document

fake = Faker()


# Clinical note templates
TEMPLATES = [
    # Admission note
    """ADMISSION NOTE

Patient: {name}
DOB: {dob}
MRN: {mrn}
Admission Date: {admit_date}

Chief Complaint: {complaint}

History of Present Illness:
{name_short} is a {age} year old {gender} who presents with {complaint}. 
Patient reports symptoms began {duration} ago. {history}

Past Medical History:
{pmh}

Medications:
{medications}

Allergies: {allergies}

Social History:
Patient lives in {city}, {state}. {social}

Physical Exam:
Vitals: BP {bp}, HR {hr}, Temp {temp}F, RR {rr}, SpO2 {spo2}%
{exam}

Assessment/Plan:
{assessment}

Attending: {doctor}
Contact: {phone}
""",
    
    # Progress note
    """PROGRESS NOTE

Date: {date}
Patient: {name}
MRN: {mrn}

Subjective:
Patient reports {subjective}. {name_short} denies {denies}.

Objective:
Vitals: BP {bp}, HR {hr}, Temp {temp}F
{objective}

Assessment:
{assessment}

Plan:
{plan}

{doctor}, MD
Pager: {pager}
""",

    # Discharge summary
    """DISCHARGE SUMMARY

Patient: {name}
DOB: {dob}  
MRN: {mrn}
SSN: {ssn}
Admission: {admit_date}
Discharge: {discharge_date}

Discharge Diagnosis:
{diagnosis}

Hospital Course:
{name_short} was admitted on {admit_date} for {complaint}. {course}

Discharge Medications:
{medications}

Follow-up:
Appointment with Dr. {doctor_last} on {followup_date} at {time}.
Call {phone} with questions.

Discharge Physician: {doctor}
Email: {email}
""",

    # Lab result
    """LABORATORY REPORT

Patient: {name}
DOB: {dob}
MRN: {mrn}
Collection Date: {date}
Ordering Provider: Dr. {doctor}

Results:
{labs}

Reviewed by: {reviewer}
{date} {time}
""",

    # Radiology report
    """RADIOLOGY REPORT

Patient: {name}
MRN: {mrn}
DOB: {dob}
Exam Date: {date}
Exam: {exam_type}

Clinical History: {history}

Findings:
{findings}

Impression:
{impression}

Radiologist: {doctor}
Dictated: {date} {time}
Transcribed: {transcribe_date}
""",

    # Brief clinical note
    """Patient {name} ({mrn}) seen in {location} on {date}.

CC: {complaint}
{name_short} is {age}yo {gender}. {brief_hpi}

A: {assessment}
P: {plan}

- Dr. {doctor_last}
  {phone}
""",

    # Nursing note
    """NURSING NOTE

{date} {time}
Patient: {name}
Room: {room}
MRN: {mrn}

Assessment:
{nursing_assessment}

Vitals: BP {bp}, HR {hr}, Temp {temp}, RR {rr}, SpO2 {spo2}%
Pain: {pain}/10

Interventions:
{interventions}

Patient/Family Education:
{education}

{nurse}, RN
Contact: {phone}
""",
]


# Clinical data pools
COMPLAINTS = [
    "chest pain", "shortness of breath", "abdominal pain", "headache",
    "fever", "cough", "nausea and vomiting", "back pain", "dizziness",
    "weakness", "fatigue", "leg swelling", "palpitations", "syncope",
]

PMH_CONDITIONS = [
    "Hypertension", "Diabetes mellitus type 2", "Hyperlipidemia",
    "Coronary artery disease", "COPD", "Asthma", "CHF", "Atrial fibrillation",
    "Chronic kidney disease", "GERD", "Osteoarthritis", "Depression",
]

MEDICATIONS = [
    "Metformin 500mg BID", "Lisinopril 10mg daily", "Atorvastatin 40mg QHS",
    "Aspirin 81mg daily", "Metoprolol 25mg BID", "Omeprazole 20mg daily",
    "Amlodipine 5mg daily", "Levothyroxine 50mcg daily", "Gabapentin 300mg TID",
]

EXAM_FINDINGS = [
    "Alert and oriented x3", "No acute distress", "Lungs clear bilaterally",
    "Heart regular rate and rhythm, no murmurs", "Abdomen soft, non-tender",
    "No peripheral edema", "Cranial nerves II-XII intact",
]

LOCATIONS = [
    "MICU", "SICU", "CCU", "ED", "Floor 4", "Room 412", "Building A",
    "Clinic 3", "Oncology Unit", "Cardiac Care", "ER Bay 5",
]

LAB_RESULTS = [
    "WBC: 8.2 (4.5-11.0)", "Hgb: 12.5 (12.0-16.0)", "Plt: 245 (150-400)",
    "Na: 138 (136-145)", "K: 4.2 (3.5-5.0)", "Cr: 1.1 (0.7-1.3)",
    "Glucose: 126 (70-100)", "BUN: 18 (7-20)", "AST: 25 (10-40)",
]


def generate_clinical_note() -> tuple[str, dict]:
    """Generate a synthetic clinical note with known PHI locations."""
    template = random.choice(TEMPLATES)
    
    # Generate PHI values
    name = fake.name()
    name_parts = name.split()
    name_short = name_parts[0]
    doctor = f"Dr. {fake.last_name()}"
    doctor_last = fake.last_name()
    
    values = {
        "name": name,
        "name_short": name_short,
        "doctor": doctor,
        "doctor_last": doctor_last,
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y"),
        "mrn": str(random.randint(10000000, 99999999)),
        "ssn": fake.ssn(),
        "admit_date": fake.date_this_year().strftime("%m/%d/%Y"),
        "discharge_date": fake.date_this_year().strftime("%m/%d/%Y"),
        "date": fake.date_this_year().strftime("%m/%d/%Y"),
        "followup_date": fake.date_this_year().strftime("%m/%d/%Y"),
        "transcribe_date": fake.date_this_year().strftime("%m/%d/%Y"),
        "time": fake.time(),
        "age": random.randint(18, 95),
        "gender": random.choice(["male", "female"]),
        "phone": fake.phone_number(),
        "pager": f"{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        "email": fake.email(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "room": f"{random.randint(1, 8)}{random.randint(10, 50)}",
        "location": random.choice(LOCATIONS),
        
        # Clinical content (not PHI)
        "complaint": random.choice(COMPLAINTS),
        "duration": f"{random.randint(1, 14)} days",
        "history": fake.paragraph(nb_sentences=2),
        "pmh": "\n".join(f"- {c}" for c in random.sample(PMH_CONDITIONS, 3)),
        "medications": "\n".join(f"- {m}" for m in random.sample(MEDICATIONS, 4)),
        "allergies": random.choice(["NKDA", "Penicillin", "Sulfa", "Codeine"]),
        "social": random.choice([
            "Non-smoker, occasional alcohol.",
            "Former smoker, quit 5 years ago.",
            "Denies tobacco, alcohol, illicit drugs.",
        ]),
        "bp": f"{random.randint(110, 160)}/{random.randint(60, 95)}",
        "hr": str(random.randint(60, 100)),
        "temp": f"{random.uniform(97.5, 100.5):.1f}",
        "rr": str(random.randint(12, 22)),
        "spo2": str(random.randint(94, 100)),
        "exam": "\n".join(random.sample(EXAM_FINDINGS, 4)),
        "assessment": fake.paragraph(nb_sentences=2),
        "plan": fake.paragraph(nb_sentences=2),
        "subjective": fake.sentence(),
        "denies": random.choice(["chest pain", "shortness of breath", "fever", "nausea"]),
        "objective": random.choice(EXAM_FINDINGS),
        "diagnosis": random.choice(PMH_CONDITIONS),
        "course": fake.paragraph(nb_sentences=3),
        "labs": "\n".join(random.sample(LAB_RESULTS, 5)),
        "reviewer": fake.name(),
        "exam_type": random.choice(["Chest X-ray", "CT Abdomen", "MRI Brain", "Ultrasound"]),
        "findings": fake.paragraph(nb_sentences=3),
        "impression": fake.sentence(),
        "brief_hpi": fake.sentence(),
        "nursing_assessment": fake.paragraph(nb_sentences=2),
        "interventions": fake.paragraph(nb_sentences=1),
        "education": fake.sentence(),
        "nurse": fake.name(),
        "pain": str(random.randint(0, 8)),
    }
    
    try:
        content = template.format(**values)
    except KeyError as e:
        # Fallback to simple note if template has issues
        content = f"""Patient: {values['name']}
DOB: {values['dob']}
MRN: {values['mrn']}

Note: {values['complaint']}. Patient is a {values['age']} year old {values['gender']}.

Dr. {values['doctor_last']}
{values['phone']}
"""
    
    return content, values


def generate_documents(n: int = 10) -> List[Document]:
    """Generate n synthetic clinical documents."""
    documents = []
    
    for _ in range(n):
        content, _ = generate_clinical_note()
        
        doc = Document(
            id=str(uuid.uuid4()),
            content=content,
            source="faker",
        )
        documents.append(doc)
    
    return documents


def generate_simple_pii_examples(n: int = 20) -> List[Document]:
    """Generate simple PII examples for testing."""
    documents = []
    
    templates = [
        "Contact {name} at {phone} or {email}.",
        "Patient {name}, DOB {dob}, MRN {mrn}.",
        "Send records to {address}.",
        "SSN: {ssn}, Account: {account}",
        "Dr. {doctor} saw {name} on {date}.",
        "{name}'s credit card ending in {cc_last4}.",
        "IP address {ip} accessed record for {name}.",
        "Meeting with {name} at {address} on {date}.",
    ]
    
    for _ in range(n):
        template = random.choice(templates)
        
        values = {
            "name": fake.name(),
            "phone": fake.phone_number(),
            "email": fake.email(),
            "dob": fake.date_of_birth().strftime("%m/%d/%Y"),
            "mrn": str(random.randint(10000000, 99999999)),
            "address": fake.address().replace("\n", ", "),
            "ssn": fake.ssn(),
            "account": str(random.randint(1000000000, 9999999999)),
            "doctor": fake.name(),
            "date": fake.date_this_year().strftime("%m/%d/%Y"),
            "cc_last4": str(random.randint(1000, 9999)),
            "ip": fake.ipv4(),
        }
        
        try:
            content = template.format(**values)
        except KeyError:
            content = f"Contact {values['name']} at {values['phone']}."
        
        doc = Document(
            id=str(uuid.uuid4()),
            content=content,
            source="faker",
        )
        documents.append(doc)
    
    return documents
