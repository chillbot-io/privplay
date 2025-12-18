#!/usr/bin/env python3
"""
High-Quality Synthetic PII Data Generator

Generates diverse, realistic PII samples with perfect labels.
Focus on:
- Varied contexts (emails, forms, conversations, documents)
- Multicultural names
- Multiple formats per entity type
- Different sentence positions
- Realistic surrounding text

Usage:
    python3 create_synthetic_pii.py --count 15000 --output synthetic_pii.json
"""

import json
import random
import string
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# =============================================================================
# NAME DATA - Diverse, multicultural
# =============================================================================

FIRST_NAMES = [
    # English
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica",
    "Sarah", "Emily", "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia",
    # Hispanic
    "Carlos", "Miguel", "José", "Luis", "Juan", "Maria", "Rosa", "Carmen", "Ana", "Sofia",
    "Diego", "Alejandro", "Gabriela", "Valentina", "Lucia", "Camila", "Mariana",
    # Asian
    "Wei", "Fang", "Ming", "Hui", "Yan", "Chen", "Lin", "Yuki", "Kenji", "Hiroshi",
    "Aiko", "Sakura", "Jin", "Soo", "Min", "Hyun", "Raj", "Priya", "Amit", "Deepak",
    "Ananya", "Neha", "Vikram", "Arun", "Kavitha", "Lakshmi",
    # European
    "Hans", "Klaus", "Stefan", "Anna", "Katrina", "Ingrid", "Pierre", "Jean", "Marie",
    "François", "Giovanni", "Marco", "Lucia", "Francesca", "Alessandro", "Dmitri",
    "Natasha", "Olga", "Svetlana", "Erik", "Lars", "Astrid", "Freya",
    # African
    "Kwame", "Kofi", "Amara", "Zara", "Fatima", "Omar", "Ahmed", "Aisha", "Yusuf",
    "Chinwe", "Ngozi", "Emeka", "Chidi", "Adaeze", "Nneka",
    # Middle Eastern
    "Mohammed", "Ali", "Hassan", "Fatima", "Layla", "Sara", "Youssef", "Nadia",
]

LAST_NAMES = [
    # English
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
    "Martin", "Lee", "Thompson", "White", "Harris", "Clark", "Lewis", "Walker",
    # Hispanic
    "Hernandez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez", "Torres",
    "Flores", "Rivera", "Gomez", "Diaz", "Reyes", "Morales", "Cruz", "Ortiz",
    # Asian
    "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Wu", "Kim", "Park",
    "Choi", "Tanaka", "Yamamoto", "Nakamura", "Suzuki", "Patel", "Shah", "Kumar",
    "Singh", "Sharma", "Gupta", "Nguyen", "Tran", "Le", "Pham",
    # European
    "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
    "Rossi", "Russo", "Ferrari", "Esposito", "Bernard", "Dubois", "Moreau",
    "Kowalski", "Novak", "Horvat", "Johansson", "Larsson", "Nielsen", "Hansen",
    # African/Middle Eastern
    "Ibrahim", "Hassan", "Ali", "Ahmed", "Okonkwo", "Mensah", "Diallo", "Toure",
]

PREFIXES = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", ""]
SUFFIXES = ["Jr.", "Sr.", "III", "MD", "PhD", "Esq.", ""]

# =============================================================================
# EMAIL DOMAINS
# =============================================================================

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "aol.com", "mail.com", "protonmail.com", "tutanota.com", "zoho.com",
    "company.com", "business.org", "work.net", "corp.io", "enterprise.co",
    "university.edu", "college.edu", "hospital.org", "clinic.net",
]

# =============================================================================
# CONTEXT TEMPLATES - Diverse document types
# =============================================================================

# Email contexts
EMAIL_CONTEXTS = [
    "Please contact {entity} for more information.",
    "You can reach me at {entity} anytime.",
    "Send your response to {entity} by Friday.",
    "For questions, email {entity} directly.",
    "CC: {entity}",
    "From: {entity}\nSubject: Important Update",
    "Reply to {entity} with your availability.",
    "His email address is {entity} - feel free to reach out.",
    "My work email is {entity}, personal is below.",
    "Forward this to {entity} when you get a chance.",
    "Best regards,\n{entity}",
    "Contact: {entity}",
    "Email the team lead at {entity} for approval.",
    "Add {entity} to the meeting invite.",
    "She can be reached via email: {entity}",
]

# Phone contexts  
PHONE_CONTEXTS = [
    "Call {entity} to schedule an appointment.",
    "For immediate assistance, dial {entity}.",
    "Phone: {entity}",
    "Reach us at {entity} during business hours.",
    "Mobile: {entity}",
    "His cell phone is {entity}, but try the office first.",
    "Contact number: {entity}",
    "Please call {entity} at your earliest convenience.",
    "Fax: {entity}",
    "Emergency contact: {entity}",
    "You can text me at {entity} if that's easier.",
    "The main line is {entity}, press 2 for support.",
    "Leave a voicemail at {entity} if no answer.",
    "Tel: {entity}",
    "Direct line: {entity}",
]

# Name contexts
NAME_CONTEXTS = [
    "Dear {entity},",
    "The patient, {entity}, was admitted yesterday.",
    "Please welcome our new colleague, {entity}.",
    "Signed: {entity}",
    "{entity} will be leading the project.",
    "According to {entity}, the deadline is flexible.",
    "I spoke with {entity} this morning.",
    "Meeting attendees: {entity}, plus three others.",
    "The account manager is {entity}.",
    "{entity} has approved the request.",
    "Report prepared by {entity}",
    "Attention: {entity}",
    "Cc: {entity}",
    "As {entity} mentioned in the last meeting...",
    "Thank you, {entity}, for your assistance.",
    "{entity} joined the company in 2019.",
    "The physician, {entity}, recommended further testing.",
    "Contact {entity} for technical support.",
    "Referred by {entity}",
    "{entity}'s schedule is attached.",
]

# SSN contexts
SSN_CONTEXTS = [
    "SSN: {entity}",
    "Social Security Number: {entity}",
    "Patient SSN: {entity}",
    "Applicant's SSN is {entity}",
    "For verification, provide your SSN: {entity}",
    "Tax ID/SSN: {entity}",
    "Social: {entity}",
    "SSN (last 4: XXX-XX-{last4}): {entity}",
    "Employee SSN: {entity}",
    "The SSN on file is {entity}",
    "Please verify SSN {entity} matches our records.",
    "Social Security #: {entity}",
]

# Credit card contexts
CC_CONTEXTS = [
    "Card Number: {entity}",
    "Credit Card: {entity}",
    "Payment card ending in {last4}: {entity}",
    "Visa: {entity}",
    "MasterCard: {entity}",
    "Card on file: {entity}",
    "Charged to card {entity}",
    "Enter card number: {entity}",
    "Primary payment method: {entity}",
    "Credit card number is {entity}",
    "The transaction was processed using card {entity}.",
    "Card #: {entity}",
]

# Address contexts
ADDRESS_CONTEXTS = [
    "Address: {entity}",
    "Ship to: {entity}",
    "Located at {entity}",
    "Our office is at {entity}.",
    "Mailing address: {entity}",
    "Please send documents to {entity}",
    "The property at {entity} is available.",
    "Delivery address: {entity}",
    "Home address: {entity}",
    "Residence: {entity}",
    "Patient resides at {entity}",
    "Business address: {entity}",
    "Send to: {entity}, Attn: Billing",
    "The incident occurred at {entity}.",
    "Moving to {entity} next month.",
]

# Date contexts
DATE_CONTEXTS = [
    "Date of Birth: {entity}",
    "DOB: {entity}",
    "Born on {entity}",
    "Appointment scheduled for {entity}",
    "Effective date: {entity}",
    "The meeting is on {entity}.",
    "Submitted on {entity}",
    "Date: {entity}",
    "As of {entity}, the policy changed.",
    "Patient DOB is {entity}",
    "Event date: {entity}",
    "Start date: {entity}",
    "Expiration: {entity}",
    "Valid until {entity}",
    "Birthday: {entity}",
]

# Username contexts
USERNAME_CONTEXTS = [
    "Username: {entity}",
    "Login ID: {entity}",
    "User: {entity}",
    "Account name: {entity}",
    "Handle: @{entity}",
    "Log in as {entity}",
    "Your username is {entity}",
    "Created by user {entity}",
    "Author: {entity}",
    "Assigned to {entity}",
    "Posted by {entity}",
    "Submitted by user {entity}",
    "Profile: {entity}",
    "User ID {entity} has been created.",
    "Contact username: {entity}",
]

# Password contexts  
PASSWORD_CONTEXTS = [
    "Password: {entity}",
    "Temporary password: {entity}",
    "Your new password is {entity}",
    "Default pwd: {entity}",
    "Reset to: {entity}",
    "Initial password: {entity}",
    "Passcode: {entity}",
    "Access code: {entity}",
    "PIN: {entity}",
    "Security code: {entity}",
    "One-time password: {entity}",
    "Login credentials - password: {entity}",
]

# IP Address contexts
IP_CONTEXTS = [
    "IP Address: {entity}",
    "Server IP: {entity}",
    "Connection from {entity}",
    "Blocked IP: {entity}",
    "Access from {entity} was denied.",
    "Client IP: {entity}",
    "The request originated from {entity}.",
    "Whitelist IP {entity}",
    "Source: {entity}",
    "Network address: {entity}",
    "Host: {entity}",
    "Remote IP: {entity}",
    "Logged in from IP {entity}",
    "Firewall rule for {entity}",
]

# URL contexts
URL_CONTEXTS = [
    "Visit {entity} for more details.",
    "Link: {entity}",
    "Website: {entity}",
    "See {entity} for documentation.",
    "API endpoint: {entity}",
    "Redirect to {entity}",
    "Homepage: {entity}",
    "Resource URL: {entity}",
    "Access the portal at {entity}",
    "Click here: {entity}",
    "More information at {entity}",
    "Download from {entity}",
]

# Account number contexts
ACCOUNT_CONTEXTS = [
    "Account Number: {entity}",
    "Account #: {entity}",
    "Bank account: {entity}",
    "Routing/Account: XXXXXX/{entity}",
    "Deposit to account {entity}",
    "Reference account {entity}",
    "Acct: {entity}",
    "Account ending in {last4}: {entity}",
    "Wire to account {entity}",
    "Checking account: {entity}",
    "Savings account: {entity}",
]

# ZIP code contexts
ZIP_CONTEXTS = [
    "ZIP: {entity}",
    "Zip Code: {entity}",
    "Postal code: {entity}",
    "{city}, {state} {entity}",
    "ZIP code {entity}",
    "Mail to ZIP {entity}",
    "Area code/ZIP: {entity}",
]

# IBAN contexts
IBAN_CONTEXTS = [
    "IBAN: {entity}",
    "International transfer to {entity}",
    "Bank IBAN: {entity}",
    "Wire to IBAN {entity}",
    "Account IBAN: {entity}",
    "Payment IBAN: {entity}",
]


# =============================================================================
# GENERATORS
# =============================================================================

def generate_name() -> Tuple[str, str]:
    """Generate a realistic full name with optional prefix/suffix."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    # Sometimes add prefix
    if random.random() < 0.15:
        prefix = random.choice([p for p in PREFIXES if p])
        full = f"{prefix} {first} {last}"
    # Sometimes add suffix
    elif random.random() < 0.05:
        suffix = random.choice([s for s in SUFFIXES if s])
        full = f"{first} {last}, {suffix}"
    else:
        full = f"{first} {last}"
    
    return full, "NAME_PERSON"


def generate_email() -> Tuple[str, str]:
    """Generate a realistic email address."""
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower()
    domain = random.choice(EMAIL_DOMAINS)
    
    patterns = [
        f"{first}.{last}@{domain}",
        f"{first}_{last}@{domain}",
        f"{first}{last}@{domain}",
        f"{first[0]}{last}@{domain}",
        f"{first}.{last[0]}@{domain}",
        f"{first}{random.randint(1, 99)}@{domain}",
        f"{last}.{first}@{domain}",
        f"{first}_{random.randint(100, 999)}@{domain}",
    ]
    
    email = random.choice(patterns)
    # Sometimes uppercase first letter
    if random.random() < 0.1:
        email = email[0].upper() + email[1:]
    
    return email, "EMAIL"


def generate_phone() -> Tuple[str, str]:
    """Generate a phone number in various formats."""
    area = random.randint(200, 999)
    exchange = random.randint(200, 999)
    subscriber = random.randint(1000, 9999)
    
    formats = [
        f"({area}) {exchange}-{subscriber}",
        f"{area}-{exchange}-{subscriber}",
        f"{area}.{exchange}.{subscriber}",
        f"{area} {exchange} {subscriber}",
        f"+1-{area}-{exchange}-{subscriber}",
        f"+1 ({area}) {exchange}-{subscriber}",
        f"1-{area}-{exchange}-{subscriber}",
        f"{area}{exchange}{subscriber}",  # No separators
    ]
    
    return random.choice(formats), "PHONE"


def generate_ssn() -> Tuple[str, str]:
    """Generate a valid-format SSN."""
    # Avoid invalid area numbers
    area = random.randint(1, 665)
    if area >= 666:
        area = random.randint(667, 899)
    
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    
    formats = [
        f"{area:03d}-{group:02d}-{serial:04d}",
        f"{area:03d} {group:02d} {serial:04d}",
    ]
    
    return random.choice(formats), "SSN"


def luhn_checksum_digit(partial: str) -> str:
    """Calculate Luhn check digit."""
    digits = [int(d) for d in partial]
    # Double every second digit from right
    for i in range(len(digits) - 1, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    total = sum(digits)
    return str((10 - (total % 10)) % 10)


def generate_credit_card() -> Tuple[str, str]:
    """Generate a Luhn-valid credit card number."""
    # Prefixes for different card types
    prefixes = [
        "4",  # Visa
        "5" + str(random.randint(1, 5)),  # MasterCard
        "37",  # Amex
        "6011",  # Discover
    ]
    
    prefix = random.choice(prefixes)
    
    if prefix.startswith("37"):
        # Amex is 15 digits
        length = 15
    else:
        length = 16
    
    # Generate random digits
    partial = prefix + ''.join([str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1)])
    check = luhn_checksum_digit(partial)
    cc = partial + check
    
    # Format
    formats = [
        cc,  # No separators
        ' '.join([cc[i:i+4] for i in range(0, len(cc), 4)]),  # Spaced
        '-'.join([cc[i:i+4] for i in range(0, len(cc), 4)]),  # Dashed
    ]
    
    return random.choice(formats), "CREDIT_CARD"


def generate_address() -> Tuple[str, str]:
    """Generate a realistic street address."""
    number = random.randint(1, 9999)
    
    street_names = [
        "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lincoln",
        "Park", "Lake", "Hill", "River", "Forest", "Meadow", "Spring", "Valley",
        "Highland", "Sunset", "Broadway", "Market", "Church", "School", "Mill",
        "North", "South", "East", "West", "Center", "First", "Second", "Third",
    ]
    
    street_types = [
        "Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd",
        "Drive", "Dr", "Lane", "Ln", "Way", "Court", "Ct", "Place", "Pl",
        "Circle", "Cir", "Terrace", "Ter", "Highway", "Hwy", "Parkway", "Pkwy",
    ]
    
    street = random.choice(street_names)
    st_type = random.choice(street_types)
    
    # Sometimes add apartment/suite
    if random.random() < 0.3:
        unit_types = ["Apt", "Suite", "Unit", "#", "Floor"]
        unit = f", {random.choice(unit_types)} {random.randint(1, 500)}"
    else:
        unit = ""
    
    return f"{number} {street} {st_type}{unit}", "ADDRESS"


def generate_date() -> Tuple[str, str]:
    """Generate a date in various formats."""
    # Random date in reasonable range
    start = datetime(1940, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    random_date = start + timedelta(days=random.randint(0, delta.days))
    
    formats = [
        random_date.strftime("%m/%d/%Y"),
        random_date.strftime("%m-%d-%Y"),
        random_date.strftime("%Y-%m-%d"),
        random_date.strftime("%d/%m/%Y"),
        random_date.strftime("%B %d, %Y"),
        random_date.strftime("%b %d, %Y"),
        random_date.strftime("%d %B %Y"),
        random_date.strftime("%m/%d/%y"),
    ]
    
    return random.choice(formats), "DATE"


def generate_username() -> Tuple[str, str]:
    """Generate a realistic username."""
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower()
    
    patterns = [
        f"{first}_{last}",
        f"{first}.{last}",
        f"{first}{last}{random.randint(1, 99)}",
        f"{first[0]}{last}",
        f"{first}_{random.randint(100, 9999)}",
        f"{last}_{first}",
        f"{first}{random.choice(['_', '.', ''])}{random.randint(1, 999)}",
        f"user_{random.randint(10000, 99999)}",
        f"{first[:3]}{last[:3]}{random.randint(1, 99)}",
    ]
    
    return random.choice(patterns), "USERNAME"


def generate_password() -> Tuple[str, str]:
    """Generate a realistic password."""
    # Different complexity levels
    if random.random() < 0.3:
        # Simple
        word = random.choice(FIRST_NAMES + LAST_NAMES)
        pwd = word + str(random.randint(1, 999))
    elif random.random() < 0.6:
        # Medium
        word = random.choice(FIRST_NAMES + LAST_NAMES)
        special = random.choice(['!', '@', '#', '$', '%', '&', '*'])
        pwd = word.capitalize() + str(random.randint(10, 99)) + special
    else:
        # Complex
        chars = string.ascii_letters + string.digits + "!@#$%&*"
        pwd = ''.join(random.choice(chars) for _ in range(random.randint(10, 16)))
    
    return pwd, "PASSWORD"


def generate_ip() -> Tuple[str, str]:
    """Generate a valid IPv4 address."""
    # Avoid reserved ranges
    first = random.choice([random.randint(1, 9), random.randint(11, 126), 
                          random.randint(128, 191), random.randint(192, 223)])
    ip = f"{first}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return ip, "IP_ADDRESS"


def generate_url() -> Tuple[str, str]:
    """Generate a realistic URL."""
    protocols = ["https://", "http://"]
    www = random.choice(["www.", ""])
    
    domains = [
        "example.com", "company.org", "business.net", "service.io", "app.co",
        "portal.com", "platform.io", "dashboard.net", "system.org", "api.dev",
        "docs.example.com", "mail.company.org", "secure.business.net",
    ]
    
    paths = [
        "", "/login", "/dashboard", "/api/v1", "/users", "/account", 
        "/settings", "/profile", "/data", "/reports", "/admin",
        "/products", "/services", "/contact", "/about", "/help",
    ]
    
    domain = random.choice(domains)
    path = random.choice(paths)
    protocol = random.choice(protocols)
    
    url = f"{protocol}{www}{domain}{path}"
    return url, "URL"


def generate_account_number() -> Tuple[str, str]:
    """Generate a bank account number."""
    length = random.choice([8, 10, 12])
    acct = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return acct, "ACCOUNT_NUMBER"


def generate_zip() -> Tuple[str, str]:
    """Generate a ZIP code."""
    if random.random() < 0.7:
        # 5-digit ZIP
        return f"{random.randint(10000, 99999)}", "ZIP"
    else:
        # ZIP+4
        return f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}", "ZIP"


def generate_iban() -> Tuple[str, str]:
    """Generate an IBAN-like number."""
    countries = [
        ("DE", 22), ("GB", 22), ("FR", 27), ("ES", 24), ("IT", 27), ("NL", 18),
    ]
    country, length = random.choice(countries)
    digits = ''.join([str(random.randint(0, 9)) for _ in range(length - 4)])
    iban = f"{country}{random.randint(10, 99)}{digits}"
    
    # Format with spaces
    formatted = ' '.join([iban[i:i+4] for i in range(0, len(iban), 4)])
    return formatted, "IBAN"


# =============================================================================
# CONTEXT MAP
# =============================================================================

GENERATORS = {
    "NAME_PERSON": (generate_name, NAME_CONTEXTS),
    "EMAIL": (generate_email, EMAIL_CONTEXTS),
    "PHONE": (generate_phone, PHONE_CONTEXTS),
    "SSN": (generate_ssn, SSN_CONTEXTS),
    "CREDIT_CARD": (generate_credit_card, CC_CONTEXTS),
    "ADDRESS": (generate_address, ADDRESS_CONTEXTS),
    "DATE": (generate_date, DATE_CONTEXTS),
    "USERNAME": (generate_username, USERNAME_CONTEXTS),
    "PASSWORD": (generate_password, PASSWORD_CONTEXTS),
    "IP_ADDRESS": (generate_ip, IP_CONTEXTS),
    "URL": (generate_url, URL_CONTEXTS),
    "ACCOUNT_NUMBER": (generate_account_number, ACCOUNT_CONTEXTS),
    "ZIP": (generate_zip, ZIP_CONTEXTS),
    "IBAN": (generate_iban, IBAN_CONTEXTS),
}

# Distribution weights - favor critical types
TYPE_WEIGHTS = {
    "NAME_PERSON": 2.0,
    "EMAIL": 1.5,
    "PHONE": 1.5,
    "SSN": 1.5,
    "CREDIT_CARD": 1.5,
    "ADDRESS": 1.5,
    "DATE": 1.5,
    "USERNAME": 1.0,
    "PASSWORD": 1.0,
    "IP_ADDRESS": 1.0,
    "URL": 1.0,
    "ACCOUNT_NUMBER": 0.8,
    "ZIP": 0.8,
    "IBAN": 0.8,
}


def weighted_choice(weights: Dict[str, float]) -> str:
    """Choose a type based on weights."""
    types = list(weights.keys())
    probs = [weights[t] for t in types]
    total = sum(probs)
    probs = [p / total for p in probs]
    return random.choices(types, weights=probs, k=1)[0]


def create_sample(entity_type: str) -> Dict:
    """Create a single training sample."""
    generator, contexts = GENERATORS[entity_type]
    
    # Generate entity
    entity_value, _ = generator()
    
    # Choose context
    context_template = random.choice(contexts)
    
    # Special handling for templates with extra placeholders
    if "{last4}" in context_template:
        last4 = entity_value[-4:].replace(" ", "").replace("-", "")
        context_template = context_template.replace("{last4}", last4)
    if "{city}" in context_template:
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        context_template = context_template.replace("{city}", random.choice(cities))
    if "{state}" in context_template:
        states = ["NY", "CA", "IL", "TX", "AZ", "FL", "WA", "MA"]
        context_template = context_template.replace("{state}", random.choice(states))
    
    # Build text
    text = context_template.replace("{entity}", entity_value)
    
    # Find entity position
    start = text.find(entity_value)
    if start == -1:
        # Handle edge case where @ got modified
        return None
    
    end = start + len(entity_value)
    
    return {
        "text": text,
        "entities": [{
            "label": entity_type,
            "value": entity_value,
            "start": start,
            "end": end,
            "quality_confidence": 0.99,
        }],
        "quality_score": 0.99,
        "source": "synthetic",
    }


def create_synthetic_dataset(count: int, output_path: str):
    """Create synthetic PII dataset."""
    from rich.console import Console
    from rich.progress import Progress
    from collections import Counter
    
    console = Console()
    console.print(f"\n[bold cyan]═══ Generating Synthetic PII Data ═══[/bold cyan]\n")
    
    samples = []
    type_counts = Counter()
    
    with Progress() as progress:
        task = progress.add_task("Generating...", total=count)
        
        while len(samples) < count:
            # Choose type
            entity_type = weighted_choice(TYPE_WEIGHTS)
            
            # Create sample
            sample = create_sample(entity_type)
            
            if sample:
                samples.append(sample)
                type_counts[entity_type] += 1
            
            progress.update(task, completed=len(samples))
    
    # Shuffle
    random.shuffle(samples)
    
    # Save
    output = {
        "samples": samples,
        "metadata": {
            "total_samples": len(samples),
            "source": "synthetic",
            "type_distribution": dict(type_counts),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n[bold]Type Distribution:[/bold]")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {t:20} {c:5}")
    
    console.print(f"\n[green]✓ Saved {len(samples)} samples to {output_path}[/green]\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15000)
    parser.add_argument("--output", type=str, default="synthetic_pii.json")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    create_synthetic_dataset(args.count, args.output)
